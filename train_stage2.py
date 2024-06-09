# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""


import os
import torch
import argparse
import json

import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
from glob import glob
from time import time
from copy import deepcopy
from einops import rearrange
from datasets import get_dataset
from diffusion import create_diffusion
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils import (clip_grad_norm_, create_logger, update_ema,
                   requires_grad, cleanup, create_tensorboard, write_tensorboard)
from transformers import CLIPTokenizer, CLIPTextModel,CLIPVisionModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.clip.configuration_clip import CLIPTextConfig
from transformers.models.clip.modeling_clip import CLIP_TEXT_INPUTS_DOCSTRING, _expand_mask
from typing import Optional, Tuple, Union
import torchvision.transforms as transforms

from models.unet import UNet3DConditionModel
from torchvision.utils import save_image
import torchvision
import numpy as np

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
# More information can be found at
# https://www.zhihu.com/question/573022357/answer/2807382990?utm_campaign=shareopn&utm_medium=social&utm_oi
# =729778080528228352&utm_psn=1629783739608895488&utm_source=wechat_session

# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices

# Maybe use fp16 percision training need to set to False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Mapper to transform the clip image embedding to clip text embedding
class Mapper(nn.Module):
    def __init__(self,
        input_dim: int,
        output_dim: int,
    ):
        super(Mapper, self).__init__()

        for i in range(5):
            setattr(self, f'mapping_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                         nn.LayerNorm(1024),
                                         nn.LeakyReLU(),
                                         nn.Linear(1024, 1024),
                                         nn.LayerNorm(1024),
                                         nn.LeakyReLU(),
                                         nn.Linear(1024, output_dim)))

            setattr(self, f'mapping_patch_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, output_dim)))

    def forward(self, embs):
        hidden_states = ()
        for i, emb in enumerate(embs):
            hidden_state = getattr(self, f'mapping_{i}')(emb[:, :1]) + getattr(self, f'mapping_patch_{i}')(emb[:, 1:]).mean(dim=1, keepdim=True)
            # print(hidden_state.size())
            hidden_states += (hidden_state, )
        hidden_states = torch.cat(hidden_states, dim=1)

        return hidden_states

def _build_causal_attention_mask(bsz, seq_len, dtype):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
    mask.fill_(torch.tensor(torch.finfo(dtype).min))
    mask.triu_(1)  # zero out the lower diagonal
    mask = mask.unsqueeze(1)  # expand mask
    return mask

# replace some operations in the text encoder
@add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)
def inj_forward_text(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPooling]:
    r"""
    Returns:
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is None:
        raise ValueError("You have to specify either input_ids")

    # original input token ids
    raw_input_ids = input_ids['input_ids']
    if 'inj_embedding' in input_ids:
        inj_embedding = input_ids['inj_embedding']
        inj_index = input_ids['inj_index']
    else:
        inj_embedding = None
        inj_index = None

    input_shape = raw_input_ids.size()
    r_input_ids = raw_input_ids.view(-1, input_shape[-1])

    inputs_embeds = self.embeddings.token_embedding(r_input_ids)
    new_inputs_embeds = inputs_embeds.clone()
    if inj_embedding is not None:
        emb_length = inj_embedding.shape[1]
        for bsz, idx_list in enumerate(inj_index):
            start_idx_list = torch.where(raw_input_ids[bsz] == idx_list[0])[0]
            for start_idx in start_idx_list:
                end_idx = start_idx + len(idx_list) - 1
                if len(idx_list) > emb_length:
                    lll = new_inputs_embeds[bsz, end_idx + 1:].shape[0]
                    try:
                        new_inputs_embeds[bsz, start_idx+emb_length:] = torch.cat([inputs_embeds[bsz, end_idx+1:end_idx+1+lll], inputs_embeds[bsz, -(len(idx_list) - emb_length):]], dim=0)
                    except:
                        print(f'Index Error: point1, {start_idx}, {end_idx}, {new_inputs_embeds[bsz, start_idx+emb_length:].size()}, {inputs_embeds[bsz, end_idx+1:end_idx+1+lll].size()}, {inputs_embeds[bsz, -(len(idx_list) - emb_length):].size()}')
                else:
                    lll = new_inputs_embeds[bsz, start_idx+emb_length:].shape[0]
                    try:
                        new_inputs_embeds[bsz, start_idx+emb_length:] = inputs_embeds[bsz, end_idx+1:end_idx+1+lll]
                    except:
                        print(f'Index Error: point2, {start_idx}, {end_idx}, {new_inputs_embeds[bsz, start_idx+emb_length:].size()}, {inputs_embeds[bsz, end_idx+1:end_idx+1+lll].size()}')
                try:
                    new_inputs_embeds[bsz, start_idx:start_idx+emb_length] = inj_embedding[bsz]
                except:
                    remain_length = new_inputs_embeds[bsz, start_idx:start_idx+emb_length].size(0)
                    new_inputs_embeds[bsz, start_idx:start_idx+emb_length] = inj_embedding[bsz, :remain_length]

    hidden_states = self.embeddings(input_ids=None, position_ids=position_ids, inputs_embeds=new_inputs_embeds)

    bsz, seq_len = input_shape
    # CLIP's text model uses causal mask, prepare it here.
    # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
    causal_attention_mask = _build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
        hidden_states.device
    )
    # expand attention_mask
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = self.final_layer_norm(last_hidden_state)

    # text_embeds.shape = [batch_size, sequence_length, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=r_input_ids.device), r_input_ids.to(torch.int).argmax(dim=-1)
    ]

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )

def inj_forward_crossattention(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
    batch_size, sequence_length, _ = hidden_states.shape

    encoder_hidden_states = encoder_hidden_states

    if self.group_norm is not None:
        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = self.to_q(hidden_states)
    dim = query.shape[-1]
    query = self.reshape_heads_to_batch_dim(query)

    if encoder_hidden_states is not None:
        key = self.to_k_global(encoder_hidden_states)
        value = self.to_v_global(encoder_hidden_states)
    else:
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

    key = self.reshape_heads_to_batch_dim(key)
    value = self.reshape_heads_to_batch_dim(value)

    if attention_mask is not None:
        if attention_mask.shape[-1] != query.shape[1]:
            target_length = query.shape[1]
            attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
            attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

    # attention, what we cannot get enough of
    if self._use_memory_efficient_attention_xformers:
        hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
        # Some versions of xformers return output in fp32, cast it back to the dtype of the input
        hidden_states = hidden_states.to(query.dtype)
    else:
        if self._slice_size is None or query.shape[0] // self._slice_size == 1:
            hidden_states = self._attention(query, key, value, attention_mask)
        else:
            hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)

    # dropout
    hidden_states = self.to_out[1](hidden_states)
    return hidden_states


def inj_forward_stattention(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, exemplar_latent=None):
    batch_size, sequence_length, _ = hidden_states.shape

    encoder_hidden_states = encoder_hidden_states

    if self.group_norm is not None:
        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = self.to_q(hidden_states)
    dim = query.shape[-1]
    query = self.reshape_heads_to_batch_dim(query)

    if self.added_kv_proj_dim is not None:
        raise NotImplementedError

    encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
    key = self.to_k_new(encoder_hidden_states)
    value = self.to_v_new(encoder_hidden_states)

    former_frame_index = torch.arange(video_length) - 1
    former_frame_index[0] = 0

    if exemplar_latent is not None:
        exemplar_key = self.to_k_exemplar(exemplar_latent)
        exemplar_value = self.to_v_exemplar(exemplar_latent)

        query_frames = self.reshape_batch_dim_to_heads(query)
        query_frames = rearrange(query_frames, "(b f) d c -> b f d c", f=video_length)
        key_frames = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        value_frames = rearrange(value, "(b f) d c -> b f d c", f=video_length)

        # update the first key
        query_0 = query_frames[:, 0]
        query_0 = self.reshape_heads_to_batch_dim(query_0)

        key_0 = key_frames[:, :1]
        exemplar_key = rearrange(exemplar_key, "(b f) d c -> b f d c", f=video_length)[:, :1]
        key_for_update_0 = torch.cat([exemplar_key, key_0], dim=2)

        value_0 = value_frames[:, :1]
        exemplar_value = rearrange(exemplar_value, "(b f) d c -> b f d c", f=video_length)[:, :1]
        value_for_update_0 = torch.cat([exemplar_value, value_0], dim=2)

        key_for_update_0 = rearrange(key_for_update_0, "b f d c -> (b f) d c")
        value_for_update_0 = rearrange(value_for_update_0, "b f d c -> (b f) d c")

        key_for_update_0 = self.reshape_heads_to_batch_dim(key_for_update_0)
        value_for_update_0 = self.reshape_heads_to_batch_dim(value_for_update_0)
        updated_0 = self._attention(query_0, key_for_update_0, value_for_update_0, attention_mask)

        updated_0 = rearrange(updated_0, "(b f) d c -> b f d c", f=1)

        # original cross frame attention
        key = torch.cat([key_frames[:, [0] * video_length], key_frames[:, former_frame_index]], dim=2)
        key = rearrange(key, "b f d c -> (b f) d c")

        value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        updated_frames = torch.cat([updated_0, value[:, 1:]], dim= 1)
        value = torch.cat([updated_frames[:, [0] * video_length], updated_frames[:, former_frame_index]], dim=2)
        value = rearrange(value, "b f d c -> (b f) d c")
    else:
        # raise NotImplementedError
        key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        key = torch.cat([key[:, [0] * video_length], key[:, former_frame_index]], dim=2)
        key = rearrange(key, "b f d c -> (b f) d c")

        value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        value = torch.cat([value[:, [0] * video_length], value[:, former_frame_index]], dim=2)
        value = rearrange(value, "b f d c -> (b f) d c")

    key = self.reshape_heads_to_batch_dim(key)
    value = self.reshape_heads_to_batch_dim(value)

    if attention_mask is not None:
        if attention_mask.shape[-1] != query.shape[1]:
            target_length = query.shape[1]
            attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
            attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

    # attention, what we cannot get enough of
    if self._use_memory_efficient_attention_xformers:
        hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
        # Some versions of xformers return output in fp32, cast it back to the dtype of the input
        hidden_states = hidden_states.to(query.dtype)
    else:
        if self._slice_size is None or query.shape[0] // self._slice_size == 1:
            hidden_states = self._attention(query, key, value, attention_mask)
        else:
            hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)

    # dropout
    hidden_states = self.to_out[1](hidden_states)
    return hidden_states

def freeze_params(params):
    for param in params:
        param.requires_grad = False

def unfreeze_params(params):
    for param in params:
        param.requires_grad = True

def validation(diffusion, unet, mapper, image_encoder, tokenizer, video_data, text_encoder, vae, cfg_scale, device, save_dir):

    normalize_exemplar = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                              (0.26862954, 0.26130258, 0.27577711))
    z = torch.randn(video_data['video'].size(0), 4, args.num_frames, 32, 32).to(device)
    image = F.interpolate(video_data["masked_first_frame"].to(device), (224, 224), mode='bilinear')
    image = normalize_exemplar(image)

    # get the image embeddings
    image_features = image_encoder(image, output_hidden_states=True)
    image_embeddings = [image_features[0], image_features[2][4], image_features[2][8], image_features[2][12], image_features[2][16]]
    image_embeddings = [emb.detach() for emb in image_embeddings]
    inj_embedding = mapper(image_embeddings)

    original_ids = tokenizer(
        video_data['video_name'],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )["input_ids"].to(device)

    placeholder_idx = tokenizer(video_data['word_prompt'], add_special_tokens=False)["input_ids"]
    # Get the text embedding for conditioning
    encoder_hidden_states_con = text_encoder({'input_ids': original_ids,
                                        "inj_embedding": inj_embedding,
                                        "inj_index": placeholder_idx})[0]

    # get null text embedding
    null_ids = tokenizer(["None"] * video_data['video'].size(0),
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )["input_ids"].to(device)
    encoder_hidden_states_uncon = text_encoder({'input_ids': null_ids})[0]
    encoder_hidden_states = torch.cat([encoder_hidden_states_con, encoder_hidden_states_uncon], dim=0)

    exemplar_encoder_hidden_states = torch.cat([encoder_hidden_states_uncon, encoder_hidden_states_uncon], dim=0)
    # get the exemplar_latent
    x = video_data['video'].to(device)
    with torch.no_grad():
        # Map input images to latent space + normalize latents:
        b, _, _, _, _ = x.shape
        x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
        x = vae.encode(x).latent_dist.sample().mul_(0.18215)
        x = rearrange(x, '(b f) c h w -> b c f h w', b=b).contiguous() # for tav unet; b c f h w is for conv3d
        exemplar_latent = x[:, :, :1, :, :].repeat(1, 1, 16, 1, 1)
        # TODO: check if is promper to provide cfg like this
        exemplar_latent = torch.cat([exemplar_latent, exemplar_latent], 0)

    model_kwargs = dict(encoder_hidden_states=encoder_hidden_states, class_labels=None, cfg_scale=cfg_scale, exemplar_latent_ori = exemplar_latent, add_noise_to_exemplar=True, exemplar_encoder_hidden_states=exemplar_encoder_hidden_states)

    z = torch.cat([z, z], 0)

    samples = diffusion.p_sample_loop(
        unet.forward_with_cfg_with_exemplar, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device)

    samples, _ = samples.chunk(2, dim=0)

    b, f, c, h, w = samples.shape
    samples = rearrange(samples, 'b c f h w -> (b f) c h w')
    samples = vae.decode(samples / 0.18215).sample
    samples = rearrange(samples, '(b f) c h w -> b f c h w', b=b)

    for batch_idx in range(b):
        video_dir = f'{save_dir}/video_{batch_idx}'
        ori_video_dir = f'{save_dir}/ori_video_{batch_idx}'
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(ori_video_dir, exist_ok=True)
        for frame_idx in range(args.num_frames):
            save_image(samples[batch_idx][frame_idx], f'{video_dir}/{frame_idx:04d}.png', normalize=True, value_range=(-1, 1))
            save_image(video_data['video'][batch_idx][frame_idx], f'{ori_video_dir}/{frame_idx:04d}.png', normalize=True, value_range=(-1, 1))
        video_ = ((samples[batch_idx] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        torchvision.io.write_video(f'{save_dir}/video_{batch_idx}.mp4', video_, fps=8)

        save_image(video_data["masked_first_frame"][batch_idx], f'{video_dir}/image_prompt.png', normalize=True, value_range=(0, 1))

    with open(f'{save_dir}/prompts.txt', 'w') as file:
        for prompt in video_data['video_name']:
            file.write(f'{prompt}\n')

    with open(f'{save_dir}/replaced_words.txt', 'w') as file:
        for word in video_data['word_prompt']:
            file.write(f'{word}\n')

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        num_frame_string = 'F' + str(args.num_frames) + 'S' + str(args.frame_interval)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{num_frame_string}-{args.dataset}"  # Create an experiment folder
        # if args.use_lora:
        #     experiment_dir = experiment_dir + '-lora'
        if args.class_guided:
            experiment_dir = experiment_dir + '-Class' # conditional generation
        if args.use_compile:
            experiment_dir = experiment_dir + '-Compile' # speedup by torch compile
        if args.use_timecross_transformer:
            experiment_dir = experiment_dir + '-TimeCross'
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        tb_writer = create_tensorboard(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        with open(f'{experiment_dir}/config.json', 'wt') as f:
            json.dump(vars(args), f, indent=4)
    else:
        logger = create_logger(None)
        tb_writer = None

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    args.latent_size = latent_size

    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path='./pretrained/stable-diffusion-v1-4', subfolder="unet").to(device)
    state_dict = torch.load(args.pretrained_t2v_model,  map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))["ema"]
    unet.load_state_dict(state_dict)

    # print(unet)

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    # define tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # modify the text encoder
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    for _module in text_encoder.modules():
        if _module.__class__.__name__ == "CLIPTextTransformer":
            _module.__class__.__call__ = inj_forward_text

    image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

    mapper = Mapper(input_dim=1024, output_dim=768).to(device)

    # replace the forward method of the crossattention to finetune the to_k and to_v layers
    for _name, _module in unet.named_modules():
        if _module.__class__.__name__ == "CrossAttention":
            # print(_name, _module.__class__.__name__)
            if 'attn1' in _name:
                continue
            if 'attn_temp' in _name:
                continue

            _module.__class__.forward = inj_forward_crossattention

            shape = _module.to_k.weight.shape
            to_k_global = nn.Linear(shape[1], shape[0], bias=False)
            to_k_global.weight.data = _module.to_k.weight.data.clone()
            mapper.add_module(f'{_name.replace(".", "_")}_to_k', to_k_global)

            shape = _module.to_v.weight.shape
            to_v_global = nn.Linear(shape[1], shape[0], bias=False)
            to_v_global.weight.data = _module.to_v.weight.data.clone()
            mapper.add_module(f'{_name.replace(".", "_")}_to_v', to_v_global)

            if args.global_mapper_path is None:
                _module.add_module('to_k_global', to_k_global)
                _module.add_module('to_v_global', to_v_global)

        if _module.__class__.__name__ == "SparseCausalAttention":
            # print(_name, _module.__class__.__name__)
            if 'attn2' in _name:
                continue
            if 'attn_temp' in _name:
                continue

            _module.__class__.forward = inj_forward_stattention

            shape = _module.to_k.weight.shape
            to_k_exemplar = nn.Linear(shape[1], shape[0], bias=False)
            to_k_exemplar.weight.data = _module.to_k.weight.data.clone()
            mapper.add_module(f'{_name.replace(".", "_")}_to_k_exemplar', to_k_exemplar)

            shape = _module.to_v.weight.shape
            to_v_exemplar = nn.Linear(shape[1], shape[0], bias=False)
            to_v_exemplar.weight.data = _module.to_v.weight.data.clone()
            mapper.add_module(f'{_name.replace(".", "_")}_to_v_exemplar', to_v_exemplar)

            shape = _module.to_k.weight.shape
            to_k_new = nn.Linear(shape[1], shape[0], bias=False)
            to_k_new.weight.data = _module.to_k.weight.data.clone()
            mapper.add_module(f'{_name.replace(".", "_")}_to_k_new', to_k_new)

            shape = _module.to_v.weight.shape
            to_v_new = nn.Linear(shape[1], shape[0], bias=False)
            to_v_new.weight.data = _module.to_v.weight.data.clone()
            mapper.add_module(f'{_name.replace(".", "_")}_to_v_new', to_v_new)

            if args.global_mapper_path is None:
                _module.add_module('to_k_exemplar', to_k_exemplar)
                _module.add_module('to_v_exemplar', to_v_exemplar)

                _module.add_module('to_k_new', to_k_new)
                _module.add_module('to_v_new', to_v_new)

    if args.global_mapper_path is not None:
        state_dict = torch.load(args.global_mapper_path, map_location='cpu')['ema']
        for k, v in mapper.state_dict().items():
            if 'to_k_exemplar' in k:
                state_dict[k] = v
            if '_to_v_exemplar' in k:
                state_dict[k] = v

            if 'to_k_new' in k:
                state_dict[k] = v
            if '_to_v_new' in k:
                state_dict[k] = v

        mapper.load_state_dict(state_dict)
        logger.info(f"Load Pretrained Mapper v2.")
        for _name, _module in unet.named_modules():
            if _module.__class__.__name__ == "CrossAttention":
                if 'attn1' in _name:
                    continue
                if 'attn_temp' in _name:
                    continue
                _module.add_module('to_k_global', getattr(mapper, f'{_name.replace(".", "_")}_to_k'))
                _module.add_module('to_v_global', getattr(mapper, f'{_name.replace(".", "_")}_to_v'))

            if _module.__class__.__name__ == "SparseCausalAttention":
                if 'attn2' in _name:
                    continue
                if 'attn_temp' in _name:
                    continue

                _module.add_module('to_k_exemplar', getattr(mapper, f'{_name.replace(".", "_")}_to_k_exemplar'))
                _module.add_module('to_v_exemplar', getattr(mapper, f'{_name.replace(".", "_")}_to_v_exemplar'))

                _module.add_module('to_k_new', getattr(mapper, f'{_name.replace(".", "_")}_to_k_new'))
                _module.add_module('to_v_new', getattr(mapper, f'{_name.replace(".", "_")}_to_v_new'))

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(mapper).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    # Freeze vae and unet, encoder
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    freeze_params(text_encoder.parameters())
    freeze_params(image_encoder.parameters())

    # Unfreeze the mapper
    unfreeze_params(mapper.parameters())
    mapper = DDP(mapper.to(device), device_ids=[rank])

    logger.info(f"Model Parameters: {sum(p.numel() for p in mapper.parameters() if p.requires_grad):,}")
    opt = torch.optim.AdamW(mapper.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    dataset = get_dataset(args)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} videos.")

    # Prepare models for training:
    update_ema(ema, mapper.module, decay=0)  # Ensure EMA is initialized with synced weights
    mapper.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_loss_ldm = 0
    running_loss_reg = 0
    running_loss_reg_text = 0
    start_time = time()

    # create diffusion for validation
    diffusion = create_diffusion(str(args.num_sampling_steps))

    normalize_exemplar = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                              (0.26862954, 0.26130258, 0.27577711))

    # logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        # logger.info(f"Beginning epoch {epoch}...")
        for video_data in loader:
            x = video_data['video'].to(device)
            video_name = video_data['video_name']
            # x = x.to(device)
            # y = y.to(device) # y is text prompt; no need put in gpu
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                b, _, _, _, _ = x.shape
                x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                x = rearrange(x, '(b f) c h w -> b c f h w', b=b).contiguous() # for tav unet; b c f h w is for conv3d
                exemplar_latent = x[:, :, :1, :, :].repeat(1, 1, 16, 1, 1)
                x = x[:, :, 1:, :, :]

                # get zero exemplar
                zero_exemplar = torch.zeros([1, 16, 3, 256, 256], dtype=x.dtype, device=device)
                zero_exemplar = rearrange(zero_exemplar, 'b f c h w -> (b f) c h w').contiguous()
                zero_exemplar = vae.encode(zero_exemplar).latent_dist.sample().mul_(0.18215)
                zero_exemplar = rearrange(zero_exemplar, '(b f) c h w -> b c f h w', b=1).contiguous()

            # TODO: why should we resize again?
            image = F.interpolate(video_data["masked_first_frame"].to(device), (224, 224), mode='bilinear')
            image = normalize_exemplar(image)

            # get the image embeddings
            image_features = image_encoder(image, output_hidden_states=True)
            image_embeddings = [image_features[0], image_features[2][4], image_features[2][8], image_features[2][12], image_features[2][16]]
            image_embeddings = [emb.detach() for emb in image_embeddings]
            inj_embedding = mapper(image_embeddings)

            original_ids = tokenizer(
                video_data['video_name'],
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )["input_ids"].to(device)

            if args.reg_text_weight > 0:
                original_encoder_hidden_states = text_encoder({'input_ids': original_ids})[0].detach()

            placeholder_idx = tokenizer(video_data['word_prompt'], add_special_tokens=False)["input_ids"]
            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder({'input_ids': original_ids,
                                                "inj_embedding": inj_embedding,
                                                "inj_index": placeholder_idx})[0]

            drop_ids = np.random.uniform(0, 1, encoder_hidden_states.size(0)) < args.dropout_prob
            null_ids = tokenizer(["None"] * 1,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )["input_ids"].to(device)
            encoder_hidden_states_uncon = text_encoder({'input_ids': null_ids})[0]
            encoder_hidden_states[drop_ids, :, :] = encoder_hidden_states_uncon

            # replace the exemplar_latent
            exemplar_latent[drop_ids, :, :, :, :] = zero_exemplar

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            noise = torch.randn_like(exemplar_latent)
            exemplar_latent_t = diffusion.q_sample(exemplar_latent, t, noise=noise)
            map_tensor = torch.tensor(diffusion.timestep_map, device=t.device, dtype=t.dtype)
            new_ts = map_tensor[t]
            # model_kwargs = dict(context=text_embedding, y=None)

            null_ids = tokenizer(["None"] * video_data['video'].size(0),
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )["input_ids"].to(device)
            exemplar_encoder_hidden_states = text_encoder({'input_ids': null_ids})[0]
            model_kwargs = dict(encoder_hidden_states=encoder_hidden_states, class_labels=None, exemplar_latent = exemplar_latent_t, exemplar_timestep = new_ts, exemplar_encoder_hidden_states=exemplar_encoder_hidden_states) # tav unet

            loss_dict = diffusion.training_losses(unet, x, t, model_kwargs)
            loss_ldm = loss_dict["loss"].mean()
            loss_reg = torch.mean(torch.abs(inj_embedding)) * args.reg_weight
            if args.reg_text_weight:
                loss_reg_text = torch.mean(torch.abs(encoder_hidden_states - original_encoder_hidden_states)) * args.reg_text_weight
                loss = loss_ldm + loss_reg + loss_reg_text
            else:
                loss_reg_text = 0
                loss = loss_ldm + loss_reg
            opt.zero_grad()
            loss.backward()

            if args.clip_max_norm:
                gradient_norm = clip_grad_norm_(mapper.module.parameters(), args.clip_max_norm, clip_grad=True)

            opt.step()
            update_ema(ema, mapper.module)

            # Log loss values:
            running_loss += loss.item()
            running_loss_ldm += loss_ldm.item()
            running_loss_reg += loss_reg.item()
            if args.reg_text_weight > 0:
                running_loss_reg_text += loss_reg_text.item()
            else:
                running_loss_reg_text += 0
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss_ldm = torch.tensor(running_loss_ldm / log_steps, device=device)
                avg_loss_reg = torch.tensor(running_loss_reg / log_steps, device=device)
                avg_loss_reg_text = torch.tensor(running_loss_reg_text / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                # logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                logger.info(f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Loss ldm: {avg_loss_ldm:.4f}, Loss reg: {avg_loss_reg:.4f}, Loss reg text: {avg_loss_reg_text:.4f}, Gradient Norm: {gradient_norm:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                write_tensorboard(tb_writer, 'Train Loss', avg_loss, train_steps)
                write_tensorboard(tb_writer, 'Loss ldm', avg_loss_ldm, train_steps)
                write_tensorboard(tb_writer, 'Loss reg', avg_loss_reg, train_steps)
                write_tensorboard(tb_writer, 'Loss reg text', avg_loss_reg_text, train_steps)
                write_tensorboard(tb_writer, 'Gradient Norm', gradient_norm, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                running_loss_ldm = 0
                running_loss_reg = 0
                running_loss_reg_text = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                            "model": mapper.module.state_dict(),
                            "ema": ema.state_dict(),
                            # "opt": opt.state_dict(),
                            # "args": args
                        }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                    validation(diffusion, unet, mapper, image_encoder, tokenizer, video_data, text_encoder, vae, args.cfg_scale, device, f'{experiment_dir}/validation/iters_{train_steps}')
                dist.barrier()

    mapper.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=3407)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=20000)
    # added by maxin
    parser.add_argument("--class-guided", default=False, action='store_true')
    parser.add_argument("--use-timecross-transformer", default=False, action='store_true')
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--num-frames", type=int, default=16, help='video frames for training')
    parser.add_argument("--frame-interval", type=int, default=1, help='video frames interval')
    parser.add_argument("--attention-mode", default='math', type=str, help='which attention used')
    parser.add_argument("--dataset", type=str, default='ffs', help='dataset for training')
    parser.add_argument("--clip-max-norm", default=None, type=float, help='clip gradient')
    parser.add_argument("--use-compile", default=False, action='store_true', help='speedup by torch compile')
    parser.add_argument("--global-mapper-path", type=str, default=None)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--reg-weight", type=float, default=0.01)
    parser.add_argument("--reg-text-weight", type=float, default=0.01)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--pretrained-t2v-model", type=str, required=True)
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
