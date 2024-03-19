import os
import argparse
import torch
import torch.nn as nn
from einops import rearrange
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel,CLIPVisionModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.clip.configuration_clip import CLIPTextConfig
from transformers.models.clip.modeling_clip import CLIP_TEXT_INPUTS_DOCSTRING, _expand_mask
from typing import Optional, Tuple, Union
from models.unet import UNet3DConditionModelWaterMark
from torchvision.utils import save_image
import torchvision

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torchvision import transforms
import random
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image

import torchvision.transforms as transforms

from omegaconf import OmegaConf


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
            # import pdb
            # pdb.set_trace()
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


def validation(diffusion, unet, mapper, image_encoder, tokenizer, video_data, text_encoder, vae, cfg_scale, device, save_dir):

    normalize_exemplar = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                              (0.26862954, 0.26130258, 0.27577711))
    z = torch.randn(video_data['video'].size(0), 4, 16, 32, 32).to(device)
    image = torch.nn.functional.interpolate(video_data["masked_first_frame"].to(device), (224, 224), mode='bilinear')
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
        b, _, _, _, _ = x.shape
        x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
        x = vae.encode(x).latent_dist.sample().mul_(0.18215)
        x = rearrange(x, '(b f) c h w -> b c f h w', b=b).contiguous() # for tav unet; b c f h w is for conv3d
        exemplar_latent = x[:, :, :1, :, :].repeat(1, 1, 16, 1, 1)
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

    os.makedirs(save_dir, exist_ok=True)

    for batch_idx in range(b):
        video_ = ((samples[batch_idx] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        torchvision.io.write_video(f'{save_dir}/sampled_video.mp4', video_, fps=8)

        save_image(video_data["masked_first_frame"][batch_idx], f'{save_dir}/image_prompt.png', normalize=True, value_range=(0, 1))

    with open(f'{save_dir}/prompts.txt', 'w') as file:
        for prompt in video_data['video_name']:
            file.write(f'{prompt}\n')

    with open(f'{save_dir}/replaced_words.txt', 'w') as file:
        for word in video_data['word_prompt']:
            file.write(f'{word}\n')


def load_data_pair(config):

    img_random_trans = transforms.Compose([transforms.Resize([224, 224]),])
    first_frame_random_trans = transforms.Compose([transforms.Resize([256, 256]),])

    mask = np.array(Image.open(config.mask_path))

    first_frame = np.array(Image.open(config.img_path))

    masked_first_frame = first_frame.copy()
    masked_first_frame[mask==0] = 255

    x1, y1, x2, y2 = config.bbox

    masked_first_frame = masked_first_frame[int(y1):int(y2), int(x1):int(x2), :]

    masked_first_frame = torch.from_numpy(masked_first_frame).permute(2, 0, 1).contiguous()
    height, width = masked_first_frame.size(1), masked_first_frame.size(2)

    if height == width:
        pass
    elif height < width:
        diff = width - height
        top_pad = diff // 2
        down_pad = diff - top_pad
        left_pad = 0
        right_pad = 0
        padding_size = [left_pad, top_pad, right_pad, down_pad]
        masked_first_frame = F.pad(masked_first_frame, padding=padding_size, fill = 255)
    else:
        diff = height - width
        left_pad = diff // 2
        right_pad = diff - left_pad
        top_pad = 0
        down_pad = 0
        padding_size = [left_pad, top_pad, right_pad, down_pad]
        masked_first_frame = F.pad(masked_first_frame, padding=padding_size, fill = 255)

    masked_first_frame_ori = masked_first_frame.clone()
    masked_first_frame = img_random_trans(masked_first_frame)
    masked_first_frame = masked_first_frame / 255.0

    aug_first_frame = first_frame_random_trans(masked_first_frame_ori).unsqueeze(0) / 127.5 - 1

    video = aug_first_frame # T C H W

    return {'video': video, 'masked_first_frame': masked_first_frame}



def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    config  = OmegaConf.load(args.config)

    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pretrained_t2v_model_path = './pretrained_models/watermark_remove_module.pt'
    unet = UNet3DConditionModelWaterMark.from_pretrained_2d(pretrained_model_path='./pretrained/stable-diffusion-v1-4', subfolder="unet").to(device)
    state_dict = torch.load(pretrained_t2v_model_path,  map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))["ema_unet"]
    unet.load_state_dict(state_dict, strict=False)

    num_sampling_steps = 250
    diffusion = create_diffusion(str(num_sampling_steps))  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
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

        if _module.__class__.__name__ == "SparseCausalAttention":

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

    mapper_path = './pretrained_models/mapper.pt'
    state_dict = torch.load(mapper_path, map_location='cpu')["ema"]
    mapper.load_state_dict(state_dict)
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


    load_data = load_data_pair(config)
    video_data = {}
    video_data['video'] = load_data['video'].unsqueeze(0)
    video_data['video_name'] = [config.text_prompt.lower().replace(f'{config.replace_word.lower()}', 'sdksd')]

    video_data['word_prompt'] = ['sdksd']
    video_data['masked_first_frame'] = load_data['masked_first_frame'].unsqueeze(0)

    set_random_seed(config.seed)
    validation(diffusion, unet, mapper, image_encoder, tokenizer, video_data, text_encoder, vae, 4.0, device, f'./sample_results')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()
    main(args)
