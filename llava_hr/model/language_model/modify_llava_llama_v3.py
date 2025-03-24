import torch
import torch.functional as F

from typing import Optional, List, Union, Tuple

from transformers.models.llama.modeling_llama import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
    apply_rotary_pos_emb,
    repeat_kv,
    _make_causal_mask,
    _expand_mask)

from chunkoptim.utils import SecoCache
from llava_hr.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from functools import partial


def _set_to_incomming_grad(_, incomming_grad):
    return incomming_grad


def find_boundaries(mask):
    """找到mask中0和1交替变化的边界索引"""
    boundaries = []
    if len(mask) == 0:
        return boundaries
    prev = mask[0]
    for i in range(1, len(mask)):
        if mask[i] != prev:
            boundaries.append(i)
            prev = mask[i]
    return boundaries


def prepare_inputs_labels_for_multimodal(
    self, input_ids, attention_mask, past_key_values, labels, images
):

    vision_tower = self.get_vision_tower()
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
            attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
        return input_ids, attention_mask, past_key_values, None, labels

    if type(images) is list or images.ndim == 5:
        concat_images = torch.cat([image for image in images], dim=0)
        image_features = self.encode_images(concat_images)
        split_sizes = [image.shape[0] for image in images]
        image_features = torch.split(image_features, split_sizes, dim=0)
        image_features = [x.flatten(0, 1) for x in image_features]
    else:
        image_features = self.encode_images(images)

    new_input_embeds = []
    new_labels = [] if labels is not None else None

    # =================
    new_image_mask = []
    # =================

    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
            # multimodal LLM, but the current sample is not multimodal
            # FIXME: this is a hacky fix, for deepspeed zero3 to work
            half_len = cur_input_ids.shape[0] // 2
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
            cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
            new_input_embeds.append(cur_input_embeds)
            if labels is not None:
                new_labels.append(labels[batch_idx])
            cur_image_idx += 1

            # ==============================================
            cur_new_image_mask = [0] * len(cur_input_embeds)
            new_image_mask.append(cur_new_image_mask)
            # ==============================================

            continue
        image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
        cur_new_input_embeds = []
        if labels is not None:
            cur_labels = labels[batch_idx]
            cur_new_labels = []
            assert cur_labels.shape == cur_input_ids.shape

        # =====================
        cur_new_image_mask = []
        # =====================

        while image_token_indices.numel() > 0:
            cur_image_features = image_features[cur_image_idx]
            image_token_start = image_token_indices[0]
            if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):

                # =======================
                raise NotImplementedError
                # =======================

                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                cur_new_input_embeds.append(cur_image_features)
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                    cur_labels = cur_labels[image_token_start+2:]
            else:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                cur_new_input_embeds.append(cur_image_features)

                # =================================================
                cur_new_image_mask += [0] * image_token_start
                cur_new_image_mask += [1] * len(cur_image_features)
                # =================================================

                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[image_token_start+1:]

            cur_image_idx += 1
            if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):

                # =======================
                raise NotImplementedError
                # =======================

                cur_input_ids = cur_input_ids[image_token_start+2:]
            else:
                cur_input_ids = cur_input_ids[image_token_start+1:]
                
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

        if cur_input_ids.numel() > 0:
            if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):

                # =======================
                raise NotImplementedError
                # =======================

                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
            else:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))

                # ============================================
                cur_new_image_mask += [0] * len(cur_input_ids)
                # ============================================

            if labels is not None:
                cur_new_labels.append(cur_labels)


        cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
        cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
        new_input_embeds.append(cur_new_input_embeds)

        # =======================================
        new_image_mask.append(cur_new_image_mask)
        # =======================================

        if labels is not None:
            cur_new_labels = torch.cat(cur_new_labels, dim=0)
            new_labels.append(cur_new_labels)

    if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
        max_len = max(x.shape[0] for x in new_input_embeds)

        new_input_embeds_align = []
        for cur_new_embed in new_input_embeds:
            cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
            new_input_embeds_align.append(cur_new_embed)
        new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

        # ====================================================================================================
        new_image_masks_align = []
        for cur_image_mask in new_image_mask:
            cur_image_mask += [0] * (max_len - len(cur_image_mask))
            new_image_masks_align.append(cur_image_mask)
        new_image_mask = new_image_masks_align
        # ====================================================================================================

        if labels is not None:
            new_labels_align = []
            _new_labels = new_labels
            for cur_new_label in new_labels:
                cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                new_labels_align.append(cur_new_label)
            new_labels = torch.stack(new_labels_align, dim=0)

        if attention_mask is not None:
            new_attention_mask = []
            for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                new_attention_mask.append(cur_new_attention_mask)
            attention_mask = torch.stack(new_attention_mask, dim=0)
            assert attention_mask.shape == new_labels.shape
    else:
        new_input_embeds = torch.stack(new_input_embeds, dim=0)
        if labels is not None:
            new_labels  = torch.stack(new_labels, dim=0)

        if attention_mask is not None:
            new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
            assert attention_mask.shape == new_input_embeds.shape[:2]

    # =============================================================================================
    new_image_mask = torch.tensor(new_image_mask, dtype=torch.bool, device=new_input_embeds.device)
    return None, attention_mask, past_key_values, new_input_embeds, new_labels, new_image_mask
    # =============================================================================================


@torch.no_grad()
def _build_position_ids(inputs_embeds):
    bsz, seq_len, _ = inputs_embeds.shape
    posids = torch.arange(seq_len, dtype=torch.int64, device='cuda')
    posids = posids.unsqueeze(0).view(-1, seq_len)
    return posids


@torch.no_grad()
def _build_attention_mask(inputs_embeds, attention_mask, kv_cache_length):
    bsz, seq_len, _ = inputs_embeds.shape
    combined_attention_mask = None
    if seq_len > 1:
        combined_attention_mask = _make_causal_mask(
            (bsz, seq_len),
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=kv_cache_length)

    expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=seq_len).to(inputs_embeds.device)
    
    if combined_attention_mask is None:
        combined_attention_mask = expanded_attn_mask
    else:
        combined_attention_mask = combined_attention_mask + expanded_attn_mask
    
    return combined_attention_mask


def _compute_loss(logits, labels, vocab_size, shift_label=True, is_reduce=True):
    if shift_label:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
    else:
        shift_logits = logits.contiguous()
        shift_labels = labels.contiguous()

    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    shift_labels = shift_labels.to(shift_logits.device)
    loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduce=is_reduce)
    return loss


def causal_forward(
    self,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    checkpoints: Optional[List[torch.Tensor]] = None,
    shift_label: Optional[bool] = True,
    backward: Optional[callable] = None,
    valid_label_count: Optional[int] = None,
    gather_ckpt: Optional[bool] = False,
) -> Union[Tuple, CausalLMOutputWithPast]:
    if torch.is_grad_enabled():
        """
        Backward
        """
        for ckpt in checkpoints:
            ckpt.requires_grad_(True)
        x = self.model.norm(checkpoints[-1])
        x = self.lm_head(x) 
        x = _compute_loss(x, labels, self.config.vocab_size, shift_label, valid_label_count is None)
        if valid_label_count is not None:
            x = x.sum() / valid_label_count
        backward(x)

        grad = checkpoints[-1].grad.data
        del checkpoints[-1]

        arguments = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            checkpoints=checkpoints,
            position_ids=position_ids,
            backward=backward,
            grad=grad)
        
        grad = self.model(**arguments)

        return x.detach(), grad
    
    else:
        """
        Forward
        """
        assert checkpoints is None
        arguments = dict(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            gather_ckpt=gather_ckpt)

        checkpoints = self.model(**arguments)
        return checkpoints


def model_forward(
    self,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[SecoCache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    backward: Optional[callable] = None,
    checkpoints: Optional[List[torch.Tensor]] = None,
    grad: Optional[torch.Tensor] = None,
    gather_ckpt: Optional[bool] = False
) -> Union[Tuple, BaseModelOutputWithPast]:
    
    # construct position ids and attention mask
    if position_ids is None:
        position_ids = _build_position_ids(inputs_embeds)
    if attention_mask.ndim == 2:
        past_kv_length = 0 if past_key_values is None else past_key_values.length(0)
        attention_mask = _build_attention_mask(inputs_embeds, attention_mask, past_kv_length)

    if torch.is_grad_enabled():
        for idx, decoder_layer in reversed(list(enumerate(self.layers))):
            decoder_layer(
                checkpoints[idx],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                grad=grad,
                backward=backward,
                layer_idx=idx)

            grad = checkpoints[idx].grad.data
            del checkpoints[idx]

        return grad
    else:
        hidden_states = inputs_embeds.detach()
        checkpoints = []

        for idx, decoder_layer in enumerate(self.layers):
            if gather_ckpt:
                checkpoints.append(hidden_states)
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                layer_idx=idx)
        if gather_ckpt:
            checkpoints.append(hidden_states)
        return checkpoints


def layer_forward(
    self,
    hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[SecoCache] = None,
    position_ids: Optional[torch.LongTensor] = None,
    grad: Optional[torch.Tensor] = None,
    backward: Optional[callable] = None,
    layer_idx: Optional[int] = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

    if torch.is_grad_enabled():
        # reconstruction
        r = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self.self_attn(
            hidden_states=x,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            layer_idx=layer_idx)
        x = x + r

        r = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + r

        x.register_hook(partial(_set_to_incomming_grad, incomming_grad=grad))
        backward(x.sum())
    
    else:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            layer_idx=layer_idx
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


def attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[SecoCache] = None,
    layer_idx: Optional[int] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]

    if past_key_values is not None:
        kv_seq_len += past_key_values.length(layer_idx)

    cos, sin = self.rotary_emb(value_states, seq_len=max(kv_seq_len, position_ids.max().item() + 1))
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(layer_idx, key_states, value_states)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query=query_states,
        key=key_states,
        value=value_states,
        attn_mask=attention_mask)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    
    return attn_output
