import torch
import torch.functional as F
from torch import nn

from typing import Optional, List, Union, Tuple

from transformers.models.llama.modeling_llama import (
    CausalLMOutputWithPast, 
    CrossEntropyLoss,
    BaseModelOutputWithPast,
    logger,
    apply_rotary_pos_emb,
    repeat_kv,
    _make_causal_mask,
    _expand_mask)

import math


import numpy as np
import matplotlib.pyplot as plt
import os

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


def encode_images(self, images):
    image_features = self.get_model().get_vision_tower()(images)
    image_features = self.get_model().mm_projector(image_features)
    return image_features


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


def causal_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    images: Optional[torch.FloatTensor] = None,
    return_dict: Optional[bool] = None,
    image_masks: Optional[List[List[int]]] = None,
    shift_label: Optional[bool] = True,
    is_reduce: Optional[bool] = True,
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # ======================================================================================
    if inputs_embeds is None:
        (
            input_ids, 
            attention_mask, 
            past_key_values, 
            inputs_embeds, 
            labels, 
            image_masks
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, 
            attention_mask, 
            past_key_values, 
            labels, 
            images)

    arguments = dict(image_masks=image_masks) if hasattr(self, 'modify_version') else dict()
    arguments.update(dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict))

    outputs = self.model(**arguments)
    # ======================================================================================

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:

        # =================================================
        if shift_label:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        else:
            shift_logits = logits.contiguous()
            shift_labels = labels.contiguous()
        # =================================================

        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)

        # Enable model/pipeline parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        # ====================================================================================
        loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduce=is_reduce)
        # ====================================================================================

    return CausalLMOutputWithPast(loss=loss)


@torch.no_grad()
def model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    image_masks: Optional[List[List[int]]] = None
) -> Union[Tuple, BaseModelOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    # embed positions
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        )


    # ===================================================================================
    # attention_mask = self._prepare_decoder_attention_mask(
    #     attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    # )
    # ===================================================================================


    # =============================================================================================================
    combined_attention_mask = None
    if seq_length > 1:
        combined_attention_mask = _make_causal_mask(
            (batch_size, seq_length),
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=seq_length).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )
    
    attention_mask = combined_attention_mask
    # =============================================================================================================


    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = ()
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, True, None)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                position_ids,
                None,
                image_masks
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=True,
                use_cache=use_cache,
                image_masks=image_masks,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2],)

        all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)


    # ===================================================================================================
    # 转换list为numpy数组
    image_masks_np = np.array(image_masks)  # shape (32, 606)

    # 创建输出目录
    output_dir = "attention_visualization"
    os.makedirs(output_dir, exist_ok=True)

    for layer_idx in range(len(self.layers)):
        attention_head0 = all_self_attns[layer_idx][:, layer_idx].cpu().float().numpy()
        gains = np.arange(attention_head0.shape[1]) + 1
        gains = gains[None, :, None]
        attention_head0 *= gains

        os.makedirs(os.path.join("attention_visualization", f"layer-{layer_idx}"), exist_ok=True)

        for sample_idx in range(len(image_masks_np)):
            # 获取当前样本的mask和attention map
            mask = image_masks_np[sample_idx]
            attention = attention_head0[sample_idx]
            
            # 找到边界点
            boundaries = find_boundaries(mask)
            
            # 创建画布
            plt.figure(figsize=(12, 10))
            
            # 绘制attention map
            plt.imshow(attention, cmap='viridis', aspect='auto')
            plt.colorbar()
            
            # 绘制分界线（红色虚线）
            for b in boundaries:
                plt.axvline(b-0.5, color='red', linestyle='--', linewidth=2)
                plt.axhline(b-0.5, color='red', linestyle='--', linewidth=2)
            
            # 设置标题和坐标轴
            plt.title(f"Sample {sample_idx} - Layer {layer_idx} - Head 0\n{len(boundaries)} boundaries")
            plt.xlabel("Key Position")
            plt.ylabel("Query Position")
            
            # 保存并关闭
            plt.savefig(os.path.join(output_dir, f"layer-{layer_idx}", f"sample_{sample_idx}_head0.png"), 
                        bbox_inches='tight', dpi=150)
            plt.close()

    print(f"可视化结果已保存至：{os.path.abspath(output_dir)}")
    # ===================================================================================================

    raise NotImplementedError

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    image_masks: Optional[List[List[int]]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        image_masks=image_masks
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    image_masks: Optional[List[List[int]]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    bsz, q_len, _ = hidden_states.size()

    if self.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    # ========================================
    attn_weights_record = attn_weights.clone()
    # ========================================

    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights_record, past_key_value