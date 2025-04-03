import types
import torch
import torch.distributed as dist

from functools import partial
from .utils import (
    _set_to_incomming_grad,
    _backward,
    # _maybe_align_length_across_gpus,
    _first_forward_prop_seco,
    _chunkize_inputs,
    _prepare_inputs,
    # _sample_chunks,
    _step,
    _construct_4d_mask,
    _sample_tokens_v2
)


def reset_training_step(trainer):

    # v1 training step function
    func_mapper = {
        # profile
        'v1-profile-parallel': _profile_parallel,
        'v1-profile-seco': _profile_seco,
        'v1-profile-spaco': _profile_spaco,
        'v1-profile-spaco2': _profile_spaco2,
        # train
        'v1-seco': _seco,
        'v1-spaco': _spaco,
        'v1-spaco2': _spaco2,
        }
    
    # v2 training step function
    func_mapper.update({
        # train
        'v2': _sparse_mlp,
        'v2-profile': _profile_sparse_mlp
    })

    if hasattr(trainer.model, "modify_version") and trainer.model.modify_version in func_mapper:
        step_func = func_mapper[trainer.model.modify_version]
        trainer.training_step = types.MethodType(step_func, trainer)


def _seco(self, model, inputs):
    fwd_chunk_size = 512
    bwd_chunk_size = 512
    valid_label_count = (inputs['labels'] != -100).sum()

    # prepare inputs
    attention_mask, inputs_embeds, labels, _ = _prepare_inputs(model, inputs)

    # align length across gpus
    inputs_embeds, labels, attention_mask, _ = _maybe_align_length_across_gpus(
        inputs_embeds, 
        labels, 
        attention_mask,
        None)

    inputs_embeds_detach = inputs_embeds.detach()
    inputs_embeds_detach.requires_grad_(True)
    labels = torch.cat((labels[:, 1:], torch.full_like(labels[:, :1], fill_value=-100)), dim=-1)

    # chunkize inputs
    inputs_embeds_list, labels_list, masks_list = _chunkize_inputs(
        [inputs_embeds_detach, labels, attention_mask],
        chunk_size=fwd_chunk_size)

    # LLM forward prop
    accum_loss, seco_cache = _first_forward_prop_seco(
        model, 
        inputs_embeds_list, 
        labels_list, 
        masks_list, 
        valid_label_count,
        cpu_offload=1)

    # maybe change chunk size before backward prop
    if fwd_chunk_size != bwd_chunk_size:
        seco_cache.reorganize(bwd_chunk_size)
        inputs_embeds_list, labels_list, masks_list = _chunkize_inputs(
            [inputs_embeds_detach, labels, attention_mask],
            chunk_size=bwd_chunk_size,)

    # LLM backward prop
    generator = reversed(list(enumerate(zip(inputs_embeds_list, labels_list))))
    for i, (chunk_embeds, chunk_labels) in generator:
        tmp_cache = seco_cache.index(i)
        seco_cache.delete(i)
        outputs = model(
            input_ids=None,
            attention_mask=torch.cat(masks_list[:i+1], dim=1),
            inputs_embeds=chunk_embeds,
            labels=chunk_labels,
            past_key_values=seco_cache,
            shift_label=False,
            is_reduce=False)
        loss = outputs['loss'].sum() / valid_label_count
        seco_cache.link_grad(tmp_cache, i)
        _backward(loss, model)
        seco_cache.delete(i)
        del tmp_cache

    inputs_embeds.register_hook(partial(
        _set_to_incomming_grad, 
        incomming_grad=inputs_embeds_detach.grad))
    _backward(inputs_embeds.sum(), model)
    _step(model, self.optimizer)

    return accum_loss / self.args.gradient_accumulation_steps


def visualize(inputs, attns):
    import numpy as np
    import matplotlib.pyplot as plt
    for t, (image, attn) in enumerate(zip(inputs['images'], attns)):
        image = (image - image.min()) / (image.max() - image.min())
        image = torch.clip(image, min=0, max=1)
        image_array = image.permute(1, 2, 0).cpu().float().numpy()
        block_size = 32
        height, width = image_array.shape[:2]
        fig, _ = plt.subplots(figsize=(10, 10))
        for idx, sp in enumerate([0.25, 0.167, 0.1]):
            ax = fig.add_subplot(int(f"13{idx+1}"))
            scores = torch.zeros_like(attn)
            topk_indices = torch.topk(attn, k=int(attn.numel() * sp)).indices
            scores[topk_indices] = 1
            scores = scores.unflatten(0, (block_size, block_size))
            scores = scores.cpu().float().numpy()
            num_blocks_height = height // block_size
            num_blocks_width = width // block_size
            mask = np.zeros((height, width, 4))
            for i in range(num_blocks_height):
                for j in range(num_blocks_width):
                    color = [1.0, 0.0, 0.0, 0.5] if scores[i, j] > 0.5 else [0.0, 1.0, 0.0, 0.5]
                    mask[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = color
            ax.imshow(image_array)
            ax.imshow(mask, alpha=mask[:, :, 3])
        plt.axis('off')
        plt.savefig(f"output_image_with_mask_{t}.png", bbox_inches='tight', pad_inches=0)
        plt.show()


def _spaco2(self, model, inputs):

    fwd_chunk_size = 512
    bwd_chunk_size = 384
    chunk_budget = 1

    valid_label_count = (inputs['labels'] != -100).sum()

    # prepare inputs
    attention_mask, inputs_embeds, labels, image_masks, attns = _prepare_inputs(model, inputs)

    if False:
        visualize(inputs, attns)
        if dist.get_rank() == 0:
            import IPython
            IPython.embed() 
        dist.barrier()

    # construct labels and 4d attention mask
    labels = torch.cat((labels[:, 1:], torch.full_like(labels[:, :1], fill_value=-100)), dim=-1)
    mask_4d = _construct_4d_mask(
        attention_mask, 
        inputs_embeds.dtype, 
        inputs_embeds.device)

    # chunkize inputs
    embeds_list, labels_list = _chunkize_inputs(
        [inputs_embeds.detach(), labels],
        chunk_size=fwd_chunk_size)

    # LLM forward prop
    raw_loss, seco_cache = _first_forward_prop_seco(
        model, 
        embeds_list, 
        labels_list, 
        mask_4d, 
        valid_label_count,
        return_raw_loss=True)
    accum_loss = raw_loss.sum() / valid_label_count
    
    inputs_embeds_detach, labels, position_ids, sparse_indices = _sample_tokens_v2(
        inputs_embeds, 
        labels,
        attention_mask,
        image_masks,
        chunk_budget,
        bwd_chunk_size,
        attns,
        raw_loss)
    
    seco_cache.squeeze()
    embeds_list, labels_list, position_list, indices_list = _chunkize_inputs(
        (inputs_embeds_detach, labels, position_ids, sparse_indices),
        chunk_size=bwd_chunk_size)

    # recompute valid label count
    valid_label_count = 0
    for chunk_labels in labels_list:
        valid_label_count += (chunk_labels != -100).count_nonzero().item()

    # LLM backward prop
    neg_inf = torch.finfo(mask_4d.dtype).min
    generator = reversed(list(enumerate(zip(embeds_list, labels_list, position_list, indices_list))))

    for i, (chunk_embeds, chunk_labels, chunk_position, chunk_indices) in generator:
        
        # reconstruct input attention mask
        window_size = chunk_indices.shape[-1]

        chunk_mask_4d = torch.gather(
            mask_4d, 
            dim=2, 
            index=chunk_indices[:, None, :, None].expand(-1, -1, -1, mask_4d.shape[-1]))
        chunk_mask_4d.scatter_(
            dim=-1, 
            index=chunk_indices[:, None, None, :].expand(-1, -1, window_size, -1), 
            value=neg_inf)
        pad_mask_4d = torch.full(
            size=(window_size, window_size), 
            fill_value=neg_inf,
            dtype=chunk_mask_4d.dtype, 
            device=chunk_mask_4d.device)
        pad_mask_4d = pad_mask_4d.triu(1)[None, None, :, :].expand(chunk_mask_4d.shape[0], -1, -1, -1)
        mixed_mask_4d = torch.cat([chunk_mask_4d, pad_mask_4d], dim=-1)

        outputs = model(
            input_ids=None,
            attention_mask=mixed_mask_4d,
            inputs_embeds=chunk_embeds,
            labels=chunk_labels,
            position_ids=chunk_position,
            past_key_values=seco_cache,
            shift_label=False,
            is_reduce=False)

        loss = outputs['loss'].sum() / valid_label_count

        sparse_grad = seco_cache.index(0).collect_sparse_grad(chunk_indices)
        seco_cache.index(1).copy_scaled_grad(gd=sparse_grad)
        _backward(loss, model)
        seco_cache.delete(1)

    inputs_embeds.register_hook(partial(
        _set_to_incomming_grad,
        incomming_grad=inputs_embeds_detach.grad,
        indices=sparse_indices))
    _backward(inputs_embeds.sum(), model)
    _step(model, self.optimizer)

    return accum_loss / self.args.gradient_accumulation_steps


def _spaco(self, model, inputs):
    fwd_chunk_size = 512
    bwd_chunk_size = 128
    chunk_budget = 4

    valid_label_count = (inputs['labels'] != -100).sum()

    # prepare inputs
    attention_mask, inputs_embeds, labels, image_masks = _prepare_inputs(model, inputs)

    inputs_embeds_detach = inputs_embeds.detach()
    inputs_embeds_detach.requires_grad_(True)
    labels = torch.cat((labels[:, 1:], torch.full_like(labels[:, :1], fill_value=-100)), dim=-1)

    # chunkize inputs
    embeds_list, labels_list, masks_list = _chunkize_inputs(
        [inputs_embeds_detach, labels, attention_mask],
        chunk_size=fwd_chunk_size)

    # LLM forward prop
    accum_loss, seco_cache = _first_forward_prop_seco(
        model, 
        embeds_list, 
        labels_list, 
        masks_list, 
        valid_label_count)

    if fwd_chunk_size != bwd_chunk_size:
        seco_cache.reorganize(bwd_chunk_size)
        embeds_list, labels_list, masks_list = _chunkize_inputs(
            [inputs_embeds_detach, labels, attention_mask],
            chunk_size=bwd_chunk_size,)

    I = _sample_chunks(attention_mask, image_masks, bwd_chunk_size, chunk_budget)

    # recompute valid label count
    valid_label_count = 0
    for i, chunk_labels in enumerate(labels_list):
        valid_label_count += (chunk_labels != -100).count_nonzero()

    # LLM backward prop
    generator = reversed(list(enumerate(zip(embeds_list, labels_list))))
    for i, (chunk_embeds, chunk_labels) in generator:
        if i in I:
            tmp_cache = seco_cache.range(i)
            outputs = model(
                input_ids=None,
                attention_mask=torch.cat(masks_list[:i+1], dim=1),
                inputs_embeds=chunk_embeds,
                labels=chunk_labels,
                past_key_values=tmp_cache,
                shift_label=False,
                is_reduce=False)
            loss = outputs['loss'].sum() / valid_label_count
            tmp_cache.index(i).copy_grad(gd=seco_cache.index(i).grad)
            _backward(loss, model)

    inputs_embeds.register_hook(partial(
        _set_to_incomming_grad,
        incomming_grad=inputs_embeds_detach.grad))
    _backward(inputs_embeds.sum(), model)

    _step(model, self.optimizer)

    return accum_loss / self.args.gradient_accumulation_steps


def _sparse_mlp(self, model, inputs):

    attention_mask, inputs_embeds, labels, _ = _prepare_inputs(model, inputs)   
    inputs_embeds_detach = inputs_embeds.detach()

    # forward propagation
    with torch.no_grad():
        checkpoints = model(
            inputs_embeds=inputs_embeds_detach,
            attention_mask=attention_mask)

    # backward propagation
    loss, grad = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        backward=partial(_backward, model=model),
        checkpoints=checkpoints)

    inputs_embeds.register_hook(partial(_set_to_incomming_grad, incomming_grad=grad))
    _backward(inputs_embeds.sum(), model)
    _step(model, self.optimizer)

    return loss / self.args.gradient_accumulation_steps


def _profile_sparse_mlp(self, model, inputs):
    from pygments.console import colorize
    from profiler import WallTime

    if dist.is_initialized():
        gpu_device = dist.get_rank()
    else:
        gpu_device = 0
    
    t0 = WallTime("end to end time", gpu_device)
    t1 = WallTime("total forward", gpu_device)
    t2 = WallTime("total backward", gpu_device)

    for _ in range(3):
        with t0:
            with t1:
                attention_mask, inputs_embeds, labels, _ = _prepare_inputs(model, inputs)   
                inputs_embeds_detach = inputs_embeds.detach()

                # forward propagation
                with torch.no_grad():
                    checkpoints = model(
                        inputs_embeds=inputs_embeds_detach,
                        attention_mask=attention_mask)
            
            with t2:
                # backward propagation
                loss, grad = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    backward=partial(_backward, model=model),
                    checkpoints=checkpoints)
            
                inputs_embeds.register_hook(partial(_set_to_incomming_grad, incomming_grad=grad))
                _backward(inputs_embeds.sum(), model)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # output key information
        t0.result(detail=True)
        t1.result(detail=True)
        t2.result(detail=True)

        # compute memory allocation
        param_count = 0
        grad_count = 0
        for param in model.parameters():
            param_count += param.data.numel()
            if param.requires_grad:
                grad_count += param.data.numel()
        param_memory = f"{param_count * 2 / 1024 ** 3: .1f}"
        grad_memory = f"{grad_count * 2 / 1024 ** 3: .1f}"

        memory_info = {
            "cur mem alloc": f"{torch.cuda.memory_allocated(gpu_device) / 1024 ** 3: .1f}",
            "max mem alloc": f"{torch.cuda.max_memory_allocated(gpu_device) / 1024 ** 3: .1f}",
            "parameters": param_memory,
            "gradients": grad_memory,
        }
        print('=' * 10)

        for key, value in memory_info.items():
            key = colorize("green", key)
            value = colorize("yellow", value)
            print(f"{key}:\t{value}")

    # breakpoint
    import IPython
    IPython.embed()
    raise NotImplementedError


def _profile_parallel(self, model, inputs):
    from pygments.console import colorize
    from profiler import WallTime

    context_length = 2048
    valid_label_count = (inputs['labels'] != -100).sum()

    if dist.is_initialized():
        gpu_device = dist.get_rank()
    else:
        gpu_device = 0
    
    t0 = WallTime("end to end time", gpu_device)
    t1 = WallTime("vision tower fwd", gpu_device)
    t2 = WallTime("forward prop", gpu_device)

    while inputs['input_ids'].shape[-1] < context_length:
        inputs['input_ids'] = torch.cat([inputs['input_ids'], torch.full_like(inputs['input_ids'], fill_value=self.tokenizer.pad_token_id)], dim=-1)
        inputs['labels'] = torch.cat([inputs['labels'], torch.full_like(inputs['labels'], fill_value=-100)], dim=-1)
        inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.full_like(inputs['attention_mask'], fill_value=0)], dim=-1)
    inputs['input_ids'] = inputs['input_ids'][:, :context_length]
    inputs['labels'] = inputs['labels'][:, :context_length]
    inputs['attention_mask'] = inputs['attention_mask'][:, :context_length]

    for _ in range(3):
        with t0:
            with t1:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    (
                        _, 
                        attention_mask, 
                        _, 
                        inputs_embeds, 
                        labels, 
                        _
                    ) = model.prepare_inputs_labels_for_multimodal(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        past_key_values=None,
                        labels=inputs['labels'],
                        images=inputs['images'])

                inputs_embeds, labels, attention_mask, _ = _maybe_align_length_across_gpus(
                    inputs_embeds, 
                    labels, 
                    attention_mask,
                    None)

                outputs = model(
                    input_ids=None,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    past_key_values=None,
                    is_reduce=False)

                loss = outputs['loss'].sum() / valid_label_count

            with t2:
                _backward(loss, model)
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        # output key information
        t0.result(detail=True)
        t1.result(detail=True)
        t2.result(detail=True)

        # compute memory allocation
        param_count = 0
        grad_count = 0
        for param in model.parameters():
            param_count += param.data.numel()
            if param.requires_grad:
                grad_count += param.data.numel()
        param_memory = f"{param_count * 2 / 1024 ** 3: .1f}"
        grad_memory = f"{grad_count * 2 / 1024 ** 3: .1f}"

        memory_info = {
            "cur mem alloc": f"{torch.cuda.memory_allocated(gpu_device) / 1024 ** 3: .1f}",
            "max mem alloc": f"{torch.cuda.max_memory_allocated(gpu_device) / 1024 ** 3: .1f}",
            "parameters": param_memory,
            "gradients": grad_memory
        }
        print('=' * 10)

        for key, value in memory_info.items():
            key = colorize("green", key)
            value = colorize("yellow", value)
            print(f"{key}:\t{value}")

    # breakpoint
    import IPython
    IPython.embed()
    raise NotImplementedError


def _profile_seco(self, model, inputs):
    from pygments.console import colorize
    from profiler import WallTime

    fwd_chunk_size = 512
    bwd_chunk_size = 512
    context_length = 3072
    valid_label_count = (inputs['labels'] != -100).sum()

    if dist.is_initialized():
        gpu_device = dist.get_rank()
    else:
        gpu_device = 0
    
    t0 = WallTime("end to end time", gpu_device)
    t1 = WallTime("vision tower fwd", gpu_device)
    t2 = WallTime("forward prop", gpu_device)
    t3 = WallTime("reorganize time", gpu_device)
    t4 = WallTime("backward prop", gpu_device)
    t5 = WallTime("vision tower bwd", gpu_device)

    while inputs['input_ids'].shape[-1] < context_length:
        inputs['input_ids'] = torch.cat([inputs['input_ids'], torch.full_like(inputs['input_ids'], fill_value=self.tokenizer.pad_token_id)], dim=-1)
        inputs['labels'] = torch.cat([inputs['labels'], torch.full_like(inputs['labels'], fill_value=-100)], dim=-1)
        inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.full_like(inputs['attention_mask'], fill_value=0)], dim=-1)
    inputs['input_ids'] = inputs['input_ids'][:, :context_length]
    inputs['labels'] = inputs['labels'][:, :context_length]
    inputs['attention_mask'] = inputs['attention_mask'][:, :context_length]
 
    for _ in range(3):
        with t0:
            with t1:
                attention_mask, inputs_embeds, labels, _ = _prepare_inputs(model, inputs)

            # align length across gpus
            inputs_embeds, labels, attention_mask, _ = _maybe_align_length_across_gpus(
                inputs_embeds, 
                labels, 
                attention_mask,
                None)
            
            inputs_embeds_detach = inputs_embeds.detach()
            inputs_embeds_detach.requires_grad_(True)
            labels = torch.cat((labels[:, 1:], torch.full_like(labels[:, :1], fill_value=-100)), dim=-1)

            # chunkize inputs
            inputs_embeds_list, labels_list, masks_list = _chunkize_inputs(
                [inputs_embeds_detach, labels, attention_mask],
                chunk_size=fwd_chunk_size)

            with t2:
                accum_loss, seco_cache = _first_forward_prop_seco(
                    model, 
                    inputs_embeds_list, 
                    labels_list, 
                    masks_list, 
                    valid_label_count,
                    cpu_offload=1)

            with t3:
                if fwd_chunk_size != bwd_chunk_size:
                    seco_cache.reorganize(bwd_chunk_size)
                    inputs_embeds_list, labels_list, masks_list = _chunkize_inputs(
                        [inputs_embeds_detach, labels, attention_mask],
                        chunk_size=bwd_chunk_size)

            with t4:
                generator = reversed(list(enumerate(zip(inputs_embeds_list, labels_list))))
                for i, (chunk_embeds, chunk_labels) in generator:
                    
                    # reconstruction
                    seco_cache.pre_reconstruction(i)
                    outputs = model(
                        input_ids=None,
                        attention_mask=torch.cat(masks_list[:i+1], dim=1),
                        inputs_embeds=chunk_embeds,
                        labels=chunk_labels,
                        past_key_values=seco_cache,
                        shift_label=False,
                        is_reduce=False)
                    loss = outputs['loss'].sum() / valid_label_count

                    # backward propagation
                    seco_cache.pre_backward(i)
                    _backward(loss, model) 
                    seco_cache.after_backward()

            with t5:
                inputs_embeds.register_hook(partial(
                    _set_to_incomming_grad, 
                    incomming_grad=inputs_embeds_detach.grad))
                _backward(inputs_embeds.sum(), model)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # output key information
        t0.result(detail=True)
        t1.result(detail=True)
        t2.result(detail=True)
        t3.result(detail=True)
        t4.result(detail=True)
        t5.result(detail=True)

        # compute memory allocation
        param_count = 0
        grad_count = 0
        for param in model.parameters():
            param_count += param.data.numel()
            if param.requires_grad:
                grad_count += param.data.numel()
        param_memory = f"{param_count * 2 / 1024 ** 3: .1f}"
        grad_memory = f"{grad_count * 2 / 1024 ** 3: .1f}"

        memory_info = {
            "cur mem alloc": f"{torch.cuda.memory_allocated(gpu_device) / 1024 ** 3: .1f}",
            "max mem alloc": f"{torch.cuda.max_memory_allocated(gpu_device) / 1024 ** 3: .1f}",
            "parameters": param_memory,
            "gradients": grad_memory
        }
        print('=' * 10)

        for key, value in memory_info.items():
            key = colorize("green", key)
            value = colorize("yellow", value)
            print(f"{key}:\t{value}")

        import IPython
        IPython.embed(header=f"fwd: {fwd_chunk_size}, bwd: {bwd_chunk_size}, len: {inputs_embeds.shape[-2]}")

    if dist.is_initialized():
        dist.barrier()

    raise NotImplementedError


def _profile_spaco(self, model, inputs):
    from pygments.console import colorize
    from profiler import WallTime

    fwd_chunk_size = 512
    bwd_chunk_size = 128
    chunk_budget = 4

    valid_label_count = (inputs['labels'] != -100).sum()

    if dist.is_initialized():
        gpu_device = dist.get_rank()
    else:
        gpu_device = 0
    
    t0 = WallTime("end to end time", gpu_device)
    t1 = WallTime("vision tower fwd", gpu_device)
    t2 = WallTime("forward prop", gpu_device)
    t3 = WallTime("reorganize time", gpu_device)
    t4 = WallTime("backward prop", gpu_device)
    t5 = WallTime("vision tower bwd", gpu_device)

    for _ in range(3):

        with t0:
            with t1:
                attention_mask, inputs_embeds, labels, image_masks = _prepare_inputs(model, inputs)
                origin_seq_len = inputs_embeds.shape[-2]

            # align length across gpus
            inputs_embeds, labels, attention_mask, image_masks = _maybe_align_length_across_gpus(
                inputs_embeds, 
                labels, 
                attention_mask,
                image_masks)
            
            inputs_embeds_detach = inputs_embeds.detach()
            inputs_embeds_detach.requires_grad_(True)
            labels = torch.cat((labels[:, 1:], torch.full_like(labels[:, :1], fill_value=-100)), dim=-1)

            # chunkize inputs
            inputs_embeds_list, labels_list, masks_list = _chunkize_inputs(
                [inputs_embeds_detach, labels, attention_mask],
                chunk_size=fwd_chunk_size)

            with t2:
                # LLM forward prop
                accum_loss, seco_cache = _first_forward_prop_seco(
                    model, 
                    inputs_embeds_list, 
                    labels_list, 
                    masks_list, 
                    valid_label_count)

            with t3:
                # maybe change chunk size before backward prop
                if fwd_chunk_size != bwd_chunk_size:
                    seco_cache.reorganize(bwd_chunk_size)
                    inputs_embeds_list, labels_list, masks_list = _chunkize_inputs(
                        [inputs_embeds_detach, labels, attention_mask],
                        chunk_size=bwd_chunk_size,)
                    
            I = _sample_chunks(attention_mask, image_masks, bwd_chunk_size, chunk_budget)

            with t4:
                # LLM backward prop
                generator = reversed(list(enumerate(zip(inputs_embeds_list, labels_list))))
                for i, (chunk_embeds, chunk_labels) in generator:
                    if i in I:
                        tmp_cache = seco_cache.range(i)
                        outputs = model(
                            input_ids=None,
                            attention_mask=torch.cat(masks_list[:i+1], dim=1),
                            inputs_embeds=chunk_embeds,
                            labels=chunk_labels,
                            past_key_values=tmp_cache,
                            shift_label=False,
                            is_reduce=False)
                        loss = outputs['loss'].sum() / valid_label_count
                        tmp_cache.index(i).copy_grad(gd=seco_cache.index(i).grad)
                        _backward(loss, model)

            with t5:
                # vision tower backward propgation
                inputs_embeds.register_hook(partial(
                    _set_to_incomming_grad, 
                    incomming_grad=inputs_embeds_detach.grad))
                _backward(inputs_embeds.sum(), model)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # output key information
        t0.result(detail=True)
        t1.result(detail=True)
        t2.result(detail=True)
        t3.result(detail=True)
        t4.result(detail=True)
        t5.result(detail=True)

        # compute memory allocation
        param_count = 0
        grad_count = 0
        for param in model.parameters():
            param_count += param.data.numel()
            if param.requires_grad:
                grad_count += param.data.numel()
        param_memory = f"{param_count * 2 / 1024 ** 3: .1f}"
        grad_memory = f"{grad_count * 2 / 1024 ** 3: .1f}"

        kv_cache_count = 0
        for x in seco_cache.k_cache:
            for y in x:
                kv_cache_count += y.numel()
        kv_cache_memory = f"{kv_cache_count * 8 / 1024 ** 3: .1f}"

        memory_info = {
            "cur mem alloc": f"{torch.cuda.memory_allocated(gpu_device) / 1024 ** 3: .1f}",
            "max mem alloc": f"{torch.cuda.max_memory_allocated(gpu_device) / 1024 ** 3: .1f}",
            "parameters": param_memory,
            "gradients": grad_memory,
            "kv cache mem": kv_cache_memory
        }
        print('=' * 10)

        for key, value in memory_info.items():
            key = colorize("green", key)
            value = colorize("yellow", value)
            print(f"{key}:\t{value}")

    # breakpoint
    import IPython
    IPython.embed(header=f"fwd: {fwd_chunk_size}, bwd: {bwd_chunk_size}")
    raise NotImplementedError


def _profile_spaco2(self, model, inputs):
    from pygments.console import colorize
    from profiler import WallTime

    fwd_chunk_size = 512
    bwd_chunk_size = 512
    chunk_budget = 1

    context_length = 2048
    valid_label_count = (inputs['labels'] != -100).sum()

    if dist.is_initialized():
        gpu_device = dist.get_rank()
    else:
        gpu_device = 0
    
    t0 = WallTime("end to end time", gpu_device)
    t1 = WallTime("vision tower fwd", gpu_device)
    t2 = WallTime("forward prop", gpu_device)
    t3 = WallTime("reorganize time", gpu_device)
    t4 = WallTime("backward prop", gpu_device)
    t5 = WallTime("vision tower bwd", gpu_device)

    while inputs['input_ids'].shape[-1] < context_length:
        inputs['input_ids'] = torch.cat([inputs['input_ids'], torch.full_like(inputs['input_ids'], fill_value=self.tokenizer.pad_token_id)], dim=-1)
        inputs['labels'] = torch.cat([inputs['labels'], torch.full_like(inputs['labels'], fill_value=-100)], dim=-1)
        inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.full_like(inputs['attention_mask'], fill_value=0)], dim=-1)
    inputs['input_ids'] = inputs['input_ids'][:, :context_length]
    inputs['labels'] = inputs['labels'][:, :context_length]
    inputs['attention_mask'] = inputs['attention_mask'][:, :context_length]

    for _ in range(3):

        with t0:

            with t1:
                # prepare inputs
                attention_mask, inputs_embeds, labels, image_masks, attns = _prepare_inputs(model, inputs)

            # construct labels and 4d attention mask
            labels = torch.cat((labels[:, 1:], torch.full_like(labels[:, :1], fill_value=-100)), dim=-1)

            mask_4d = _construct_4d_mask(
                attention_mask, 
                inputs_embeds.dtype, 
                inputs_embeds.device)

            # chunkize inputs
            embeds_list, labels_list = _chunkize_inputs(
                [inputs_embeds.detach(), labels],
                chunk_size=fwd_chunk_size)

            with t2:
                # LLM forward prop
                raw_loss, seco_cache = _first_forward_prop_seco(
                    model, 
                    embeds_list, 
                    labels_list, 
                    mask_4d, 
                    valid_label_count,
                    return_raw_loss=True)
                accum_loss = raw_loss.sum() / valid_label_count
            
            with t3:
                inputs_embeds_detach, labels, position_ids, sparse_indices = _sample_tokens_v2(
                    inputs_embeds, 
                    labels,
                    attention_mask,
                    image_masks,
                    chunk_budget,
                    bwd_chunk_size,
                    attns,
                    raw_loss)
                
                # reorganize inputs
                seco_cache.squeeze()

            embeds_list, labels_list, position_list, indices_list = _chunkize_inputs(
                (inputs_embeds_detach, labels, position_ids, sparse_indices),
                chunk_size=bwd_chunk_size)

            # recompute valid label count
            valid_label_count = 0
            for chunk_labels in labels_list:
                valid_label_count += (chunk_labels != -100).count_nonzero().item()

            with t4:
                # LLM backward prop
                neg_inf = torch.finfo(mask_4d.dtype).min
                generator = reversed(list(enumerate(zip(embeds_list, labels_list, position_list, indices_list))))
                assert len(generator) == 1
                for i, (chunk_embeds, chunk_labels, chunk_position, chunk_indices) in generator:
                    
                    # reconstruct input attention mask
                    window_size = chunk_indices.shape[-1]

                    chunk_mask_4d = torch.gather(
                        mask_4d, 
                        dim=2, 
                        index=chunk_indices[:, None, :, None].expand(-1, -1, -1, mask_4d.shape[-1]))
                    chunk_mask_4d.scatter_(
                        dim=-1, 
                        index=chunk_indices[:, None, None, :].expand(-1, -1, window_size, -1), 
                        value=neg_inf)
                    pad_mask_4d = torch.full(
                        size=(window_size, window_size), 
                        fill_value=neg_inf,
                        dtype=chunk_mask_4d.dtype, 
                        device=chunk_mask_4d.device)
                    pad_mask_4d = pad_mask_4d.triu(1)[None, None, :, :].expand(chunk_mask_4d.shape[0], -1, -1, -1)
                    mixed_mask_4d = torch.cat([chunk_mask_4d, pad_mask_4d], dim=-1)

                    outputs = model(
                        input_ids=None,
                        attention_mask=mixed_mask_4d,
                        inputs_embeds=chunk_embeds,
                        labels=chunk_labels,
                        position_ids=chunk_position,
                        past_key_values=seco_cache,
                        shift_label=False,
                        is_reduce=False)

                    loss = outputs['loss'].sum() / valid_label_count

                    sparse_grad = seco_cache.index(0).collect_sparse_grad(chunk_indices)
                    seco_cache.index(1).copy_scaled_grad(gd=sparse_grad)
                    _backward(loss, model)
                    seco_cache.delete(1)

            with t5:
                inputs_embeds.register_hook(partial(
                    _set_to_incomming_grad,
                    incomming_grad=inputs_embeds_detach.grad,
                    indices=sparse_indices))
                _backward(inputs_embeds.sum(), model)


    if not dist.is_initialized() or dist.get_rank() == 0:
        # output key information
        t0.result(detail=True)
        t1.result(detail=True)
        t2.result(detail=True)
        t3.result(detail=True)
        t4.result(detail=True)
        t5.result(detail=True)

        # compute memory allocation
        param_count = 0
        grad_count = 0
        for param in model.parameters():
            param_count += param.data.numel()
            if param.requires_grad:
                grad_count += param.data.numel()
        param_memory = f"{param_count * 2 / 1024 ** 3: .1f}"
        grad_memory = f"{grad_count * 2 / 1024 ** 3: .1f}"

        kv_cache_count = 0
        for x in seco_cache.k_cache:
            for y in x:
                kv_cache_count += y.numel()
        kv_cache_memory = f"{kv_cache_count * 8 / 1024 ** 3: .1f}"

        memory_info = {
            "cur mem alloc": f"{torch.cuda.memory_allocated(gpu_device) / 1024 ** 3: .1f}",
            "max mem alloc": f"{torch.cuda.max_memory_allocated(gpu_device) / 1024 ** 3: .1f}",
            "parameters": param_memory,
            "gradients": grad_memory,
            "kv cache mem": kv_cache_memory
        }
        print('=' * 10)

        for key, value in memory_info.items():
            key = colorize("green", key)
            value = colorize("yellow", value)
            print(f"{key}:\t{value}")

    # breakpoint
    import IPython
    IPython.embed(header=f"fwd: {fwd_chunk_size}, bwd: {bwd_chunk_size}")
    raise NotImplementedError
