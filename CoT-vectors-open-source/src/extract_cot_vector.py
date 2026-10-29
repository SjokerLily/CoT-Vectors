import argparse
import os
from collections import defaultdict

import torch
from omegaconf import OmegaConf, DictConfig
import hydra
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import paths
from src.utils import build_model
from src.qwen_model_wrapper import QwenModelWrapper
from src.llama_model_wrapper import LlamaModelWrapper
from src.dataset_utils import dataset_mapping
from src.shift_encoder import AttnFFNShift, AttnApproximator, ShiftStrategy

@hydra.main(config_path="config", config_name="extract", version_base=None)
def main(cfg: DictConfig):
    # 1. Load the Qwen model (frozen, non-adapter)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if "qwen2.5" in cfg.model_name:
        # build_model now returns qwen_hf_id, model_path
        qwen_hf_id, model_path = build_model(cfg)
        # QwenModelWrapper now takes classes, model_root, and common_args
        lmm = QwenModelWrapper(
            model_root=model_path, # Pass the local path as model_root
            processor_class=AutoTokenizer, # Pass the class itself
            model_class=AutoModelForCausalLM, # Pass the class itself
            support_models=[qwen_hf_id], # Pass original HF ID for tracking/validation
            local_files_only=True, # Ensure local files are used
            torch_dtype=eval(cfg.dtype), # Pass torch_dtype to ModelBase
            processor_args={"padding_side": "left"}, # Add this line to set padding_side
            model_name=cfg.model_name # Pass model_name to QwenModelWrapper
        ).to(device)
    elif "llama-3.1-8b-instruct" in cfg.model_name:
        llama_hf_id,model_path = build_model(cfg)
        lmm=LlamaModelWrapper(
            model_root=model_path,  # Pass the local path as model_root
            processor_class=AutoTokenizer,  # Pass the class itself
            model_class=AutoModelForCausalLM,  # Pass the class itself
            support_models=[llama_hf_id],  # Pass original HF ID for tracking/validation
            local_files_only=True,  # Ensure local files are used
            torch_dtype=eval(cfg.dtype),  # Pass torch_dtype to ModelBase
            processor_args={
                "padding_side": "left",
                "use_fast":False
                },  # Add this line to set padding_side
            model_name=cfg.model_name  # Pass model_name to LlamaModelWrapper
        ).to(device)
        if lmm.processor.pad_token is None:
            lmm.processor.pad_token = lmm.processor.eos_token

    lmm.eval()
    for param in lmm.model.parameters():  # Ensure the actual model's parameters are frozen
        param.requires_grad = False

    if getattr(cfg, "main_experiment", False):
        import random
        random.seed(cfg.data.seed)
        torch.manual_seed(cfg.data.seed)
        torch.cuda.manual_seed_all(cfg.data.seed)

    # 2. Instantiate dataset and dataloader for Q+CoT+A and Q+A inputs
    # Use the 'train' split of GSM8k for extraction
    # Set training mode to TRAIN_TEACHER to get both Q+CoT+A and Q+A in collate_fn
    dataset = dataset_mapping[cfg.data.name](cfg.data, model_processor=lmm.processor, model_name=cfg.model_name) # Pass lmm.processor
    dataloader = dataset.train_dataloader(lmm, batch_size=cfg.batch_size, distributed=False)  # Pass lmm here

    # 3. Instantiate ShiftEncoder with RECORD_HIDDEN_STATES strategy
    # Dynamically instantiate based on cfg.encoder.cls
    # For extraction, we always want to record hidden states for both attention and FFN.
    shift_encoder = hydra.utils.instantiate(
        cfg.encoder.cls,
        lmm=lmm,
        # Explicitly set strategies for extraction to ensure all required hooks are enabled.
        attn_strategy="ShiftStrategy.VECTOR_SHIFT | ShiftStrategy.RECORD_HIDDEN_STATES | ShiftStrategy.MULTI_HEAD | ShiftStrategy.RECORD_RAW_ATTN_OUTPUTS",
        ffn_strategy="ShiftStrategy.RECORD_HIDDEN_STATES",
        _recursive_=False  # Prevent deep instantiation if not needed, for clarity
    ).to(device)
    
    # Set encoder to evaluation mode and register shift hooks for proper behavior
    shift_encoder.eval()
    # shift_encoder.register_shift_hooks() is responsible for patching the model's forward
    # method (e.g., qwen_attn_forward) to enable shift application and raw output recording.
    # We need to store the handles to remove them later, though for extraction they might not be critical.
    _ = shift_encoder.register_shift_hooks()  # Assign to _ as we don't need the hooks object directly here

    extraction_layers = getattr(cfg, "extraction_layers", None)
    if extraction_layers is None:
        extraction_layers = list(range(lmm.model.config.num_hidden_layers))
    if any(not 0 <= layer < lmm.model.config.num_hidden_layers for layer in extraction_layers):
        raise ValueError("Invalid extraction layer")

    # Initialize lists to store the accumulated differences (gaps) for each layer
    # The format depends on whether it's AttnFFNShift or AttnApproximator (MULTI_HEAD)
    # We will store them as lists of tensors, then average.
    
    # For FFN gaps: (num_layers, hidden_dim)
    accumulated_ffn_gaps = [torch.zeros(lmm.model.config.hidden_size).to(device) for _ in range(lmm.model.config.num_hidden_layers)]
    ffn_sample_count_per_layer = [0 for _ in range(lmm.model.config.num_hidden_layers)]
    accumulated_ffn_hs = [torch.zeros(lmm.model.config.hidden_size).to(device) for _ in range(lmm.model.config.num_hidden_layers)]
    teacher_hidden_states_dict = defaultdict(list)
    student_hidden_states_dict = defaultdict(list)
    teacher_hidden_states_dict_type = defaultdict(lambda: defaultdict(list))

    # For Attention gaps: (num_layers, num_heads, head_dim)
    num_heads = lmm.model.config.num_attention_heads
    head_dim = lmm.model.config.hidden_size // num_heads
    accumulated_attn_gaps = [torch.zeros(num_heads, head_dim, device=device) for _ in range(lmm.model.config.num_hidden_layers)]
    attn_sample_count_per_layer = [0 for _ in range(lmm.model.config.num_hidden_layers)]

    print("Starting CoT vector extraction...")
    # Iterate through dataloader
    for batch in tqdm(dataloader, desc="Extracting CoT vectors"):
        # Reset hidden states before each batch processing
        # sample_type = batch['type']
        shift_encoder.reset_hidden_states()
        
        # --- NEW: Set recording context for teacher pass ---
        shift_encoder.current_recording_context = "teacher"
        # -----------------------------------------------------

        # Process teacher batch (Q+CoT+A) to get hidden states
        hooks_teacher = shift_encoder.register_record_hooks()
        with torch.no_grad():
            teacher_inputs = {k: v.to(device) for k, v in batch["prefix_inputs"].items()}
            if "offset_mapping" in teacher_inputs.keys():
                del teacher_inputs["offset_mapping"]
            teacher_outputs = lmm.model(
                **teacher_inputs,
                output_hidden_states=True
            )
            # Store raw attention outputs for teacher before resetting
            per_batch_teacher_raw_attn_outputs = [t.clone() if t is not None else None for t in shift_encoder.teacher_raw_attn_outputs]

            teacher_ffn_hidden_states = shift_encoder.ffn_hidden_states
            teacher_attn_hidden_states = shift_encoder.attn_hidden_states

        # shift_encoder.remove_hooks(hooks_teacher)
        for handle_name, handle_obj in hooks_teacher.items():
            if isinstance(handle_obj, list):
                for h in handle_obj:
                    h.remove()
            else:
                handle_obj.remove()

        # Reset hidden states again for student pass
        shift_encoder.reset_hidden_states()

        # --- NEW: Set recording context for student pass ---
        shift_encoder.current_recording_context = "student"
        # ---------------------------------------------------

        # Process student batch (Q+A) to get hidden states
        hooks_student = shift_encoder.register_record_hooks()
        with torch.no_grad():
            student_outputs = lmm.model(
                **{k: v.to(device) for k, v in batch["student_inputs"].items()},
                output_hidden_states=True
            )
            # Store raw attention outputs for student
            per_batch_student_raw_attn_outputs = [t.clone() if t is not None else None for t in shift_encoder.student_raw_attn_outputs]

            # student_labels != -100 identifies the answer tokens (and EOS) for student. We need to mask out EOS token.
            if batch["student_labels"].dtype == torch.bool:
                student_answer_mask = batch["student_labels"].to(device).bool()
            else:
                student_answer_mask_raw = (batch["student_labels"] != -100) & batch["student_inputs"]["attention_mask"].bool()
                # student_answer_mask = torch.zeros_like(student_answer_mask_raw).bool()
                # for i in range(student_answer_mask_raw.shape[0]):
                #     true_indices = student_answer_mask_raw[i].nonzero(as_tuple=True)[0]
                #     if true_indices.numel() > 0:
                #         # Keep all but the last true index (which is EOS)
                #         last_true_idx = true_indices[-1]
                #         student_answer_mask_raw[i, last_true_idx] = 0
                student_answer_mask = student_answer_mask_raw.to(device).bool()

            student_ffn_hidden_states = shift_encoder.ffn_hidden_states
            student_attn_hidden_states = shift_encoder.attn_hidden_states

        # shift_encoder.remove_hooks(hooks_student)
        for handle_name, handle_obj in hooks_student.items():
            if isinstance(handle_obj, list):
                for h in handle_obj:
                    h.remove()
            else:
                handle_obj.remove()

        # Calculate the difference (gap) and accumulate
        for layer_idx in extraction_layers:
            # FFN
            teacher_ffn_hs = teacher_ffn_hidden_states[layer_idx]
            student_ffn_hs = student_ffn_hidden_states[layer_idx]


            if teacher_ffn_hs is not None and student_ffn_hs is not None:
                #print(f"DEBUG FFN (Layer {layer_idx}): teacher_ffn_hs shape: {teacher_ffn_hs.shape}, dtype: {teacher_ffn_hs.dtype}")
                #print(f"DEBUG FFN (Layer {layer_idx}): batch['teacher_answer_mask'] shape: {batch['teacher_answer_mask'].shape}, dtype: {batch['teacher_answer_mask'].dtype}")
                # Apply mask (B, S) to (B, S, D) -> result is (N, D)
                masked_teacher_ffn_hs = teacher_ffn_hs[batch['teacher_answer_mask'].bool()]
                masked_student_ffn_hs = student_ffn_hs[student_answer_mask.bool()]
                
                # Check for empty tensors before subtraction
                if masked_teacher_ffn_hs.numel() == 0 or masked_student_ffn_hs.numel() == 0:
                    print(f"DEBUG FFN (Layer {layer_idx}): Skipping due to empty masked tensors. Teacher numel: {masked_teacher_ffn_hs.numel()}, Student numel: {masked_student_ffn_hs.numel()}")
                    continue  # Skip if no tokens were masked

                if masked_teacher_ffn_hs.shape == masked_student_ffn_hs.shape:
                    ffn_gap = masked_teacher_ffn_hs - masked_student_ffn_hs
                    accumulated_ffn_gaps[layer_idx] += ffn_gap.mean(dim=0)  # Average across all masked tokens
                    accumulated_ffn_hs[layer_idx] += masked_teacher_ffn_hs.mean(dim=0)
                    ffn_sample_count_per_layer[layer_idx] += 1
                    teacher_hidden_states_dict[layer_idx].append(masked_teacher_ffn_hs.mean(dim=0))
                    student_hidden_states_dict[layer_idx].append(masked_student_ffn_hs.mean(dim=0))
                    # teacher_hidden_states_dict_type[sample_type][layer_idx].append(masked_teacher_ffn_hs.mean(dim=0))
                else:
                    print(f"Warning: FFN shapes mismatch for layer {layer_idx}: Teacher {masked_teacher_ffn_hs.shape} vs Student {masked_student_ffn_hs.shape}")

            # ATTN (using raw_attn_outputs for per-head extraction)
            if accumulated_attn_gaps is not None:
                # Retrieve raw attention outputs from the shift_encoder
                teacher_raw_attn_hs = per_batch_teacher_raw_attn_outputs[layer_idx]
                student_raw_attn_hs = per_batch_student_raw_attn_outputs[layer_idx]

                if teacher_raw_attn_hs is not None and student_raw_attn_hs is not None:
                    # Ensure they are on the correct device for calculation
                    teacher_raw_attn_hs = teacher_raw_attn_hs.to(device)
                    student_raw_attn_hs = student_raw_attn_hs.to(device)

                    # Apply masks to select only answer tokens
                    # Create a mask of the exact target shape [B, NH, SL, HD]
                    expanded_teacher_mask = torch.zeros_like(teacher_raw_attn_hs, dtype=torch.bool, device=device)
                    expanded_student_mask = torch.zeros_like(student_raw_attn_hs, dtype=torch.bool, device=device)

                    # Get the 2D masks [B, SL] for convenience
                    mask_2d_teacher = batch['teacher_answer_mask'].bool().to(device)
                    mask_2d_student = student_answer_mask.bool().to(device)

                    # Fill the 4D mask based on the 2D answer mask
                    # This is a robust way to ensure correct broadcasting behavior for masked_select.
                    for b_idx in range(teacher_raw_attn_hs.shape[0]):
                        for sl_idx in range(teacher_raw_attn_hs.shape[2]):
                            if mask_2d_teacher[b_idx, sl_idx]:
                                expanded_teacher_mask[b_idx, :, sl_idx, :] = True
                        
                        # For student mask, iterate up to student_raw_attn_hs.shape[2]
                        for sl_idx in range(student_raw_attn_hs.shape[2]):
                            if mask_2d_student[b_idx, sl_idx]:
                                expanded_student_mask[b_idx, :, sl_idx, :] = True

                    # Use masked_select with the exact-shaped mask
                    masked_teacher_attn_hs = teacher_raw_attn_hs.masked_select(expanded_teacher_mask).view(-1, num_heads, head_dim)
                    masked_student_attn_hs = student_raw_attn_hs.masked_select(expanded_student_mask).view(-1, num_heads, head_dim)

                    # Check for empty tensors after masking
                    if masked_teacher_attn_hs.numel() == 0 or masked_student_attn_hs.numel() == 0:
                        print(f"DEBUG ATTN (Layer {layer_idx}): Skipping due to empty masked attention tensors. Teacher numel: {masked_teacher_attn_hs.numel()}, Student numel: {masked_student_attn_hs.numel()}")
                        # No continue here, as we still want to record that this layer/batch was processed (even if no valid tokens)
                        # However, to avoid NaN/Inf from mean of empty, we should ensure no accumulation for empty batches.
                        # For now, just skip accumulation if empty, and the sample count will also reflect this.
                    else:
                        if masked_teacher_attn_hs.shape == masked_student_attn_hs.shape:
                            attn_gap = masked_teacher_attn_hs - masked_student_attn_hs
                            # Average across the first dimension (all masked tokens) to get [NH, HD]
                            accumulated_attn_gaps[layer_idx] += attn_gap.mean(dim=0)
                            attn_sample_count_per_layer[layer_idx] += 1
                        else:
                            print(f"Warning: Attention shapes mismatch after masking for layer {layer_idx}: Teacher {masked_teacher_attn_hs.shape} vs Student {masked_student_attn_hs.shape}")
                else:
                    print(f"DEBUG ATTN (Layer {layer_idx}): Skipping due to None raw attention outputs. Teacher: {teacher_raw_attn_hs is not None}, Student: {student_raw_attn_hs is not None}")


    # Average the accumulated differences to get the final CoT vector
    final_ffn_cot_vector = torch.stack([
        vec / count if count > 0 else torch.zeros_like(vec)
        for vec, count in zip(accumulated_ffn_gaps, ffn_sample_count_per_layer)
    ])

    final_ffn_hs_vector = torch.stack([
        vec / count if count > 0 else torch.zeros_like(vec)
        for vec, count in zip(accumulated_ffn_hs, ffn_sample_count_per_layer)
    ])

    final_attn_cot_vector = None
    if accumulated_attn_gaps is not None:
        final_attn_cot_vector = torch.stack([
            vec / count if count > 0 else torch.zeros_like(vec)
            for vec, count in zip(accumulated_attn_gaps, attn_sample_count_per_layer)
        ])

    # Save the extracted average CoT vector
    output_dir = os.path.dirname(cfg.output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save a dictionary containing both FFN and Attention vectors
    cot_vectors_to_save = {
        "ffn_cot_vector": final_ffn_cot_vector.cpu(),
        "ffn_hs_vector": final_ffn_hs_vector.cpu(),
        "attn_cot_vector": final_attn_cot_vector.cpu() if final_attn_cot_vector is not None else None,
        "encoder_type": cfg.encoder.cls,  # Save the encoder type used for extraction
        "multi_head_attn_strategy": ShiftStrategy.MULTI_HEAD in shift_encoder.attn_strategy if isinstance(shift_encoder, AttnApproximator) else False
    }
    
    torch.save(cot_vectors_to_save, cfg.output_path)
    print(f"Average CoT vectors saved to {cfg.output_path}")


if __name__ == "__main__":
    main() 