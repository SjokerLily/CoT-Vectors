# -*- coding: utf-8 -*-

import json
import torch
import os
import re
import evaluate
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from src import paths
from src.utils import *
from src.dataset_utils import dataset_mapping, DatasetBase
from src.qwen_model_wrapper import QwenModelWrapper
from src.llama_model_wrapper import LlamaModelWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.shift_encoder import AttnFFNShift, AttnApproximator, ShiftStrategy # Import necessary classes

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#
# import sys
# print("--- Debugging Environment Info ---")
# print(f"Python Version: {sys.version}")
# print(f"Python Executable: {sys.executable}")
# print("----------------------------------")
# 
import re
import sys


@hydra.main(config_path="config", config_name="eval.yaml", version_base=None)
def main(cfg: DictConfig):
    # initialize dataset and variables
    runname = get_expand_runname(cfg)
    is_icl = cfg.ckpt_path is None
    cfg.data.is_icl = is_icl
    
    mode = getattr(cfg, "mode", "eval")  # 获取mode参数，默认为eval

    if getattr(cfg, "main_experiment", False):
        import random
        random.seed(cfg.data.seed)
        torch.manual_seed(cfg.data.seed)
        torch.cuda.manual_seed_all(cfg.data.seed)

    dataset: DatasetBase = dataset_mapping[cfg.data.name](cfg.data, model_name=cfg.model_name)
    
    if mode == "generate_self_cot":
        # 对于Self-CoT生成模式，直接使用cfg中指定的output_path
        output_dir = os.path.dirname(cfg.output_path)
        os.makedirs(output_dir, exist_ok=True)
        # record_path 在这种模式下直接就是 cfg.output_path
        print(f"Self-CoT generation mode. Output will be saved to {cfg.output_path}")
    else:
        # 对于常规评估模式
        if is_icl:
            if cfg.use_extracted_cot_vector:
                record_dir = os.path.join(paths.result_dir, "record")
                # Generate a record path based on the extracted vector file name
                extracted_name = os.path.basename(cfg.extracted_cot_vector_path).replace('.pt', '')
                # 以extracted_name作为文件夹名称在record_dir底下新建文件夹，并将extracted_name添加到record_dir里
                record_dir = os.path.join(record_dir, extracted_name)
                os.makedirs(record_dir, exist_ok=True)
                # if cfg.use_base_vector: 结果文件的名称为f"use_base_vector_{cfg.only_shift_at_layer}_{cfg.static_mu_value}.json"
                # else: 结果文件的名称为f"{cfg.encoder.name}_{cfg.static_mu_value}.json"
                
                eval_mode_suffix = ""
                if cfg.eval_mode == "EVAL_WITH_COT_VECTOR_ONESHOT":
                    eval_mode_suffix = "_oneshot"
                elif cfg.eval_mode == "EVAL_WITH_COT_VECTOR_DIRECT_Q":
                    eval_mode_suffix = "_direct_q"

                if cfg.use_base_vector:
                    # Format layer string for filename based on its content, assuming it's a string
                    layers_for_filename = str(cfg.only_shift_at_layer)
                    if '[' in layers_for_filename and ']' in layers_for_filename: # Heuristic for list string
                        layers_for_filename = layers_for_filename.replace('[', '').replace(']', '').replace(', ', '_')

                    record_path = os.path.join(record_dir,
                                               f"use_base_vector_layers_{layers_for_filename}{eval_mode_suffix}_{cfg.static_mu_value}.json")
                else:
                    # Format layer string for filename based on its content, assuming it's a string
                    layers_for_filename = str(cfg.only_shift_at_layer)
                    if '[' in layers_for_filename and ']' in layers_for_filename: # Heuristic for list string
                        layers_for_filename = layers_for_filename.replace('[', '').replace(']', '').replace(', ', '_')

                    record_path = os.path.join(record_dir, f"{cfg.use_extracted_cot_vector_type}_layers_{layers_for_filename}{eval_mode_suffix}_{cfg.static_mu_value}.json")
            else:
                record_dir = os.path.join(paths.result_dir, "record", runname)
                # Construct base filename parts
                file_name_parts = [
                    f"{cfg.data.num_shot}shot",
                ]
                record_filename = "_".join(file_name_parts) + ".json"
                record_path = os.path.join(record_dir, record_filename)
        else:
            record_dir = os.path.join(paths.result_dir, "record", runname)
            if not os.path.exists(cfg.ckpt_path) and not cfg.use_extracted_cot_vector:  # Add check for extracted vector
                raise FileNotFoundError(f"Checkpoint path {cfg.ckpt_path} not found.")

            epoch = re.findall(r"\d+", os.path.basename(cfg.ckpt_path))
            if len(epoch) != 1:
                raise ValueError(
                    f"Invalid checkpoint path {cfg.ckpt_path}. It should contain a single number in basename for epoch."
                )
            epoch = int(epoch[0])
            record_path = os.path.join(record_dir, f"epoch-{epoch}.json")

    if getattr(cfg, "record_path", None):
        record_path = cfg.record_path
        record_dir = os.path.dirname(record_path) or "."
    os.makedirs(record_dir, exist_ok=True)

    print(f"Evaluating at LAYER {cfg.only_shift_at_layer}")

    print("SAVE at:", record_path)

    if cfg.resume:
        if os.path.exists(record_path):
            print(f"Found exist record {record_path}, skip...")
            return

    # load model and inference
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible_devices is None or cuda_visible_devices == "":
        device = torch.device("cpu")
        print("Running on CPU.")
    else:
        device = torch.device("cuda")
        print(f"Running on GPU(s): {cuda_visible_devices}")

    if "qwen2.5" in cfg.model_name:
        # build_model now returns qwen_hf_id, model_path
        qwen_hf_id, model_path = build_model(cfg)
        # QwenModelWrapper now takes classes, model_root, and common_args
        model = QwenModelWrapper(
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
        model=LlamaModelWrapper(
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
        if model.processor.pad_token is None:
            model.processor.pad_token = model.processor.eos_token
    else:
        model = build_model(cfg).to(device, eval(cfg.dtype))

    # is_icl 的最终判断应在模型加载之后，确保 ckpt_path 的作用被考虑
    # 如果 cfg.ckpt_path 存在，那么它就不是 ICL baseline
    is_icl = cfg.ckpt_path is None # 恢复原始逻辑，根据 ckpt_path 判断是否为 ICL

    encoder = None # Initialize encoder to None
    hooks = {} # Initialize hooks dictionary

    if cfg.use_extracted_cot_vector:
        if not cfg.extracted_cot_vector_path or not os.path.exists(cfg.extracted_cot_vector_path):
            raise FileNotFoundError(f"Extracted CoT vector file not found: {cfg.extracted_cot_vector_path}")
        
        print(f"Loading extracted CoT vectors from: {cfg.extracted_cot_vector_path}")
        loaded_vectors = torch.load(cfg.extracted_cot_vector_path, map_location=device)

        # Determine strategies based on presence of vectors and multi_head_attn_strategy
        # Build strategy strings for BaseHookEncoder's eval()
        attn_strat_str = "ShiftStrategy(0)"
        if cfg.peft.name == "licv" and loaded_vectors["encoder_type"]._target_ == "src.shift_encoder.AttnApproximator":
            print("The shift encoder version conflict! Now we change it to LICV version.")
            loaded_vectors["encoder_type"]._target_ = "src.shift_encoder.AttnFFNShift"

        if loaded_vectors["encoder_type"]._target_ == "src.shift_encoder.AttnApproximator":
            if loaded_vectors["attn_cot_vector"] is not None:
                attn_strat_str = "ShiftStrategy.VECTOR_SHIFT"
                # Now loaded_vectors["encoder_type"] is a DictConfig, so compare its _target_ attribute
                if loaded_vectors["multi_head_attn_strategy"] and loaded_vectors["encoder_type"]._target_ == "src.shift_encoder.AttnApproximator":
                    attn_strat_str += " | ShiftStrategy.MULTI_HEAD"
                # Add this line to ensure LEARNABLE_SHIFT_SCALE is always included for AttnApproximator
                # when an attention shift vector is present. This is crucial for maintaining the
                # internal structure of AttnApproximator to prevent unexpected behavior.
                # if loaded_vectors["encoder_type"]._target_ == "src.shift_encoder.AttnApproximator":
                attn_strat_str += " | ShiftStrategy.LEARNABLE_SHIFT_SCALE"
                attn_strat_str += " | ShiftStrategy.EXTRACTED"

            # NEW: Add STATIC_MU_FROM_CONFIG to strategy if use_static_mu is enabled
            if getattr(cfg, "use_static_mu", False) and loaded_vectors["encoder_type"]._target_ == "src.shift_encoder.AttnApproximator":
                attn_strat_str += " | ShiftStrategy.STATIC_MU_FROM_CONFIG"

        ffn_strat_str = "ShiftStrategy.RECORD_HIDDEN_STATES"
        if loaded_vectors["ffn_cot_vector"] is not None:
            # Check the encoder type to determine if FFN shift should be activated
            if loaded_vectors["encoder_type"]._target_ == "src.shift_encoder.AttnFFNShift":
                ffn_strat_str = "ShiftStrategy.VECTOR_SHIFT"
            # For AttnApproximator, FFN shift remains off as per previous user request.
            # If recording hidden states is desired for FFN with AttnApproximator,
            # that would need a separate flag.

        # Instantiate the encoder using hydra.utils.instantiate with the stored encoder_type DictConfig
        # This handles the _target_ and other default parameters correctly.
        only_shift_at_layer = getattr(cfg, "only_shift_at_layer", None)
        if isinstance(only_shift_at_layer, str): # If it's a string, try to parse it as a list
            try:
                # Attempt to parse as JSON list or other JSON type
                parsed_value = json.loads(only_shift_at_layer)
                # Only update if parsing was successful and it's a list or a simple value (like int)
                if isinstance(parsed_value, (list, int, type(None))):
                    only_shift_at_layer = parsed_value
            except json.JSONDecodeError:
                # If it's not a valid JSON string, keep it as is (e.g., 'null' or a single integer string)
                pass
        
        encoder = hydra.utils.instantiate(
            loaded_vectors["encoder_type"],  # Pass the full DictConfig here
            lmm=model,
            attn_strategy=attn_strat_str, # Pass the string representation
            ffn_strategy=ffn_strat_str,   # Pass the string representation
            shift_scale_init_value=cfg.static_shift_scale, # Ensure scale is 1.0 for static vectors
            # NEW: Pass static_mu_value to encoder if enabled
            static_mu_value=getattr(cfg, "static_mu_value", None) if getattr(cfg, "use_static_mu", False) else None,
            # NEW: Pass only_shift_at_layer to encoder if enabled
            only_shift_at_layer=only_shift_at_layer,
            _recursive_=False  # Prevent deep instantiation if not needed, for clarity
        ).to(device, eval(cfg.dtype))

        # Assign the loaded vectors to the encoder's parameters
        with torch.no_grad():  # Ensure these assignments don't get recorded for gradients
            if loaded_vectors["encoder_type"]._target_ == "src.shift_encoder.AttnFFNShift":
                if cfg.use_base_vector:
                    encoder.ffn_shift.data = loaded_vectors["ffn_hs_vector"].to(device).to(model.model.dtype)
                    print("=" * 10, dataset.name, "Extraced CoT vectors: Base version", "=" * 10)
                else:
                    encoder.ffn_shift.data = loaded_vectors["ffn_cot_vector"].to(device).to(model.model.dtype)
                    print("=" * 10, dataset.name, "Extraced CoT vectors: LICV version", "=" * 10)
            if loaded_vectors["encoder_type"]._target_ == "src.shift_encoder.AttnApproximator":
                encoder.attn_shift.data = loaded_vectors["attn_cot_vector"].to(device).to(model.model.dtype)
                print("=" * 10, dataset.name, "Extracted attention vectors", "=" * 10)

        encoder.eval()
        hooks = encoder.register_shift_hooks()

    elif not is_icl: # This is the original logic for loading trained checkpoints
        only_shift_at_layer = getattr(cfg, "only_shift_at_layer", None)
        if isinstance(only_shift_at_layer, str): # If it's a string, try to parse it as a list
            try:
                # Attempt to parse as JSON list or other JSON type
                parsed_value = json.loads(only_shift_at_layer)
                # Only update if parsing was successful and it's a list or a simple value (like int)
                if isinstance(parsed_value, (list, int, type(None))):
                    only_shift_at_layer = parsed_value
            except json.JSONDecodeError:
                # If it's not a valid JSON string, keep it as is (e.g., 'null' or a single integer string)
                pass
        encoder = hydra.utils.instantiate(
            cfg.encoder.cls,
            only_shift_at_layer=only_shift_at_layer,
            lmm=model).to(
            device, eval(cfg.dtype)
        )

        print(f"Loading from pretrained: {cfg.ckpt_path}")
        load_from_pretrained(cfg.ckpt_path, model, encoder)
        encoder.eval()
        hooks = encoder.register_shift_hooks()

    model.eval()

    try:
        # Pass mode to dataset.eval
        result, eval_result = dataset.eval(cfg, model)
        print(f"Evaluation result for {runname}: {eval_result}")

        if mode == "generate_self_cot":
            # Save all generated results to .jsonl
            full_output_path = cfg.output_path
            with open(full_output_path, 'w', encoding='utf-8') as f:
                for record in result: # 'result' here is the list of records from dataset.eval
                    # Ensure the 'model_output' (self_cot) and 'is_correct' fields are present
                    # And 'question', 'extracted_ground_truth' (gt_answer)
                    # Map to the desired self_cot data format: question, self_cot, is_correct, gt_answer
                    self_cot_record = {
                        "question": record["question"],
                        "self_cot": record["model_output"], # Model output is the generated CoT
                        "is_correct": record["is_correct"],
                        "gt_answer": record["extracted_ground_truth"] # Ground truth answer
                    }
                    f.write(json.dumps(self_cot_record, ensure_ascii=False) + '\n')
            print(f"All generated Self-CoT saved to {full_output_path}")

            # Save only correct samples to _correct_only.jsonl
            correct_output_path = cfg.output_path.replace('.json', '_correct_only.json')
            with open(correct_output_path, 'w', encoding='utf-8') as f:
                for record in result:
                    if record["is_correct"]:
                        self_cot_record = {
                            "question": record["question"],
                            "self_cot": record["model_output"],
                            "is_correct": record["is_correct"],
                            "gt_answer": record["extracted_ground_truth"]
                        }
                        f.write(json.dumps(self_cot_record, ensure_ascii=False) + '\n')
            print(f"Correct Self-CoT samples saved to {correct_output_path}")

        else:
            # Regular evaluation saving logic
            Path(record_path).touch() # Ensure file exists for evaluate.save if it's new
        config = {"eval_args": OmegaConf.to_container(cfg, resolve=True)}
        if os.path.exists(os.path.join(record_dir, "config.json")):
            with open(os.path.join(record_dir, "config.json")) as f:
                config["train_args"] = json.load(f)

        evaluate.save(
            record_path,
            **config,
            eval_result=eval_result,
            records=result,
        )

    finally:
        if encoder:  # Only remove hooks if encoder was instantiated
            encoder.remove_hooks(hooks)
        if os.path.exists(record_path) and os.path.getsize(record_path) == 0:
            os.remove(record_path)


if __name__ == "__main__":
    main()
