from contextlib import nullcontext
import os
import torch
import torch.nn as nn
import peft
from peft import PeftModel, LoraConfig, PrefixTuningConfig, IA3Config
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re

from src import paths

OmegaConf.register_new_resolver("eval", eval, replace=True)


class NullPeftModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base_model = model
        model.requires_grad_(False)
        self.config = model.config

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def disable_adapter(self):
        return nullcontext()

    def save_pretrained(self, save_directory: str, **kwargs):
        pass


class ModuleDeviceManager:
    def __init__(self, device):
        self.device = device
        self.old_devices = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for module, old_device in self.old_devices.items():
            module.to(old_device)

    def move_module(self, module):
        self.old_devices[module] = next(module.parameters()).device
        module.to(self.device)


def convert_to_peft(cfg, lmm):
    peft_config_cls = None
    if cfg.model_name in cfg.peft:
        peft_cfg = OmegaConf.to_container(cfg.peft[cfg.model_name], resolve=True)
        if cfg.peft.name == "lora":
            peft_config_cls = LoraConfig
        elif cfg.peft.name == "prefix-tuning":
            peft_config_cls = PrefixTuningConfig
        elif cfg.peft.name == "ia3":
            peft_config_cls = IA3Config

    if peft_config_cls is None:
        model = NullPeftModel(lmm.model)
    else:
        model = peft.get_peft_model(lmm.model, peft_config_cls(**peft_cfg))
    lmm.model = model


def _model_info(cfg, model_name, hf_id):
    if cfg.model_name.lower() != model_name:
        raise ValueError(f"Unsupported main-experiment model: {cfg.model_name}")
    model_path = getattr(cfg, "model_path", None)
    if not model_path:
        raise ValueError("model_path must point to a local Hugging Face model directory")
    return hf_id, model_path


def build_qwen_model_info(cfg):
    return _model_info(cfg, "qwen2.5-math-7b", "Qwen/Qwen2.5-Math-7B")


def build_llama_model_info(cfg):
    return _model_info(cfg, "llama-3.1-8b-instruct", "meta-llama/Llama-3.1-8B-Instruct")


def build_model(cfg):
    if cfg.model_name == "qwen2.5-math-7b":
        return build_qwen_model_info(cfg)
    if cfg.model_name == "llama-3.1-8b-instruct":
        return build_llama_model_info(cfg)
    raise ValueError(f"Unsupported main-experiment model: {cfg.model_name}")


def save_pretrained(save_directory, lmm, encoder):
    os.makedirs(save_directory, exist_ok=True),
    encoder_sd = {
        k: v for k, v in encoder.state_dict().items() if not k.startswith("lmm")
    }
    torch.save(encoder_sd, os.path.join(save_directory, "encoder.pth"))
    lmm.model.save_pretrained(save_directory)


def load_from_pretrained(save_directory, lmm, encoder):
    sd = torch.load(os.path.join(save_directory, "encoder.pth"), weights_only=True)
    if sd:
        missing_keys, unexpected_keys = encoder.load_state_dict(sd, strict=False)
        # keys started with "lmm" is not related to shift encoder
        missing_keys = [k for k in missing_keys if not k.startswith("lmm")]
        if missing_keys:
            raise RuntimeError(f"Missing key(s) in state_dict: {missing_keys}")
    if os.path.exists(os.path.join(save_directory, "adapter_config.json")):
        lmm.model = PeftModel.from_pretrained(encoder.lmm.model, save_directory)


def get_expand_runname(cfg: DictConfig):
    """
    Get the expanded runname based on the train, eval or analyze config.
    """
    if hasattr(cfg, "runname"):
        # training mode
        if cfg.data.num_shot == 0:
            # peft
            return f"{cfg.runname}-{cfg.model_name}-{cfg.data.name}-{cfg.data.num_query_samples}"
        # runname-model-dataset-training_samples-num_shot
        return f"{cfg.runname}-{cfg.model_name}-{cfg.data.name}-{cfg.data.num_query_samples}-{cfg.data.num_shot}shot"

    if hasattr(cfg, "record_dir"):
        # analyze mode
        # record dir format: path/to/{expand-runname}
        return os.path.basename(cfg.record_dir)

    if getattr(cfg, "ckpt_path", None):
        # eval from a specific checkpoint
        # checkpoint dir format: path/to/{expand-runname}/epoch-{epoch}
        return os.path.basename(os.path.dirname(cfg.ckpt_path))
    else:
        # eval from ICL
        # icl-model-dataset-[num_shot]shot-[cot/direct]
        cot_suffix = "cot" if cfg.use_cot else "direct"
        return f"icl-{cfg.model_name}-{cfg.data.name}-{cfg.data.num_shot}shot-{cot_suffix}"


def get_extract_answer_function(dataset_name):
    from src.dataset_utils.answers import numeric_answer, math_answer, choice_answer, boolean_answer
    from src.dataset_utils.commonsenseqa import CommonsenseQADataset
    functions = {"gsm8k": numeric_answer, "math_dataset": math_answer, "mmlu-pro": choice_answer,
                 "commonsenseqa": CommonsenseQADataset.extract_answer, "strategyqa": boolean_answer}
    return functions[dataset_name]


def load_model(cfg, device=None):
    from src.qwen_model_wrapper import QwenModelWrapper
    from src.llama_model_wrapper import LlamaModelWrapper

    _, model_path = build_model(cfg)
    wrapper = QwenModelWrapper if cfg.model_name == "qwen2.5-math-7b" else LlamaModelWrapper
    model = wrapper(model_root=model_path, processor_class=AutoTokenizer,
                    model_class=AutoModelForCausalLM, support_models=[cfg.model_name],
                    local_files_only=True, torch_dtype=eval(cfg.dtype),
                    processor_args={"padding_side": "left"}, model_name=cfg.model_name)
    if model.processor.pad_token_id is None:
        model.processor.pad_token = model.processor.eos_token
    model.model.requires_grad_(False)
    model.eval()
    return model.to(device) if device is not None else model
