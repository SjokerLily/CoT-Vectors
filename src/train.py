import json
import os
import time

from src import paths
from src.data_module import DataModule
from src.shift_model import ShiftModel, Strategy
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    RichProgressBar, TQDMProgressBar, ModelCheckpoint, DeviceStatsMonitor,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from termcolor import colored
import hydra
from omegaconf import DictConfig
from src.utils import *
from src.qwen_model_wrapper import QwenModelWrapper
from src.llama_model_wrapper import LlamaModelWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer
from pytorch_lightning.profilers import AdvancedProfiler

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(config_path="config", config_name="train.yaml", version_base=None)
def main(cfg: DictConfig):
    def get_max_epochs():
        num_query_samples = cfg.data.num_query_samples
        model_name = cfg.model_name
        if "idefics-9b" in model_name:
            if num_query_samples < 100:
                return 15
            if num_query_samples <= 500:
                return 10
            return 10
        elif "idefics2-8b" in model_name:
            if num_query_samples < 100:
                return 15
            if num_query_samples <= 500:
                return 10
            return 5
        elif "llava" in model_name:
            if num_query_samples <= 500:
                return 10
            return 5
        elif "qwen" in model_name:
            if num_query_samples < 500:
                return 10
            return 5
        elif "llama" in model_name:
            if num_query_samples < 100:
                return 10
            return 5

    def save_when(epoch):
        num_query_samples = cfg.data.num_query_samples
        model_name = cfg.model_name
        if "idefics-9b" in model_name:
            if num_query_samples < 100:
                return epoch >= 10
            if num_query_samples <= 200:
                if cfg.data.name == "coco":
                    return epoch >= 5
                return epoch >= 7
            if num_query_samples <= 500:
                return epoch >= 5
            return epoch >= 5
        elif "idefics2-8b":
            if num_query_samples < 100:
                return epoch >= 10
            if num_query_samples <= 500:
                return epoch >= 5
            return True
        elif "llava" in model_name:
            if num_query_samples <= 1000:
                return epoch >= 5
            return True
        elif "qwen" in model_name:
            if num_query_samples <= 1:
                return epoch >= 5
            return True
        elif "llama" in model_name:
            if num_query_samples <= 1:
                return epoch >= 5
            return True

    max_epochs = cfg.epochs if cfg.epochs else get_max_epochs()
    runname = get_expand_runname(cfg)
    print(colored(f"Training for {runname} on {cfg.model_name}", "light_blue"))

    # Determine the checkpoint path to resume from
    resume_ckpt_path_for_trainer = None

    # Priority 1: Explicitly provided resume_ckpt_path
    if hasattr(cfg, "resume_ckpt_path") and cfg.resume_ckpt_path:
        resume_ckpt_path_for_trainer = cfg.resume_ckpt_path
        if not os.path.exists(resume_ckpt_path_for_trainer):
            raise FileNotFoundError(f"Explicit resume checkpoint not found: {resume_ckpt_path_for_trainer}")
        print(colored(f"Explicitly resuming training from checkpoint: {resume_ckpt_path_for_trainer}", "green"))
    # Priority 2: `cfg.resume` flag (automatically find latest checkpoint for the runname)
    elif cfg.resume:
        save_dir = os.path.join(paths.result_dir, "ckpt", runname)
        if os.path.exists(save_dir):
            exist_ckpt_epochs = [
                int(d.split("-")[-1])
                for d in os.listdir(save_dir)
                if os.path.isdir(os.path.join(save_dir, d)) and "epoch-" in d
            ]
            if exist_ckpt_epochs:
                latest_epoch = max(exist_ckpt_epochs)
                resume_ckpt_path_for_trainer = os.path.join(save_dir, f"epoch-{latest_epoch}")
                print(colored(f"Resuming training from latest checkpoint: {resume_ckpt_path_for_trainer}", "green"))
            else:
                print(f"Resume flag set, but no existing checkpoints found in {save_dir}. Starting new training.")
        else:
            print(f"Resume flag set, but checkpoint directory {save_dir} does not exist. Starting new training.")
            os.makedirs(save_dir, exist_ok=True)
            exist_ckpt_epochs = [
            int(d.split("-")[-1])
            for d in os.listdir(save_dir)
            if os.path.exists(os.path.join(save_dir, d))
        ]
            for i in range(max_epochs):
                if save_when(i) and i not in exist_ckpt_epochs:
                    break
            else:
                print(f"All checkpoints {runname} matched, skip...")
                return
    else:
        print("Starting new training (no resume specified).")
        print("Training as ", cfg.use_cot_vector_type.upper(), "at LAYER ", cfg.only_shift_at_layer)

    pl.seed_everything(cfg.data.seed)
    os.makedirs(paths.result_dir, exist_ok=True)

    # wb_logger = WandbLogger(
    #     save_dir=paths.result_dir,
    #     name=runname,
    #     project="Gsm8kCoTVector",
    #     log_model=False,
    # )
    torch.set_float32_matmul_precision("medium")
    val_check_interval = 0.25 if max_epochs <= 1 else None
    # check_val_every_n_epoch = 1 if max_epochs <= 5 else 3
    check_val_every_n_epoch = 5
    prof = AdvancedProfiler()

    trainer = pl.Trainer(
        # logger=wb_logger,
        callbacks=[
            LearningRateMonitor(),
            TQDMProgressBar(),
            # RichProgressBar(),
            # DeviceStatsMonitor(),
        ],
        # fast_dev_run=True,
        # devices=1,
        max_epochs=max_epochs,
        devices=len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")),
        use_distributed_sampler=False,
        strategy=cfg.strategy,
        precision=cfg.precision,
        gradient_clip_val=cfg.grad_clip_val,
        log_every_n_steps=2,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        # enable_checkpointing=False,
        # val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        # resume_from_checkpoint=resume_ckpt_path_for_trainer,
        # profiler=prof,
    )

    model_info = build_model(cfg)
    
    if "qwen" in cfg.model_name:
        qwen_hf_id, model_path = model_info
        lmm = QwenModelWrapper(
            model_root=model_path,
            processor_class=AutoTokenizer,
            model_class=AutoModelForCausalLM,
            support_models=[qwen_hf_id],
            local_files_only=True,
            torch_dtype=eval(cfg.dtype),
            processor_args={"padding_side": "left"},
            model_args={"output_hidden_states": True},
            model_name=cfg.model_name
        )
    elif "llama-3.1-8b-instruct" in cfg.model_name:
        llama_hf_id,model_path = build_model(cfg)
        lmm = LlamaModelWrapper(
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
        )
        if lmm.processor.pad_token is None:
            lmm.processor.pad_token = lmm.processor.eos_token
    else:
        lmm = model_info

    convert_to_peft(cfg, lmm)

    cfg.data.max_seq_len = getattr(cfg.data, "max_seq_len", 2048)

    data_module = DataModule(cfg, lmm)

    only_shift_at_layer = getattr(cfg, "only_shift_at_layer", None)
    if isinstance(only_shift_at_layer, str):  # If it's a string, try to parse it as a list
        try:
            # Attempt to parse as JSON list or other JSON type
            parsed_value = json.loads(only_shift_at_layer)
            # Only update if parsing was successful and it's a list or a simple value (like int)
            if isinstance(parsed_value, (list, int, type(None))):
                only_shift_at_layer = parsed_value
        except json.JSONDecodeError:
            # If it's not a valid JSON string, keep it as is (e.g., 'null' or a single integer string)
            pass

    shift_encoder = hydra.utils.instantiate(
        cfg.encoder.cls,
        only_shift_at_layer=only_shift_at_layer,
        _partial_=True)(lmm=lmm)

    model = ShiftModel(
        cfg,
        shift_encoder,
        eval(cfg.encoder.model_strategy),
        save_checkpoint_when=save_when,
    )

    t0 = time.time()
    trainer
    trainer.fit(
        model,
        data_module,
    )
    print("Training time (minutes):", (time.time() - t0) / 60)

    print("Peak memory (MB):", torch.cuda.max_memory_allocated() / 1024 ** 2)


if __name__ == "__main__":
    main()
