import enum
from functools import reduce
import json
import os
import sys

from omegaconf import OmegaConf

sys.path.insert(0, "..")
from src import paths
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import get_cosine_schedule_with_warmup

from src.utils import save_pretrained, get_expand_runname,get_extract_answer_function


class Strategy(enum.IntFlag):
    LAYER_WISE_MSE = 2
    LAYER_WISE_COS_SIM = 64  # equivalent to normalized L2 distance
    LOGITS_KL_DIV = 4
    LM_LOSS = 8

    def has_layer_wise(self):
        try:
            self.layer_wise_strategy()
            return True
        except ValueError:
            return False

    def validate(self):
        layer_wise_loss = [
            Strategy.LAYER_WISE_MSE,
            Strategy.LAYER_WISE_COS_SIM,
        ]

        if bin(self & reduce(lambda x, y: x | y, layer_wise_loss)).count("1") > 1:
            raise ValueError(
                f"{[e.name for e in layer_wise_loss]} are mutually exclusive."
            )

    def layer_wise_strategy(self):
        if Strategy.LAYER_WISE_MSE in self:
            return "mse_loss"
        elif Strategy.LAYER_WISE_COS_SIM in self:
            return "cos_sim"
        else:
            raise ValueError("None of layer wise loss strategy is enabled")


class ShiftModel(pl.LightningModule):
    def __init__(
        self,
        cfg,
        shift_encoder,
        strategy: Strategy,
        save_checkpoint_when=None,  # should be a method f(epoch), save the last ckpt by default
    ) -> None:
        super().__init__()
        self.lmm = shift_encoder.lmm
        self.cfg = cfg
        self.shift_encoder = shift_encoder
        strategy.validate()
        self.strategy = strategy
        self.save_checkpoint_when = (
            save_checkpoint_when
            if save_checkpoint_when is not None
            else lambda epoch: epoch == self.trainer.max_epochs - 1
        )
        self.save_dir = os.path.join(paths.result_dir, "ckpt", get_expand_runname(cfg))
        self.best_val_acc = 0

        self.extract_answer = get_extract_answer_function(cfg.data.name)

        peft_names = ["lora", 'prefix-tuning', "ia3"]

        if self.cfg.peft.name in peft_names:
            self.is_lora = True
        else:
            self.is_lora = False

        if self.cfg.only_alignment_loss:
            self.cfg.ce_loss_weight = 0
            self.save_dir += "-only_alignment_loss"
        if self.cfg.only_ce_loss:
            self.cfg.ce_loss_weight = 1.0
            self.cfg.align_loss_weight = 1.0
            self.save_dir += "-1.0_loss"

    def generate_label_mask(self, inputs, num_separator, keep_bos=False):
        """
        Generates label mask which masks tokens before num_separator pad_tokens from given inputs.
        """
        input_ids = inputs["input_ids"]
        batch_size, seq_len = input_ids.shape
        pad_mask = input_ids == self.lmm.processor.pad_token_id
        non_pad_mask = ~pad_mask
        label_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if self.lmm.processor.padding_side == "left":
            bos_position = non_pad_mask.long().argmax(dim=1)

        for i in range(batch_size):
            seq_pad_positions = pad_mask[i].nonzero(as_tuple=False).squeeze(-1)

            if self.lmm.processor.padding_side == "left":
                seq_pad_positions = seq_pad_positions[
                    seq_pad_positions > bos_position[i]
                ]

            num_pads = len(seq_pad_positions)
            if num_pads < num_separator:
                raise ValueError(
                    f"Sequence {i} has fewer pad tokens ({num_pads}) than num_separator ({num_separator})"
                )

            sep_position = seq_pad_positions[num_separator - 1].item()
            label_mask[i, sep_position + 1 :] = True

        label_mask = label_mask & non_pad_mask
        if keep_bos:
            label_mask[torch.arange(batch_size, device=self.device), bos_position] = (
                True
            )

        return label_mask

    def remove_hooks(self, hooks):
        # remove all hooks
        for name, handles in hooks.items():
            if isinstance(handles, list):
                for handle in handles:
                    handle.remove()
            else:
                handles.remove()

    def get_hidden_states(self, query_label_mask):
        """
        Apply query_label_mask to extract query parts from hidden states (shape: num_layer * [batch_size, seq_len, d_model]),
        and convert to batch_size * [num_layer, query_part_len, d_model].
        """
        hidden_states_dict = {}

        for name, attr in vars(self.shift_encoder).items():
            if "hidden_states" in name:
                # [num_layer, batch_size, seq_len, d_model] -> [batch_size, num_layer, seq_len, d_model]
                hidden_states_all_layers = torch.stack(attr).transpose(0, 1) # (B, NumLayers, SeqLen, Dmodel)
                batch_size, num_layer, seq_len, d_model = hidden_states_all_layers.shape

                extracted_hss = []
                for i in range(batch_size):
                    # hs_for_batch_element: (NumLayers, SeqLen, Dmodel) for this batch element
                    hs_for_batch_element = hidden_states_all_layers[i]
                    mask_for_batch_element = query_label_mask[i] # (SeqLen,)

                    # Get the indices of the masked elements and select them
                    extracted_hs_i = hs_for_batch_element.masked_select(mask_for_batch_element[None, :, None]).view(num_layer, -1, d_model)
                    extracted_hss.append(extracted_hs_i)

                hidden_states_dict[name] = extracted_hss # This is now a list of (num_layer, query_part_len, d_model)
                                                                # which can be stacked in calculate_layer_wise_loss
        if not hidden_states_dict:
            raise RuntimeError(
                "Layer wise loss requires to record hidden states, but no any *_hidden_states in shift encoder."
            )

        return hidden_states_dict

    def calculate_layer_wise_loss(self, shift_hidden_states, prefix_hidden_states):
        if Strategy.LAYER_WISE_MSE in self.strategy:
            loss_fn = lambda input, target: F.mse_loss(
                input.float(),  # Cast to float32
                target.float(), # Cast to float32
                reduction="mean",
            )
        elif Strategy.LAYER_WISE_COS_SIM in self.strategy:
            loss_fn = lambda input, target: 1 - torch.mean(
                F.cosine_similarity(
                    input.float(), # Cast to float32
                    target.float(), # Cast to float32
                    dim=-1,
                ),
                dim=1,
            )

        layer_loss = dict()
        for (shift_hs_varname, shift_hs_list), (_, prefix_hs_list) in zip(
            shift_hidden_states.items(), prefix_hidden_states.items()
        ):

            layer_loss[
                shift_hs_varname.replace(
                    "hidden_states", self.strategy.layer_wise_strategy()
                )
            ] = torch.mean(
                torch.stack(
                    [
                        loss_fn(shift_hs, prefix_hs)
                        for shift_hs, prefix_hs in zip(shift_hs_list, prefix_hs_list)
                    ]
                )
            )
        return layer_loss

    def calculate_logits_kl_loss(
        self, shift_logits, prefix_logits, query_label_inputs, prefix_label_mask
    ):
        # Ensure the masks are boolean and on the same device as logits
        query_label_inputs = query_label_inputs.to(shift_logits.device)
        prefix_label_mask = prefix_label_mask.to(prefix_logits.device)

        # This operation creates large intermediate tensors if the masks are not sparse
        selected_shift_logits = shift_logits[query_label_inputs]
        selected_prefix_logits = prefix_logits[prefix_label_mask]

        logits_kl_loss = F.kl_div(
            selected_shift_logits.log_softmax(dim=-1),
            selected_prefix_logits.softmax(dim=-1),
            reduction="batchmean",
            log_target=False,
        )
        return {"logits_kl_loss": logits_kl_loss}

    def forward(self, prefix_inputs, teacher_answer_mask, teacher_cot_mask, student_inputs, student_labels, images):
        pad_token, pad_token_id, bos_token_id, eos_token = (
            self.lmm.processor.pad_token,
            self.lmm.processor.pad_token_id,
            self.lmm.processor.bos_token_id,
            self.lmm.processor.eos_token,
        )
        loss_dict = {"loss": 0.0}
        hooks = self.shift_encoder.register_record_hooks()

        # Step 1: Teacher model forward pass (no_grad, disable_adapter)
        # Teacher input: Q + Full CoT + Answer
        with torch.no_grad(), self.lmm.model.disable_adapter():
            if "offset_mapping" in prefix_inputs.keys():
                del prefix_inputs["offset_mapping"]
            teacher_outputs = self.lmm.model(**prefix_inputs)
            prefix_logits = teacher_outputs["logits"]

            # Use the provided teacher_answer_mask for alignment
            teacher_hidden_state_mask = teacher_answer_mask.bool()

            prefix_hidden_states = (
                self.get_hidden_states(teacher_hidden_state_mask)
                if self.strategy.has_layer_wise()
                else None
            )
            # Use the provided teacher_answer_mask directly for prefix_label_mask
            prefix_label_mask = teacher_answer_mask.bool()

        # Step 2: Student model forward pass (with shift hooks enabled)
        # Student input: 1-shot example + Q + GT_Final_Answer (for labels)
        hooks.update(self.shift_encoder.register_shift_hooks())

        if self.is_lora:
            # 1. 先准备一条全 -100 的 labels
            # cot_only_labels = torch.full_like(
            #     prefix_inputs["input_ids"], -100, dtype=torch.long
            # )
            # # 2. 把 teacher_cot_mask 对应的 token 位置填回真实 id
            # cot_only_labels[teacher_cot_mask] = prefix_inputs["input_ids"][teacher_cot_mask]
            # answer_only_labels = torch.full_like(
            #     prefix_inputs["input_ids"], -100, dtype=torch.long
            # )
            # answer_only_labels[teacher_answer_mask] = prefix_inputs["input_ids"][teacher_answer_mask]
            student_outputs = self.lmm.model(**prefix_inputs, labels=prefix_inputs["input_ids"])
            # student_outputs = self.lmm.model(**prefix_inputs, labels=answer_only_labels)
        else:
            # Supervise answer tokens only; keep alignment masks unchanged.
            ce_labels = student_labels
            if ce_labels.dtype == torch.bool:
                ce_labels = student_inputs["input_ids"].masked_fill(~ce_labels, -100)
            ce_labels = ce_labels.masked_fill(~student_inputs["attention_mask"].bool(), -100)
            student_outputs = self.lmm.model(
                **student_inputs,
                labels=ce_labels
            )
        student_logits = student_outputs["logits"]

        # Calculate student's LM loss (cross-entropy on final answer part)
        if Strategy.LM_LOSS in self.strategy:
            if self.cfg.dft_loss:
                logits = student_outputs["logits"]
                labels = student_inputs["input_ids"]
                # DFT Loss核心实现
                shift_logits = logits[..., :-1, :].contiguous()  # 移除最后一个token的预测
                shift_labels = labels[..., 1:].contiguous()  # 移除第一个token的标签
                # 展平维度 [batch, seq_len, vocab] -> [batch*seq_len, vocab]
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                # 计算每个token的交叉熵损失
                ce_loss = torch.nn.functional.cross_entropy(
                    shift_logits, shift_labels,
                    reduction='none',
                    ignore_index=-100  # 保持您的ignore_index设置
                )
                # DFT关键步骤：加权损失
                probs = torch.softmax(shift_logits, dim=-1)
                target_probs = probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
                dft_loss = ce_loss * target_probs.detach()  # 乘以目标token概率
                # 计算平均损失
                loss_dict["ce_loss"] = dft_loss.mean()
            else:
                loss_dict["ce_loss"] = student_outputs["loss"]

            ce_loss_weight = (
                1.0 if self.strategy == Strategy.LM_LOSS else self.cfg.ce_loss_weight
            )
            loss_dict["loss"] += ce_loss_weight * loss_dict["ce_loss"]

        # Extract student's hidden states for alignment: only the answer tokens
        # The student_labels != -100 identifies the answer tokens (and EOS)
        if student_labels.dtype == torch.bool:
            student_prompt_hidden_state_mask = student_labels.bool()
        else:
            student_prompt_hidden_state_mask = (student_labels != -100) & student_inputs["attention_mask"].bool()
        # for gsm8k, preserve; for mmlu-pro, hide
        # for i in range(student_prompt_hidden_state_mask.shape[0]):
        #     # Find the index of the last True value in the current row
        #     true_indices = student_prompt_hidden_state_mask[i].nonzero(as_tuple=True)[0]
        #     if true_indices.numel() > 0:
        #         last_true_idx = true_indices[-1]
        #         student_prompt_hidden_state_mask[i, last_true_idx] = 0

        shift_hidden_states = (
            self.get_hidden_states(student_prompt_hidden_state_mask.bool())
            if self.strategy.has_layer_wise()
            else None
        )

        self.remove_hooks(hooks)

        # Step 3: Calculate layer-wise alignment loss
        if self.strategy.has_layer_wise():
            if prefix_hidden_states is None or shift_hidden_states is None:
                raise RuntimeError("Hidden states not available for layer-wise loss.")

            layer_loss = self.calculate_layer_wise_loss(
                shift_hidden_states, prefix_hidden_states
            )
            loss_dict.update(layer_loss)
            loss_dict["loss"] += self.cfg.align_loss_weight * sum(layer_loss.values())

        # step 4. calculate the last logits kl div
        if Strategy.LOGITS_KL_DIV in self.strategy:
            logits_kl_loss = self.calculate_logits_kl_loss(
                student_logits,
                prefix_logits,
                student_prompt_hidden_state_mask, # Use student's answer mask
                prefix_label_mask,
            )
            loss_dict.update(logits_kl_loss)
            loss_dict["loss"] += self.cfg.align_loss_weight * sum(
                logits_kl_loss.values()
            )
        # The original KL_DIV loss is not needed for the new approach, as standard CE and layer alignment are used.
        # self.log('train_loss', loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=True,
        #          sync_dist=True)
        # self.log('train_align_loss', loss_dict['ffn_cos_sim'], on_step=True, on_epoch=True, prog_bar=True,
        #          sync_dist=True)
        # self.log('train_ce_loss', loss_dict["ce_loss"], on_step=True, on_epoch=True, prog_bar=True,
        #          sync_dist=True)

        # print(loss_dict["loss"])


        return loss_dict

    def training_step(self, batch, batch_idx):
        # The batch dict now contains: prefix_inputs, teacher_answer_mask, student_inputs, student_labels, images
        # Pass all these to the forward method.
        loss_dict = self.forward(**batch)
        self.log_dict(loss_dict, sync_dist=True, prog_bar=True)

        return loss_dict["loss"]

    def on_train_epoch_end(self):
        # if self.save_checkpoint_when(self.current_epoch) and self.global_rank == 0:
        save_pretrained(
            os.path.join(self.save_dir, f"epoch-{self.current_epoch}"),
            self.lmm,
            self.shift_encoder,
        )

    def on_train_end(self):
        if self.global_rank == 0:
            with open(os.path.join(self.save_dir, "config.json"), "w") as f:
                json.dump(OmegaConf.to_container(self.cfg, resolve=True), f, indent=4)

    def validation_step_logits(self, batch, batch_idx):
        """
        复用 forward，但只算 accuracy。
        """
        # 关闭 shift hooks，让模型保持「学生」状态
        with torch.no_grad():  # 假设封装里有这个上下文
            # loss_dict = self.forward(**batch)        # 仅为了取 student_logits
            student_outputs = self.lmm.model(
                **batch["student_inputs"],
                labels=batch["student_inputs"]["input_ids"]
            )
        student_logits = student_outputs.logits

        # 取出 label
        labels = batch["student_labels"]            # bool mask 或 token id
        if labels.dtype == torch.bool:
            answer_mask = labels
            input_ids = batch["student_inputs"]["input_ids"]
            labels = input_ids.masked_fill(~answer_mask, -100)

        # 计算 token-level acc（答案区域）
        preds = student_logits.argmax(-1)
        mask  = labels != -100
        correct = (preds == labels) & mask
        acc = correct.sum().float() / mask.sum().float()

        self.log("val_acc", acc, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        return {"val_acc": acc}

    def validation_step(self, batch, batch_idx):
        inputs = []
        correct_predictions = 0
        total_samples = 0
        batch_size = len(batch["batch_messages"])
        for messages in batch["batch_messages"]:
            text = self.lmm.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs.append(text)
        try:
            if self.is_lora:
                predictions = self.lmm.generate(
                    inputs,  # inputs is a list of strings (prompts)
                    num_beams=3,
                    max_new_tokens=1024,
                    length_penalty=0.0,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0
                )
            else:
                model_inputs = self.lmm.processor(inputs, return_tensors="pt", padding=True, truncation=True).to(
                    self.device)
                generated_ids = self.lmm.model.base_model.generate(
                    **model_inputs,  # inputs is a list of strings (prompts)
                    num_beams=3,
                    max_new_tokens=1024,
                    length_penalty=0.0,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0
                )
                input_length = model_inputs.input_ids.shape[1]
                predictions = []
                for i, output_ids in enumerate(generated_ids):
                    start_index = input_length if output_ids.shape[0] > input_length else 0
                    generated_text_ids = output_ids[start_index:]
                    predictions.append(self.lmm.processor.decode(generated_text_ids, skip_special_tokens=True))

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception

        for i, pred in enumerate(predictions):
            extracted_pred = self.extract_answer(pred)
            extracted_ground_truth = batch["ground_truths"][i]

            is_correct = (extracted_pred == extracted_ground_truth)
            if is_correct:
                correct_predictions += 1
            total_samples += 1

        acc = correct_predictions / total_samples if total_samples > 0 else 0

        self.log("val_acc", acc, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True,batch_size=batch_size)
        return {"val_acc": acc}

    def on_validation_epoch_end(self):
        # Lightning 自动把每个 step 的 val_acc_step 做平均
        # 取 Lightning 已经帮你算好的 epoch 级平均值
        acc = self.trainer.callback_metrics["val_acc"]
        print(f"\n[Epoch {self.current_epoch}] val_acc = {acc:.4f}")
        if acc > self.best_val_acc:
            self.best_val_acc = acc
            save_pretrained(
                os.path.join(self.save_dir, f"val-best-{self.current_epoch}"),
                self.lmm,
                self.shift_encoder,
            )



    def configure_optimizers(self):
        def filter_decay_params(param_dict, **common_args):
            """filter parameters for optimizer, separate parameters by adding weight_decay or not"""
            non_decay_names = ["bias"]
            non_decay = [
                {
                    "params": [
                        p
                        for n, p in param_dict.items()
                        for name in non_decay_names
                        if name in n
                    ],
                    "weight_decay": 0.0,
                    **common_args,
                }
            ]

            decay = [
                {
                    "params": [
                        p
                        for n, p in param_dict.items()
                        for name in non_decay_names
                        if name not in n
                    ],
                    "weight_decay": self.cfg.weight_decay,
                    **common_args,
                }
            ]

            return [*non_decay, *decay]

        param_dict = {
            n: p for n, p in self.shift_encoder.named_parameters() if p.requires_grad
        }
        if self.cfg.peft.get("scale_lr", None):
            # if scale_lr is provided, separate scale parameters and regular parameters
            # scale parameters will have a different learning rate, which typically is
            # used for LIVE.
            scale_params = {
                n: p for n, p in param_dict.items() if "log_Z1" in n or "scale" in n
            }
            regular_params = {
                n: p for n, p in param_dict.items() if n not in scale_params
            }

            optim_groups = [
                *filter_decay_params(regular_params, lr=self.cfg.lr),
                *filter_decay_params(scale_params, lr=self.cfg.peft.scale_lr),
            ]
        else:
            optim_groups = filter_decay_params(param_dict, lr=self.cfg.lr)

        assert any(
            group["params"] is not None for group in optim_groups if "params" in group
        ), "No parameter to optimize."

        if "deepspeed" in self.cfg.strategy:
            optimizer = DeepSpeedCPUAdam(
                optim_groups,
                weight_decay=self.cfg.weight_decay,
            )
        else:
            optimizer = optim.AdamW(
                optim_groups,
                weight_decay=self.cfg.weight_decay,
            )

        step_batches = self.trainer.estimated_stepping_batches
        warmup_steps = self.cfg.warmup_step
        if isinstance(warmup_steps, float):
            warm_steps = warmup_steps * step_batches
        elif isinstance(warmup_steps, int):
            warm_steps = warmup_steps
        else:
            raise ValueError(
                f"the warm_steps should be int or float, but got {type(warmup_steps)}"
            )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warm_steps, num_training_steps=step_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
