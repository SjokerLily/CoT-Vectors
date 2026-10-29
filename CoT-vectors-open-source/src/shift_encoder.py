import enum
from functools import partial
from typing import List, Callable, Dict, Optional, Tuple, Union
from regex import F
import torch
from torch import nn
import re
import torch.utils
from transformers.cache_utils import Cache
from omegaconf import ListConfig
from transformers.models.llama.modeling_llama import logger
from transformers import LlamaForCausalLM, LlamaConfig
import math

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class HookType(enum.Enum):
    TEXT_MODEL_LAYER = enum.auto()
    VISION_MODEL_LAYER = enum.auto()


class ShiftStrategy(enum.IntFlag):
    VECTOR_SHIFT = 1
    RECORD_HIDDEN_STATES = 4
    LEARNABLE_SHIFT_SCALE = 8
    MULTI_HEAD = 16
    RECORD_RAW_ATTN_OUTPUTS = 32
    STATIC_MU_FROM_CONFIG = 64
    EXTRACTED = 128


class BaseHookEncoder(nn.Module):
    def __init__(
        self,
        lmm,
        attn_strategy: ShiftStrategy = ShiftStrategy(0),
        ffn_strategy: ShiftStrategy = ShiftStrategy(0),
    ):
        super().__init__()
        self.attn_strategy = (
            eval(attn_strategy)
            if attn_strategy and eval(attn_strategy)
            else ShiftStrategy(0)
        )
        self.ffn_strategy = (
            eval(ffn_strategy)
            if ffn_strategy and eval(ffn_strategy)
            else ShiftStrategy(0)
        )
        self.lmm = lmm

        if "idefics-9b" in self.lmm.model_name:
            self.lmm_hidden_dim, self.lmm_layers, self.lmm_num_head = (
                lmm.model.config.hidden_size,
                lmm.model.config.num_hidden_layers,
                lmm.model.config.num_attention_heads,
            )
        elif "idefics2-8b" in self.lmm.model_name:
            self.lmm_hidden_dim, self.lmm_layers, self.lmm_num_head = (
                lmm.model.config.text_config.hidden_size,
                lmm.model.config.text_config.num_hidden_layers,
                lmm.model.config.text_config.num_attention_heads,
            )
        elif "llava-interleave" in self.lmm.model_name:
            self.lmm_hidden_dim, self.lmm_layers, self.lmm_num_head = (
                lmm.model.config.text_config.hidden_size,
                lmm.model.config.text_config.num_hidden_layers,
                lmm.model.config.text_config.num_attention_heads,
            )
        elif "qwen" in self.lmm.model_name:
            self.lmm_hidden_dim, self.lmm_layers, self.lmm_num_head = (
                lmm.model.config.hidden_size,
                lmm.model.config.num_hidden_layers,
                lmm.model.config.num_attention_heads,
            )
        elif "llama-3.1-8b-instruct" in self.lmm.model_name.lower():
            self.lmm_hidden_dim = lmm.model.config.hidden_size
            self.lmm_layers = lmm.model.config.num_hidden_layers
            self.lmm_num_head = lmm.model.config.num_attention_heads
        else:
            raise ValueError(f"{self.lmm.model_name} is not supported")

        def parse_strategy(prefix, strategy):
            if ShiftStrategy.RECORD_HIDDEN_STATES in getattr(
                self, f"{prefix}_strategy"
            ):
                setattr(
                    self,
                    f"{prefix}_hidden_states",
                    [[] for _ in range(self.lmm_layers)],
                )

            if ShiftStrategy.LEARNABLE_SHIFT_SCALE in strategy and (
                ShiftStrategy.VECTOR_SHIFT not in strategy
            ):
                raise ValueError(
                    "ShiftStrategy.LEARNABLE_SHIFT_SCALE should be used with ShiftStrategy.USE_VECTOR_SHIFT"
                )

        parse_strategy("attn", self.attn_strategy)
        parse_strategy("ffn", self.ffn_strategy)

    def register_hooks(
        self,
        register_fn_name: str,
        targets: List[Union[str, HookType]],
        hooks: Dict[str, Callable],
    ):
        return {
            name: getattr(self.lmm, register_fn_name)(target, hook_fn)
            for target, (name, hook_fn) in zip(targets, hooks.items())
            if hook_fn is not None
        }

    @property
    def decoder_mlp_name(self) -> str:
        if "idefics-9b" in self.lmm.model_name:
            return r"model\.layers\.\d+\.mlp$"
        elif "idefics2-8b" in self.lmm.model_name:
            return r"model\.text_model\.layers\.\d+\.mlp$"
        elif "llava-interleave" in self.lmm.model_name:
            return r"language_model\\.model\\.layers\\.\\d+\\.mlp$"
        elif "qwen" in self.lmm.model_name:
            return r"model\.layers\.\d+\.mlp$"
        elif "llama" in self.lmm.model_name.lower():  # 添加对 llama 系列的支持
            return r"model\.layers\.\d+\.mlp$"  # Llama 3 使用相同的模式

    @property
    def decoder_self_attn_name(self) -> str:
        if "idefics-9b" in self.lmm.model_name:
            return r"model\.layers\.\d+\.self_attn$"
        elif "idefics2-8b" in self.lmm.model_name:
            return r"model\.text_model\.layers\.\d+\.self_attn$"
        elif "llava-interleave" in self.lmm.model_name:
            return r"language_model\\.model\\.layers\\.\\d+\\.self_attn$"
        elif "qwen" in self.lmm.model_name:
            return r"model\.layers\.\d+\.self_attn$"
        elif "llama" in self.lmm.model_name.lower():  # 添加对 llama 系列的支持
            return r"model\.layers\.\d+\.self_attn$"  # Llama 3 使用相同的模式

    def register_record_hooks(self, **kwargs):
        # NOTE: record hooks should be registered AFTER all hooks
        def record_hook(m, inputs, outputs, module_name, record_varname, **kwargs):
            layer_idx = int(re.findall(r"\d+", module_name)[0])
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            hidden_states, *_ = outputs
            getattr(self, record_varname)[layer_idx] = hidden_states

        return self.register_hooks(
            "register_forward_hook",
            [
                self.decoder_self_attn_name,
                self.decoder_mlp_name,
            ],
            {
                "attn_record_hook": (
                    partial(record_hook, record_varname="attn_hidden_states")
                    if hasattr(self, "attn_hidden_states")
                    else None
                ),
                "ffn_record_hook": (
                    partial(record_hook, record_varname="ffn_hidden_states")
                    if hasattr(self, "ffn_hidden_states")
                    else None
                ),
            },
        )

    def reset_hidden_states(self):
        """Resets the recorded hidden states to clear them for a new batch."""
        if hasattr(self, "attn_hidden_states"):
            self.attn_hidden_states = [[] for _ in range(self.lmm_layers)]
        if hasattr(self, "ffn_hidden_states"):
            self.ffn_hidden_states = [[] for _ in range(self.lmm_layers)]
        if hasattr(self, "raw_attn_outputs_for_extraction"):
            self.raw_attn_outputs_for_extraction = [None for _ in range(self.lmm_layers)]
        if hasattr(self, "teacher_raw_attn_outputs"):
            self.teacher_raw_attn_outputs = [None for _ in range(self.lmm_layers)]
        if hasattr(self, "student_raw_attn_outputs"):
            self.student_raw_attn_outputs = [None for _ in range(self.lmm_layers)]
        if hasattr(self, "current_recording_context"):
            self.current_recording_context = None


class AttnFFNShift(BaseHookEncoder):
    def __init__(
        self,
        lmm,
        attn_strategy: ShiftStrategy = ShiftStrategy(0),
        ffn_strategy: ShiftStrategy = ShiftStrategy(0),
        shift_scale_init_value=None,
        static_mu_value: Optional[float] = None,
        only_shift_at_layer: Optional[int] = None,
    ):
        """
        Add shift to attention or ffn output. It can also capture hidden states for each layer
        to calculate the layer-wise alignment loss.

        Args:
            lmm: the model to apply shift.
            attn_strategy: the strategy for attention shift.
            ffn_strategy: the strategy for ffn shift.
            shift_scale_init_value: the initial value for the learnable shift scale.
        """
        super().__init__(lmm, attn_strategy, ffn_strategy)

        self.only_shift_at_layer = only_shift_at_layer

        if ffn_strategy is None:
            ffn_strategy = ""

        self.controlled_layer = None

        if only_shift_at_layer is not None and "ShiftStrategy.LEARNABLE_SHIFT_SCALE" in ffn_strategy:
            if isinstance(only_shift_at_layer, (list, ListConfig)):
                self.shift_layers = only_shift_at_layer
            else:
                self.shift_layers = [only_shift_at_layer]
        else:
            self.shift_layers = None

        def parse_strategy(prefix, strategy):
            """
            Create shift modules to ffn output or attention output, based on the strategy.
            """
            if ShiftStrategy.MULTI_HEAD in strategy:
                raise ValueError(
                    f" ShiftStrategy.MULTI_HEAD is not supported, since shift is inserted after {prefix} output"
                )
            if ShiftStrategy.VECTOR_SHIFT in strategy:
                if self.shift_layers is None:
                    setattr(
                        self,
                        f"{prefix}_shift",
                        torch.nn.Parameter(
                            torch.empty(self.lmm_layers, self.lmm_hidden_dim, dtype=self.lmm.model.base_model.dtype).normal_(
                                mean=0.0, std=0.01
                            )
                        ),
                    )
                else:
                    setattr(
                        self,
                        f"{prefix}_shift",
                        torch.nn.Parameter(
                            torch.empty(len(self.shift_layers), self.lmm_hidden_dim,
                                        dtype=self.lmm.model.base_model.dtype).normal_(
                                mean=0.0, std=0.01
                            )
                        ),
                    )

                if ShiftStrategy.LEARNABLE_SHIFT_SCALE in strategy:
                    if self.shift_layers is None:
                        setattr(
                            self,
                            f"{prefix}_shift_scale",
                            nn.Parameter(
                                torch.full(
                                    [self.lmm_layers],
                                    (
                                        shift_scale_init_value
                                        if shift_scale_init_value
                                        else 1.0
                                    ),
                                )
                            ),
                        )
                    else:
                        setattr(
                            self,
                            f"{prefix}_shift_scale",
                            nn.Parameter(
                                torch.full(
                                    [len(self.shift_layers)],
                                    (
                                        shift_scale_init_value
                                        if shift_scale_init_value
                                        else 1.0
                                    ),
                                )
                            ),
                        )

                else:
                    self.register_buffer(
                        f"{prefix}_shift_scale", torch.ones(self.lmm_layers)
                    )

        parse_strategy("attn", self.attn_strategy)
        parse_strategy("ffn", self.ffn_strategy)

        if static_mu_value is not None:
            self.static_mu_value = static_mu_value
        else:
            self.static_mu_value = None

    def register_shift_hooks(self, **kwargs):
        return self.register_hooks(
            "register_forward_hook",
            [
                self.decoder_self_attn_name,
                self.decoder_mlp_name,
            ],
            {
                "attn_hook": (
                    self._shift_hook("attn") if hasattr(self, "attn_shift") else None
                ),
                "ffn_hook": (
                    self._shift_hook("ffn") if hasattr(self, "ffn_shift") else None
                ),
            },
        )

    def _shift_hook(self, prefix):
        def hook(m, inputs, outputs, module_name, **kwargs):
            layer_idx = int(re.findall(r"\d+", module_name)[0])
            
            if self.only_shift_at_layer is not None and (
                ((isinstance(self.only_shift_at_layer, ListConfig) or isinstance(self.only_shift_at_layer, list)) and layer_idx not in self.only_shift_at_layer)
                or
                (isinstance(self.only_shift_at_layer, int) and layer_idx != self.only_shift_at_layer)
            ):
                if isinstance(outputs, tuple):
                    return outputs
                else:
                    return outputs

            if isinstance(outputs, tuple):
                hidden_states, *rest = outputs
            else:
                hidden_states = outputs


            shift = getattr(self, f"{prefix}_shift", None)
            shift_scale = getattr(self, f"{prefix}_shift_scale", None)

            if shift is not None:
                if self.shift_layers is not None:
                    # 找到该层在 shift_layers 中的索引
                    layer_idx_in_list = self.shift_layers.index(layer_idx)
                    shift = shift[layer_idx_in_list][None, None, :]
                    shift_scale = shift_scale[layer_idx_in_list]
                else:
                    if self.controlled_layer is not None:
                        shift = shift[self.controlled_layer][None, None, :]
                        shift_scale = shift_scale[self.controlled_layer]
                    else:
                        shift = shift[layer_idx][None, None, :]
                        shift_scale = shift_scale[layer_idx]
                # shifted_states = hidden_states + shift_scale[layer_idx] * shift
                if self.static_mu_value is not None:
                    shift_scale = torch.full_like(shift_scale, self.static_mu_value).to(hidden_states.dtype)
                shifted_states = hidden_states + shift_scale * shift
                hidden_states = (
                    shifted_states
                    / shifted_states.norm(dim=-1, keepdim=True)
                    * hidden_states.norm(dim=-1, keepdim=True)
                )

            if isinstance(outputs, tuple):
                return (hidden_states, *rest)
            else:
                return hidden_states

        return hook


# Copied from transformers.models.idefics.modeling_idefics.IdeficsSelfAttention
def idefics_attn_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states=None,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position=None,
    module_name=None,
    shift_encoder=None,
):
    # if key_value_states are provided this layer is used as a cross-attention layer
    is_cross_attention = self.is_cross_attention or key_value_states is not None
    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    if not is_cross_attention:
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
    else:
        _, kv_len, _ = (
            key_value_states.size()
        )  # Note that, in this case, `kv_len` == `kv_seq_len`
        key_states = (
            self.k_proj(key_value_states)
            .view(bsz, kv_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(key_value_states)
            .view(bsz, kv_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if isinstance(past_key_value, tuple):
            kv_seq_len += past_key_value[0].shape[-2]
        else:
            kv_seq_len += cache_position[0]

    if not is_cross_attention:
        from transformers.models.idefics.modeling_idefics import apply_rotary_pos_emb

        cos, sin = self.rotary_emb(value_states, seq_len=max(kv_seq_len, q_len))
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        if isinstance(past_key_value, tuple):
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            past_key_value = (key_states, value_states) if use_cache else None
        else:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

    if self.qk_layer_norms:
        query_states = self.q_layer_norm(query_states)
        key_states = self.k_layer_norm(key_states)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
    is_causal = (
        True if self.is_causal and attention_mask is None and q_len > 1 else False
    )

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)

    # ------------------------- The following part is newly added ---------------------
    layer_idx = int(re.findall(r"\d+", module_name)[0])
    attn_output = shift_encoder.do_shift(
        layer_idx, query_states, key_states, attn_output
    )
    # ---------------------------------------------------------------------------------

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    attn_weights = None

    return attn_output, attn_weights, past_key_value


# Copied from transformers.models.mistral.modeling_mistral.MistralSpdaAttention
# The latest version of MistralSpdaAttention is not available in the transformers>=4.46 (not tested)
def idefics2_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position=None,
    module_name=None,
    shift_encoder=None,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)

    from transformers.models.mistral.modeling_mistral import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()

    # ------------------------- The following part is newly added ---------------------
    layer_idx = int(re.findall(r"\d+", module_name)[0])
    attn_output = shift_encoder.do_shift(
        layer_idx, query_states, key_states, attn_output
    )
    # ---------------------------------------------------------------------------------

    attn_output = attn_output.view(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


# Copied from transformers.models.llama.modeling_llama.LlamaSdpaAttention
def llava_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value=None,
    cache_position: Optional[torch.LongTensor] = None,
    module_name=None,
    shift_encoder=None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings

    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
    )

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    sliding_window = None
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    # The following part is newly added
    layer_idx = int(re.findall(r"\d+", module_name)[0])
    attn_output = shift_encoder.do_shift(
        layer_idx, query_states, key_states, attn_output
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()

    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


# Copied from transformers.models.llama.modeling_llama.LlamaSdpaAttention
def qwen_attn_forward(
        self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    module_name=None,
    shift_encoder=None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # Handle output_attentions fallback, if it's set to True, this attention implementation does not support it directly
    if output_attentions:
        raise ValueError("output_attentions=True is not supported in the patched Qwen2 attention. Please set it to False.")

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # Replicate RoPE application logic
    if position_embeddings is None:
        # Original Qwen2SdpaAttention warns here, but we don't need to
        cos, sin = self.rotary_emb(value_states, position_ids) # Use position_ids if embeddings not provided
    else:
        cos, sin = position_embeddings

    from transformers.models.mistral.modeling_mistral import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # Dynamically calculate num_key_value_groups from the attention module (self here is the attention module)
    num_key_value_groups = self.num_heads // self.num_key_value_heads # Already in current code, keep.

    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)

    # Replicate attention_mask slicing
    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

    # Contiguity for CUDA is already in current code, keeping it as is.
    if query_states.device.type == "cuda" and attention_mask is not None: # Use original attention_mask for this check
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # Replicate is_causal determination
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask, # Use causal_mask
        dropout_p=self.attention_dropout if self.training else 0.0, # Exact dropout logic
        is_causal=is_causal, # Dynamic is_causal
    )

    # --- NEW: Capture raw attention output for extraction based on context ---
    layer_idx = int(re.findall(r"\d+", module_name)[0])
    if isinstance(shift_encoder, AttnApproximator) and ShiftStrategy.RECORD_RAW_ATTN_OUTPUTS in shift_encoder.attn_strategy:
        if shift_encoder.current_recording_context == "teacher":
            shift_encoder.teacher_raw_attn_outputs[layer_idx] = attn_output.detach().cpu()
        elif shift_encoder.current_recording_context == "student":
            shift_encoder.student_raw_attn_outputs[layer_idx] = attn_output.detach().cpu()
    # -----------------------------------------------------------------------

    # Integrate shift_encoder.do_shift here, as it's meant to modify attn_output.
    layer_idx = int(re.findall(r"\d+", module_name)[0])
    attn_output = shift_encoder.do_shift(
        layer_idx, query_states, key_states, attn_output
    )

    # Replicate exact reshaping and final projection
    # This block should only be executed if AttnApproximator is used with MULTI_HEAD strategy
    # Otherwise, do_shift already handles the reshape to (bsz, q_len, hidden_size)
    if isinstance(shift_encoder, AttnApproximator) and ShiftStrategy.MULTI_HEAD in shift_encoder.attn_strategy:
        attn_output = attn_output.transpose(1, 2).contiguous() # (bsz, q_len, num_heads, head_dim)
        attn_output = attn_output.view(bsz, q_len, self.hidden_size) # (bsz, q_len, hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def llama_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        module_name=None,
        shift_encoder=None,
        **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    # 新增
    # 1. 提取 layer_idx，用于后续操作
    layer_idx = int(re.findall(r"\d+", module_name)[0])

    # 2. 记录原始注意力输出（如果策略开启）
    if isinstance(shift_encoder,
                  AttnApproximator) and ShiftStrategy.RECORD_RAW_ATTN_OUTPUTS in shift_encoder.attn_strategy:
        if shift_encoder.current_recording_context == "teacher":
            shift_encoder.teacher_raw_attn_outputs[layer_idx] = attn_output.detach().cpu()
        elif shift_encoder.current_recording_context == "student":
            shift_encoder.student_raw_attn_outputs[layer_idx] = attn_output.detach().cpu()

    # print(f"Forward - layer_idx: {layer_idx}")
    # print(f"Forward - query_states shape: {query_states.shape}")
    # print(f"Forward - key_states shape: {key_states.shape}")
    # print(f"Forward - attn_output shape: {attn_output.shape}")
    # 3. 调用 do_shift 函数，修改 attn_output
    attn_output = shift_encoder.do_shift(layer_idx, query_states, key_states, attn_output)

    # 新增结束

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, -1)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

class MultiheadLinear(nn.Module):
    def __init__(self, lmm_num_head, lmm_hidden_dim):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(lmm_num_head, lmm_hidden_dim // lmm_num_head).normal_(0, 0.02)
        )
        self.bias = nn.Parameter(torch.zeros([lmm_num_head]))

    def forward(self, x):
        return torch.einsum("btnd,nd->btn", x, self.weight) + self.bias


class MultiheadProjection(nn.Module):
    def __init__(self, lmm_num_head, lmm_hidden_dim):
        super().__init__()
        head_dim = lmm_hidden_dim // lmm_num_head
        self.weight = nn.Parameter(
            torch.empty(lmm_num_head, head_dim, head_dim).normal_(0, 0.02)
        )
        self.bias = nn.Parameter(torch.zeros([lmm_num_head, head_dim]))

    def forward(self, x):
        return torch.einsum("btnd,ndd->btnd", x, self.weight) + self.bias


class AttnApproxHandle:
    def __init__(self, active=False):
        self.active = active

    def remove(self):
        self.active = False


class AttnApproximator(BaseHookEncoder):
    def __init__(
        self,
        lmm,
        attn_strategy: ShiftStrategy = ShiftStrategy.VECTOR_SHIFT,
        ffn_strategy: ShiftStrategy = ShiftStrategy(0),
        shift_scale_init_value=None,
        static_mu_value: Optional[float] = None,
        only_shift_at_layer: Optional[int] = None,
    ):
        """
        Attention heads with learnable shifts and magnitudes for each layer
        to approximate the in-context demonstrations affected terms (Section 3.2).

        Args:
            lmm: the model to apply shift.
            attn_strategy: the strategy for attention shift.
            ffn_strategy: the strategy for ffn shift.
            shift_scale_init_value: the initial value for the learnable shift scale.
            static_mu_value: a constant value for mu if STATIC_MU_FROM_CONFIG strategy is used.
        """
        super().__init__(lmm, attn_strategy, ffn_strategy)
        self.attn_shift_handles = [AttnApproxHandle() for _ in range(self.lmm_layers)]

        self.only_shift_at_layer = only_shift_at_layer

        if only_shift_at_layer is not None and "ShiftStrategy.EXTRACTED" not in attn_strategy:
            if isinstance(only_shift_at_layer, (list, ListConfig)):
                self.shift_layers = only_shift_at_layer
            else:
                self.shift_layers = [only_shift_at_layer]
        else:
            self.shift_layers = None
        
        # --- NEW: Initialize lists for raw attention outputs and a context flag ---
        if ShiftStrategy.RECORD_RAW_ATTN_OUTPUTS in self.attn_strategy:
            self.teacher_raw_attn_outputs = [None for _ in range(self.lmm_layers)]
            self.student_raw_attn_outputs = [None for _ in range(self.lmm_layers)]
        self.current_recording_context = None # Can be "teacher", "student", or None
        # -------------------------------------------------------------------------

        # Store static mu value if the strategy is enabled
        if ShiftStrategy.STATIC_MU_FROM_CONFIG in self.attn_strategy:
            if static_mu_value is None:
                raise ValueError("static_mu_value must be provided if STATIC_MU_FROM_CONFIG strategy is enabled.")
            self.static_mu_value = static_mu_value

        if ShiftStrategy.LEARNABLE_SHIFT_SCALE in self.attn_strategy:
            if self.shift_layers is None:
                self.log_Z1_lin = nn.ModuleList(
                    (
                        MultiheadLinear(self.lmm_num_head, self.lmm_hidden_dim)
                        if ShiftStrategy.MULTI_HEAD in self.attn_strategy
                        else nn.Linear(self.lmm_hidden_dim, 1)
                    )
                    for _ in range(self.lmm_layers)
                )
            else:
                self.log_Z1_lin = nn.ModuleList(
                    (
                        MultiheadLinear(self.lmm_num_head, self.lmm_hidden_dim)
                        if ShiftStrategy.MULTI_HEAD in self.attn_strategy
                        else nn.Linear(self.lmm_hidden_dim, 1)
                    )
                    for _ in range(len(self.shift_layers))
                )

        if ShiftStrategy.VECTOR_SHIFT in self.attn_strategy:
            if self.shift_layers is None:
                self.attn_shift = nn.Parameter(
                    torch.randn(
                        [self.lmm_layers]
                        + (
                            [self.lmm_num_head, self.lmm_hidden_dim // self.lmm_num_head]
                            if ShiftStrategy.MULTI_HEAD in self.attn_strategy
                            else [self.lmm_hidden_dim]
                        ),
                        dtype=self.lmm.model.base_model.dtype
                    )
                    * 0.001
                )
            else:
                self.attn_shift = nn.Parameter(
                    torch.randn(
                        [len(self.shift_layers)]
                        + (
                            [self.lmm_num_head, self.lmm_hidden_dim // self.lmm_num_head]
                            if ShiftStrategy.MULTI_HEAD in self.attn_strategy
                            else [self.lmm_hidden_dim]
                        ),
                        dtype=self.lmm.model.base_model.dtype
                    )
                    * 0.001
                )

        if ShiftStrategy.VECTOR_SHIFT in self.ffn_strategy:
            self.ffn_shift = nn.Parameter(
                torch.randn([self.lmm_layers, self.lmm_hidden_dim]) * 0.001
            )

        # --- NEW: Initialize list for raw attention outputs if strategy is active --- 
        if ShiftStrategy.RECORD_RAW_ATTN_OUTPUTS in self.attn_strategy:
            self.raw_attn_outputs_for_extraction = [None for _ in range(self.lmm_layers)]
        # --------------------------

    def register_shift_hooks(self, **kwargs):
        if ShiftStrategy.VECTOR_SHIFT in self.attn_strategy:
            if not hasattr(self, "attn_forward_replaced"):
                if self.lmm.model_name == "idefics-9b":
                    new_attn_foward = idefics_attn_forward
                elif "idefics2-8b" in self.lmm.model_name:
                    new_attn_foward = idefics2_attn_forward
                elif "llava-interleave" in self.lmm.model_name:
                    new_attn_foward = llava_attn_forward
                elif "qwen" in self.lmm.model_name:
                    new_attn_foward = qwen_attn_forward
                elif "llama-3.1-8b-instruct" in self.lmm.model_name:
                    new_attn_foward = llama_attn_forward
                else:
                    raise ValueError(f"{self.lmm.model_name} is not supported")

                self.lmm.replace_module_method(
                    self.decoder_self_attn_name,
                    "forward",
                    partial(new_attn_foward, shift_encoder=self),
                    strict=False,
                )
                setattr(self, "attn_forward_replaced", True)

            for idx, handle in enumerate(self.attn_shift_handles):
                if self.shift_layers and (idx in self.shift_layers):
                    handle.active = True
                elif ShiftStrategy.EXTRACTED in self.attn_strategy:
                    if idx == self.only_shift_at_layer:
                        handle.active = True
                else:
                    handle.active = False

        registered_hooks = {"attn_hook": self.attn_shift_handles}
        if ShiftStrategy.VECTOR_SHIFT in self.ffn_strategy:

            def hook(m, inputs, outputs, module_name, **kwargs):
                layer_idx = int(re.findall(r"\d+", module_name)[0])

                if isinstance(outputs, tuple):
                    hidden_states, *rest = outputs
                else:
                    hidden_states = outputs

                shift = self.ffn_shift[layer_idx][None, None, :]
                shifted_states = hidden_states + shift
                hidden_states = (
                    shifted_states
                    / shifted_states.norm(dim=-1, keepdim=True)
                    * hidden_states.norm(dim=-1, keepdim=True)
                )

                if isinstance(outputs, tuple):
                    return (hidden_states, *rest)
                else:
                    return hidden_states

            registered_hooks.update(
                self.register_hooks(
                    "register_forward_hook", [self.decoder_mlp_name], {"ffn_hook": hook}
                )
            )

        return registered_hooks

    def do_shift(self, layer_idx, query_states, key_states, attn_output):
        head_dim = self.lmm_hidden_dim // self.lmm_num_head
        bsz, nh, t, nd = query_states.shape
        if self.attn_shift_handles[layer_idx].active:
            # [bsz, nh, t, hd] -> [bsz, t, nh, nd]
            query_states_transposed = query_states.transpose(1, 2)

            if ShiftStrategy.MULTI_HEAD not in self.attn_strategy:
                # [bsz, t, nh, nd] -> [bsz, t, nh * nd]
                query_states_transposed = query_states_transposed.reshape(bsz, t, -1)
                attn_output = attn_output.reshape(bsz, t, -1)

            if ShiftStrategy.LEARNABLE_SHIFT_SCALE in self.attn_strategy:
                # Z1 = \sum{ \exp(x_i X^\top) }
                # calculate Z2 = \sum{ \exp(x_i * \hat{x}^\top) }
                log_Z2 = torch.logsumexp(
                    torch.matmul(query_states, key_states.transpose(-2, -1))
                    / (head_dim ** 0.5),
                    dim=-1,  # [bsz, nh, t, hd] * [bsz, nh, hd, t] -> [bsz, nh, t, t] -> [bsz, nh, t]
                ).transpose(
                    -2, -1
                )  # [bsz, nh, t] -> [bsz, t, nh]
                log_Z2 = log_Z2.to(query_states.dtype) # Cast to the original dtype

                if ShiftStrategy.MULTI_HEAD not in self.attn_strategy:
                    # [bsz, t, nh] -> [bsz, t, 1]
                    log_Z2 = log_Z2.mean(-1, keepdim=True)

                if self.shift_layers is not None:
                    layer_idx_in_list = self.shift_layers.index(layer_idx)
                    log_Z1 = self.log_Z1_lin[layer_idx_in_list](query_states_transposed)
                else:
                    log_Z1 = self.log_Z1_lin[layer_idx](query_states_transposed)
                log_Z1 = log_Z1.to(query_states.dtype) # Cast to the original dtype

                # shape: [bsz, t, nh] or [bsz, t, 1]
                if ShiftStrategy.STATIC_MU_FROM_CONFIG in self.attn_strategy:
                    # Use static mu value if the strategy is enabled
                    mu = torch.full_like(log_Z1, self.static_mu_value).to(query_states.dtype)
                else:
                    # Original mu calculation
                    mu = torch.exp(log_Z1 - torch.logaddexp(log_Z1, log_Z2))
                    mu = mu.to(query_states.dtype) # Cast to the original dtype
                if ShiftStrategy.MULTI_HEAD in self.attn_strategy:
                    # shape: [bsz, t, nh] -> [bsz, t, nh, 1]
                    mu = mu.unsqueeze(-1)

            if hasattr(self, "attn_shift"):
                # shape: [1, 1, nh, nd] or [1, 1, hd * nh]
                if self.shift_layers is not None:
                    layer_idx_in_list = self.shift_layers.index(layer_idx)
                    shift = self.attn_shift[layer_idx_in_list][None, None, :]
                else:
                    shift = self.attn_shift[layer_idx][None, None, :]
                if self.training and hasattr(self, "attn_proj"):
                    shift = self.attn_proj[layer_idx](shift)
                if ShiftStrategy.LEARNABLE_SHIFT_SCALE in self.attn_strategy:
                    # shift := SA(q, K_D, V_D) - SA(q, K, V)
                    if self.only_shift_at_layer is not None and (
                            ((isinstance(self.only_shift_at_layer, ListConfig) or isinstance(self.only_shift_at_layer,
                                                                                             list)) and layer_idx in self.only_shift_at_layer)
                            or
                            (isinstance(self.only_shift_at_layer, int) and layer_idx == self.only_shift_at_layer)
                    ):
                        shift_vector = mu * shift
                        shifted_output = attn_output + shift_vector.transpose(1, 2)
                    else:
                        #shifted_output = attn_output + 0 * shift
                        shifted_output = attn_output
                    return shifted_output

                # # Apply normalization to the shifted output
                # # Ensure the norm is not zero to avoid NaNs
                # epsilon = torch.tensor(1e-6, device=shifted_output.device, dtype=shifted_output.dtype) # Ensure epsilon has correct dtype
                # norm_shifted = shifted_output.norm(dim=-1, keepdim=True)
                # norm_original = attn_output.norm(dim=-1, keepdim=True)
                #
                # attn_output = (
                #     shifted_output
                #     / (norm_shifted + epsilon)
                #     * norm_original
                # ).to(shifted_output.dtype) # Explicitly cast result to float16

            else:
                # never fall in here
                shift = torch.zeros_like(attn_output)

        # attn_output: [bsz, t, nh, nd]
        return attn_output