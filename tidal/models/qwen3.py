"""Qwen3 support for the TidalDecode kernel path.

The kernel-path model (tidal/models/llama.py) is config-driven for everything that
differs in Qwen3-14B:
  - per-head QK-RMSNorm (config.qk_norm / model_type=="qwen3")  -> TDAttention
  - YaRN rope (config.rope_scaling type "yarn")                 -> TDAttention._init_rope
  - token-selection schedule (config.tidal_sparse_layer_start / tidal_correction_layer)

Qwen3-14B is structurally a GQA Llama (40 Q heads / 8 KV heads / head_dim 128), so we
reuse the tidal LlamaForCausalLM and only inject the Qwen3-specific config flags. The
checkpoint's q_norm/k_norm weights load into the QKRMSNorm modules by name.

Default correction layer = 21 (Qwen3-14B profiling, sparsity_planner).
"""
from transformers.models.llama.configuration_llama import LlamaConfig

from tidal.models.llama import LlamaForCausalLM


class Qwen3ForCausalLM(LlamaForCausalLM):
    """tidal kernel-path Qwen3. Use Qwen3ForCausalLM.from_pretrained(path, ...)."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        tidal_sparse_layer_start: int = 2,
                        tidal_correction_layer: int = 21,
                        **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = LlamaConfig.from_pretrained(pretrained_model_name_or_path)
        # Qwen3-specific flags consumed by TDAttention / LlamaModel
        config.qk_norm = True
        config.model_type = "qwen3"
        config.tidal_sparse_layer_start = tidal_sparse_layer_start
        config.tidal_correction_layer = tidal_correction_layer
        if getattr(config, "head_dim", None) is None:
            config.head_dim = config.hidden_size // config.num_attention_heads
        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, config=config, **kwargs
        )
