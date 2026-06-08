import math
from typing import Optional, Tuple
import torch.nn.functional as F
from torch import nn
import torch
from transformers.models.llama.modeling_llama import (
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache


def qwen3_tidal_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    top_k: int = None,
    sparse_layer_start=2,
    correction_layer=9,
    **kwargs,
):
    # prefilling: as full-weight attention
    # generation:
    # - non-sparse layers: full-weight attention #1
    # - sparse_layer_start: full-weight attention + top_k selection #2
    # - sattn_layer_start -> correction layer - 1: use the same top-k #3
    # - correction layer: full-weight attention + new top_k selection
    # - after correction layer: use the same top-k
    if output_attentions:
        # fall back to the original (unpatched) forward
        return self.flash_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

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

    # Qwen3 QK-norm (per-head RMSNorm on head_dim, before RoPE)
    query_states = self.q_norm(query_states)
    key_states = self.k_norm(key_states)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )
    kv_seq_len = past_key_value.get_seq_length(self.layer_idx)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    if self.layer_idx < sparse_layer_start or q_len == kv_seq_len:
        # non-sparse layers or prefilling
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

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
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value
    else:
        # generation
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        last_dim_size = attn_weights.size(-1)
        # If a per-sparse-layer keep-ratio is set (overall-sparsity matching),
        # the token budget scales with the current KV length instead of a fixed top_k.
        keep_ratio = getattr(self, "tidal_keep_ratio", None)
        if keep_ratio is not None:
            token_budget = min(last_dim_size, max(1, math.ceil(keep_ratio * last_dim_size)))
        else:
            token_budget = min(last_dim_size, top_k)

        # decoding
        if self.layer_idx == sparse_layer_start or self.layer_idx == correction_layer:
            # extract top_k mask
            _, top_k_indices = torch.topk(attn_weights[:, :, :, :-1], k=token_budget-1, dim=-1)
            top_k_indices = torch.cat([top_k_indices, torch.tensor([kv_seq_len-1]*self.num_heads).to(top_k_indices.device).reshape(1,self.num_heads,1,1)], dim=-1)
            top_k_mask = torch.zeros_like(attn_weights).scatter_(-1, top_k_indices, 1.0)
            self.pos_dict = top_k_mask  # store top_k mask
        else:
            # apply top_k mask
            if self.pos_dict == None:
                raise ValueError("pos dict should be set up in sparse attn layers")
            min_value = torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights.masked_fill(
                self.pos_dict.to(attn_weights.device) == 0, min_value
            )

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


def enable_qwen3_tidal_attention(
    model, top_k, attn_type="tidal", sparse_layer_start=2, correction_layer=9
):
    def wrap_forward(module):

        def new_tidal_forward(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,
            top_k=top_k,
            sparse_layer_start=sparse_layer_start,
            correction_layer=correction_layer,
            **kwargs,
        ):
            return qwen3_tidal_attention_forward(
                module,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
                top_k=top_k,
                sparse_layer_start=sparse_layer_start,
                correction_layer=correction_layer,
                **kwargs,
            )

        module.flash_forward = module.forward
        if attn_type == "tidal":
            module.forward = new_tidal_forward

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_qwen3_tidal_attention(
                module, top_k, attn_type, sparse_layer_start, correction_layer
            )
        # Matches Qwen3SdpaAttention / Qwen3FlashAttention2 (subclasses of Qwen3Attention)
        base_class = type(module).__bases__[0]
        if (
            base_class.__module__ == "src.models.qwen3_tidaldecoding"
            and base_class.__name__ == "Qwen3Attention"
        ):
            wrap_forward(module)
