#include "bsk_ops.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

// Note that estimated_indices does not contain the last page
void topk_filtering(torch::Tensor input_value,
							 torch::Tensor input_indices,
							 torch::Tensor d_out,
							 torch::Tensor indices_out,
							 torch::Tensor buf,
							 unsigned int token_budget) {
	#ifdef BSK_TORCH_CHECK
	CHECK_INPUT(input_value); // [num_heads, num_pages]
	CHECK_INPUT(input_indices); // [num_heads, num_pages]
	CHECK_DIM(2, input_value);
	CHECK_DIM(2, input_indices);
	#endif

	auto num_heads = input_value.size(0);
	auto kv_len = input_value.size(1);

	#ifdef BSK_TORCH_CHECK
	CHECK_EQ(num_heads, input_indices.size(0));
	CHECK_EQ(input_indices.scalar_type(), torch::kInt32);
	// num_heads (= batch_size for the select-k) is passed at runtime below, so any
	// head count is supported (Llama-7b: 32, Qwen3-14B: 40 query heads, etc.).
	CHECK_EQ(token_budget, d_out.size(1));
	CHECK_EQ(token_budget, indices_out.size(1));
	#endif

	// raft's radix select_k does NOT support nv_bfloat16 (missing IOType vectorized
	// traits). It supports fp16/half. We therefore always run the selection in half:
	// bf16 inputs are converted to half (selection only ranks scores, so half precision
	// is sufficient and keeps the same 2-byte buffer sizing as the fp16 path).
	// NOTE: this kernel (decode_topk) is currently unused — the search path selects
	// top-k via torch.topk in Python — but it must still compile for bf16 models.
	torch::Tensor val = input_value;
	torch::Tensor dout = d_out;
	if (val.scalar_type() == torch::kBFloat16) {
		val = val.to(torch::kHalf);
		dout = torch::empty_like(d_out, d_out.options().dtype(torch::kHalf));
	}
	TORCH_CHECK(val.scalar_type() == torch::kHalf,
				"topk_filtering supports fp16/bf16, got ", input_value.scalar_type());
	decode_select_k<nv_half, int32_t>(
		static_cast<nv_half*>(val.data_ptr()),
		static_cast<int32_t*>(input_indices.data_ptr()),
		static_cast<char*>(buf.data_ptr()),
		kv_len,
		token_budget,
		static_cast<nv_half*>(dout.data_ptr()),
		static_cast<int32_t*>(indices_out.data_ptr()),
		static_cast<int>(num_heads),
		true);
}