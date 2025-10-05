#!/usr/bin/env python3

"""Debug script to understand wgrad tensor layout requirements"""

import torch
import deep_gemm
from transformer_engine.pytorch.tensor.float8_deepgemm_tensor import FP8DeepGemmQuantizer
from transformer_engine_torch import DType as TE_DType

device = torch.device('cuda')

# Create test tensors matching our actual case
grad_output = torch.randn(128, 512, device=device, dtype=torch.bfloat16)
input_tensor = torch.randn(128, 256, device=device, dtype=torch.bfloat16)

print(f"Original shapes - grad_output: {grad_output.shape}, input: {input_tensor.shape}")

# Test 1: Try different quantization combinations for wgrad = grad_output.T @ input
print("\n=== Testing different quantization strategies ===")

strategies = [
    ("grad_output: rowwise only, input: rowwise only", False, True, False, True),
    ("grad_output: columnwise only, input: columnwise only", True, False, True, False),
    ("grad_output: both, input: both", True, True, True, True),
    ("grad_output: rowwise, input: columnwise", False, True, True, False),  # Like our current attempt
    ("grad_output: columnwise, input: rowwise", True, False, False, True),
]

for desc, go_col, go_row, in_col, in_row in strategies:
    try:
        print(f"\n{desc}")

        # Quantize grad_output (transposed)
        grad_output_transposed = grad_output.t().contiguous()
        print(f"  grad_output transposed: {grad_output_transposed.shape}")

        go_quantizer = FP8DeepGemmQuantizer(
            TE_DType.kFloat8E4M3,
            rowwise=go_row,
            columnwise=go_col,
            use_deepgemm_layout=True,
        )
        go_fp8 = go_quantizer.make_empty(
            grad_output_transposed.shape, dtype=grad_output_transposed.dtype, device=grad_output_transposed.device
        )
        go_quantizer.update_quantized(grad_output_transposed, go_fp8)

        # Quantize input
        in_quantizer = FP8DeepGemmQuantizer(
            TE_DType.kFloat8E4M3,
            rowwise=in_row,
            columnwise=in_col,
            use_deepgemm_layout=True,
        )
        in_fp8 = in_quantizer.make_empty(
            input_tensor.shape, dtype=input_tensor.dtype, device=input_tensor.device
        )
        in_quantizer.update_quantized(input_tensor, in_fp8)

        # Determine which data to use
        if go_row and hasattr(go_fp8, 'rowwise_data'):
            go_data, go_scales = go_fp8.rowwise_data, go_fp8.rowwise_scale_inv
        else:
            go_data, go_scales = go_fp8.columnwise_data, go_fp8.columnwise_scale_inv

        if in_row and hasattr(in_fp8, 'rowwise_data'):
            in_data, in_scales = in_fp8.rowwise_data, in_fp8.rowwise_scale_inv
        else:
            in_data, in_scales = in_fp8.columnwise_data, in_fp8.columnwise_scale_inv

        print(f"  go_data: {go_data.shape}, in_data: {in_data.shape}")

        # Try the computation
        output = torch.empty(512, 256, device=device, dtype=torch.bfloat16)
        deep_gemm.fp8_gemm_nt(
            (go_data, go_scales),
            (in_data, in_scales),
            output,
            c=None,
            recipe=None
        )
        print(f"  ✓ Success! Output: {output.shape}")

        # Verify correctness
        reference = torch.matmul(grad_output_transposed.float(), input_tensor.float())
        diff = torch.abs(output.float() - reference).max().item()
        print(f"  ✓ Max difference from reference: {diff:.6f}")

    except Exception as e:
        print(f"  ✗ Failed: {e}")