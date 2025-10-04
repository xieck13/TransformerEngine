#!/usr/bin/env python3

"""Debug script to check what data our FP8DeepGemmQTensor objects contain"""

import torch
import warnings
warnings.simplefilter("always")

def debug_tensor_data():
    print("Debugging FP8DeepGemmQTensor Data Contents")
    print("=" * 50)

    try:
        # Import all required modules
        import transformer_engine.pytorch
        import transformer_engine_torch as tex
        from transformer_engine_torch import DType as TE_DType

        from transformer_engine.pytorch.tensor.float8_deepgemm_tensor import FP8DeepGemmQuantizer
        print("✓ All imports successful")

        device = torch.device('cuda')

        # Create quantizers like in verify_deepgemm_fix.py
        A_quantizer = FP8DeepGemmQuantizer(
            fp8_dtype=TE_DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,  # A uses per_token (rowwise)
            use_deepgemm_layout=True
        )

        B_quantizer = FP8DeepGemmQuantizer(
            fp8_dtype=TE_DType.kFloat8E4M3,
            rowwise=False,
            columnwise=True,  # B uses per_block (columnwise)
            use_deepgemm_layout=True
        )

        # Create tensors
        M, K, N = 128, 128, 128
        A_tensor = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        B_tensor = torch.randn(K, N, device=device, dtype=torch.bfloat16)

        # Create quantized tensors
        A_quantized = A_quantizer.make_empty(A_tensor.shape, dtype=A_tensor.dtype, device=device)
        B_quantized = B_quantizer.make_empty(B_tensor.shape, dtype=B_tensor.dtype, device=device)

        # Quantize
        A_quantizer.update_quantized(A_tensor, A_quantized)
        B_quantizer.update_quantized(B_tensor, B_quantized)

        print("\nA_quantized tensor inspection:")
        print(f"  Type: {type(A_quantized)}")
        print(f"  _rowwise_data: {A_quantized._rowwise_data is not None} ({A_quantized._rowwise_data.shape if A_quantized._rowwise_data is not None else 'None'})")
        print(f"  _columnwise_data: {A_quantized._columnwise_data is not None} ({A_quantized._columnwise_data.shape if A_quantized._columnwise_data is not None else 'None'})")
        print(f"  _rowwise_scale_inv: {A_quantized._rowwise_scale_inv is not None} ({A_quantized._rowwise_scale_inv.shape if A_quantized._rowwise_scale_inv is not None else 'None'})")
        print(f"  _columnwise_scale_inv: {A_quantized._columnwise_scale_inv is not None} ({A_quantized._columnwise_scale_inv.shape if A_quantized._columnwise_scale_inv is not None else 'None'})")

        print("\nB_quantized tensor inspection:")
        print(f"  Type: {type(B_quantized)}")
        print(f"  _rowwise_data: {B_quantized._rowwise_data is not None} ({B_quantized._rowwise_data.shape if B_quantized._rowwise_data is not None else 'None'})")
        print(f"  _columnwise_data: {B_quantized._columnwise_data is not None} ({B_quantized._columnwise_data.shape if B_quantized._columnwise_data is not None else 'None'})")
        print(f"  _rowwise_scale_inv: {B_quantized._rowwise_scale_inv is not None} ({B_quantized._rowwise_scale_inv.shape if B_quantized._rowwise_scale_inv is not None else 'None'})")
        print(f"  _columnwise_scale_inv: {B_quantized._columnwise_scale_inv is not None} ({B_quantized._columnwise_scale_inv.shape if B_quantized._columnwise_scale_inv is not None else 'None'})")

        # Check what our _get_fp8_data_and_scales function would return
        def _get_fp8_data_and_scales(tensor, columnwise=False):
            if columnwise and tensor._columnwise_data is not None:
                return tensor._columnwise_data, tensor._columnwise_scale_inv
            elif tensor._rowwise_data is not None:
                return tensor._rowwise_data, tensor._rowwise_scale_inv
            else:
                raise ValueError("No suitable FP8 data found in tensor")

        print("\nTesting _get_fp8_data_and_scales:")
        try:
            A_data, A_scales = _get_fp8_data_and_scales(A_quantized, columnwise=False)
            print(f"  A_data: {A_data.shape}, {A_data.dtype}")
            print(f"  A_scales: {A_scales.shape}, {A_scales.dtype}")
        except Exception as e:
            print(f"  A extraction failed: {e}")

        try:
            B_data, B_scales = _get_fp8_data_and_scales(B_quantized, columnwise=True)
            print(f"  B_data: {B_data.shape}, {B_data.dtype}")
            print(f"  B_scales: {B_scales.shape}, {B_scales.dtype}")
        except Exception as e:
            print(f"  B extraction failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_tensor_data()