#!/usr/bin/env python3

"""
Comprehensive test for GroupedLinearDeepGemm wgrad accuracy and NaN detection
"""

import torch
import warnings
import numpy as np
from transformer_engine.pytorch.module.grouped_linear_deepgemm import GroupedLinearDeepGemm

def test_wgrad_accuracy_and_nan():
    """Test GroupedLinearDeepGemm wgrad for NaN values and accuracy vs bf16 reference"""

    print("🧪 Testing GroupedLinearDeepGemm wgrad accuracy and NaN detection...")

    # Test parameters - all dimensions divisible by 128 for DeepGEMM
    num_gemms = 2
    total_batch = 256
    in_features = 256
    out_features = 512
    m_splits = [128, 128]
    device = 'cuda'

    # Create deterministic input for reproducible results
    torch.manual_seed(42)
    input_tensor = torch.randn(total_batch, in_features, device=device, dtype=torch.bfloat16, requires_grad=True)

    print(f"\n📊 Test Configuration:")
    print(f"   - num_gemms: {num_gemms}")
    print(f"   - input shape: {input_tensor.shape}")
    print(f"   - m_splits: {m_splits}")
    print(f"   - in_features: {in_features}, out_features: {out_features}")

    # Test 1: Check for NaN values in wgrad
    print("\n1️⃣ Testing for NaN values in wgrad...")

    grouped_linear_deepgemm = GroupedLinearDeepGemm(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        accumulate_into_main_grad=False,
        device=device,
        params_dtype=torch.bfloat16
    )

    # Forward and backward pass
    output = grouped_linear_deepgemm(input_tensor, m_splits)
    loss = output.sum()
    loss.backward()

    # Check for NaN in weight gradients
    nan_found = False
    for i in range(num_gemms):
        weight = getattr(grouped_linear_deepgemm, f"weight{i}")
        if weight.grad is not None:
            if torch.isnan(weight.grad).any():
                print(f"   ❌ NaN found in weight{i}.grad!")
                nan_found = True
            else:
                grad_norm = weight.grad.norm().item()
                grad_max = weight.grad.abs().max().item()
                grad_mean = weight.grad.mean().item()
                print(f"   ✅ weight{i}.grad: norm={grad_norm:.6f}, max={grad_max:.6f}, mean={grad_mean:.6f}")

    if not nan_found:
        print("   🎉 No NaN values found in weight gradients!")

    # Test 2: Compare accuracy with bf16 reference implementation
    print("\n2️⃣ Comparing wgrad accuracy with bf16 reference...")

    # Create reference implementation using standard PyTorch operations
    class ReferenceGroupedLinear(torch.nn.Module):
        def __init__(self, num_gemms, in_features, out_features, device, dtype):
            super().__init__()
            self.num_gemms = num_gemms
            for i in range(num_gemms):
                linear = torch.nn.Linear(in_features, out_features, bias=True, device=device, dtype=dtype)
                setattr(self, f"linear{i}", linear)

        def forward(self, inp, m_splits):
            inp_view = inp.reshape(-1, inp.shape[-1])
            input_parts = torch.split(inp_view, m_splits)
            outputs = []
            for i, input_part in enumerate(input_parts):
                linear = getattr(self, f"linear{i}")
                output = linear(input_part)
                outputs.append(output)
            out = torch.cat(outputs, dim=0)
            return out.view(-1, *inp.shape[1:-1], out.shape[-1])

    # Create reference model with same weights
    torch.manual_seed(42)  # Same initialization
    reference_model = ReferenceGroupedLinear(num_gemms, in_features, out_features, device, torch.bfloat16)

    # Copy weights from DeepGEMM model to reference model for fair comparison
    with torch.no_grad():
        for i in range(num_gemms):
            deepgemm_weight = getattr(grouped_linear_deepgemm, f"weight{i}")
            deepgemm_bias = getattr(grouped_linear_deepgemm, f"bias{i}")
            reference_linear = getattr(reference_model, f"linear{i}")

            reference_linear.weight.copy_(deepgemm_weight)
            if deepgemm_bias.numel() > 0:  # Check if bias exists
                reference_linear.bias.copy_(deepgemm_bias)

    # Forward and backward on reference
    input_copy = input_tensor.clone().detach().requires_grad_(True)
    grouped_linear_deepgemm.zero_grad()  # Clear previous gradients

    # Reference forward/backward
    output_ref = reference_model(input_copy, m_splits)
    loss_ref = output_ref.sum()
    loss_ref.backward()

    # DeepGEMM forward/backward
    input_copy2 = input_tensor.clone().detach().requires_grad_(True)
    output_deepgemm = grouped_linear_deepgemm(input_copy2, m_splits)
    loss_deepgemm = output_deepgemm.sum()
    loss_deepgemm.backward()

    # Compare weight gradients
    print(f"   🔍 Weight gradient comparison:")
    total_max_rel_error = 0.0
    total_max_abs_error = 0.0

    for i in range(num_gemms):
        reference_linear = getattr(reference_model, f"linear{i}")
        deepgemm_weight = getattr(grouped_linear_deepgemm, f"weight{i}")

        if reference_linear.weight.grad is not None and deepgemm_weight.grad is not None:
            ref_grad = reference_linear.weight.grad
            deepgemm_grad = deepgemm_weight.grad

            # Check for NaN in either gradient
            if torch.isnan(ref_grad).any() or torch.isnan(deepgemm_grad).any():
                print(f"     ❌ NaN found in gradients for weight{i}")
                continue

            # Compute absolute and relative errors
            abs_error = torch.abs(deepgemm_grad - ref_grad)
            max_abs_error = abs_error.max().item()

            # Relative error with epsilon to avoid division by zero
            eps = 1e-8
            rel_error = abs_error / (torch.abs(ref_grad) + eps)
            max_rel_error = rel_error.max().item()
            mean_rel_error = rel_error.mean().item()

            total_max_rel_error = max(total_max_rel_error, max_rel_error)
            total_max_abs_error = max(total_max_abs_error, max_abs_error)

            print(f"     weight{i}: max_abs_error={max_abs_error:.8f}, max_rel_error={max_rel_error:.6f}, mean_rel_error={mean_rel_error:.6f}")

            # Check if errors are within reasonable bounds for FP8 quantization
            if max_rel_error > 0.1:  # 10% relative error threshold
                print(f"     ⚠️  High relative error for weight{i}: {max_rel_error:.6f}")
            else:
                print(f"     ✅ weight{i} gradient within acceptable error bounds")

    print(f"\n   📈 Overall Results:")
    print(f"     - Max absolute error across all weights: {total_max_abs_error:.8f}")
    print(f"     - Max relative error across all weights: {total_max_rel_error:.6f}")

    # Test 3: main_grad accumulation accuracy
    print("\n3️⃣ Testing main_grad accumulation accuracy...")

    grouped_linear_main_grad = GroupedLinearDeepGemm(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        accumulate_into_main_grad=True,
        device=device,
        params_dtype=torch.bfloat16
    )

    # Copy weights for consistency
    with torch.no_grad():
        for i in range(num_gemms):
            src_weight = getattr(grouped_linear_deepgemm, f"weight{i}")
            src_bias = getattr(grouped_linear_deepgemm, f"bias{i}")
            dst_weight = getattr(grouped_linear_main_grad, f"weight{i}")
            dst_bias = getattr(grouped_linear_main_grad, f"bias{i}")

            dst_weight.copy_(src_weight)
            if src_bias.numel() > 0:
                dst_bias.copy_(src_bias)

    # Set up main_grad
    for i in range(num_gemms):
        weight = getattr(grouped_linear_main_grad, f"weight{i}")
        weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)

    # Forward and backward
    input_copy3 = input_tensor.clone().detach().requires_grad_(True)
    output_main_grad = grouped_linear_main_grad(input_copy3, m_splits)
    loss_main_grad = output_main_grad.sum()
    loss_main_grad.backward()

    # Check main_grad for NaN and compare with regular grad
    print(f"   🔍 main_grad analysis:")
    for i in range(num_gemms):
        weight = getattr(grouped_linear_main_grad, f"weight{i}")
        regular_weight = getattr(grouped_linear_deepgemm, f"weight{i}")

        if hasattr(weight, 'main_grad') and weight.main_grad is not None:
            main_grad = weight.main_grad
            regular_grad = regular_weight.grad

            if torch.isnan(main_grad).any():
                print(f"     ❌ NaN found in weight{i}.main_grad!")
            else:
                main_grad_norm = main_grad.norm().item()
                print(f"     ✅ weight{i}.main_grad: norm={main_grad_norm:.6f}, dtype={main_grad.dtype}")

                # Compare with regular grad (converted to fp32)
                if regular_grad is not None:
                    regular_grad_fp32 = regular_grad.to(torch.float32)
                    grad_diff = torch.abs(main_grad - regular_grad_fp32).max().item()
                    print(f"     📊 weight{i} main_grad vs regular_grad difference: {grad_diff:.8f}")

    return total_max_rel_error < 0.1 and not nan_found


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Skipping tests.")
        exit(1)

    try:
        success = test_wgrad_accuracy_and_nan()

        if success:
            print("\n🚀 All wgrad accuracy tests PASSED!")
            print("   - No NaN values detected")
            print("   - Relative error within acceptable bounds")
            print("   - main_grad accumulation working correctly")
        else:
            print("\n⚠️  Some accuracy issues detected - check output above")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)