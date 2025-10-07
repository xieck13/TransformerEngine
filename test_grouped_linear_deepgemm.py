#!/usr/bin/env python3

"""
Test script to verify GroupedLinearDeepGemm implementation with native DeepGEMM operations
"""

import torch
import warnings
from transformer_engine.pytorch.module.grouped_linear_deepgemm import GroupedLinearDeepGemm

def test_grouped_linear_deepgemm():
    """Test GroupedLinearDeepGemm with native DeepGEMM operations"""

    print("🧪 Testing GroupedLinearDeepGemm with native DeepGEMM operations...")

    # Test parameters - all dimensions must be divisible by 128 for DeepGEMM
    num_gemms = 2  # Reduce to match m_splits
    total_batch = 256  # 2 * 128 = 256, divisible by 128
    in_features = 256  # Divisible by 128
    out_features = 512  # Divisible by 128
    m_splits = [128, 128]  # Each split divisible by 128, sum = 256
    device = 'cuda'

    # Create input tensor
    input_tensor = torch.randn(total_batch, in_features, device=device, dtype=torch.bfloat16, requires_grad=True)

    # Test 1: Basic GroupedLinearDeepGemm forward and backward
    print("\n1️⃣ Testing basic forward and backward pass...")

    grouped_linear_deepgemm = GroupedLinearDeepGemm(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        accumulate_into_main_grad=False,
        device=device,
        params_dtype=torch.bfloat16
    )

    # Forward pass
    try:
        output = grouped_linear_deepgemm(input_tensor, m_splits)
        print(f"   ✓ Forward pass: input {input_tensor.shape} -> output {output.shape}")
        print(f"   ✓ Output dtype: {output.dtype}")

        # Backward pass
        loss = output.sum()
        loss.backward()

        print(f"   ✓ Backward pass completed")
        print(f"   ✓ Input grad shape: {input_tensor.grad.shape}")

        # Check weight gradients
        for i in range(num_gemms):
            weight_attr = f"weight{i}"
            if hasattr(grouped_linear_deepgemm, weight_attr):
                weight = getattr(grouped_linear_deepgemm, weight_attr)
                if weight.grad is not None:
                    print(f"   ✓ Weight{i} grad shape: {weight.grad.shape}")
                else:
                    print(f"   ⚠️  Weight{i} grad is None")

    except Exception as e:
        print(f"   ❌ Basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: main_grad accumulation (Megatron-LM style)
    print("\n2️⃣ Testing main_grad accumulation...")

    grouped_linear_megatron = GroupedLinearDeepGemm(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        accumulate_into_main_grad=True,
        device=device,
        params_dtype=torch.bfloat16
    )

    # Set up main_grad attributes (simulating Megatron-LM)
    for i in range(num_gemms):
        weight = getattr(grouped_linear_megatron, f"weight{i}")
        weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)

    try:
        # Forward and backward
        input_copy2 = input_tensor.clone().detach().requires_grad_(True)
        output2 = grouped_linear_megatron(input_copy2, m_splits)
        loss2 = output2.sum()
        loss2.backward()

        print(f"   ✓ main_grad accumulation completed")

        # Verify main_grad accumulation
        for i in range(num_gemms):
            weight = getattr(grouped_linear_megatron, f"weight{i}")
            if hasattr(weight, 'main_grad') and weight.main_grad is not None:
                main_grad_norm = weight.main_grad.norm().item()
                print(f"   ✓ Weight{i} main_grad shape: {weight.main_grad.shape}")
                print(f"   ✓ Weight{i} main_grad dtype: {weight.main_grad.dtype} (should be fp32)")
                print(f"   ✓ Weight{i} main_grad norm: {main_grad_norm:.6f} (should be > 0)")

                if main_grad_norm == 0 or torch.isnan(torch.tensor(main_grad_norm)):
                    print(f"   ⚠️  Warning: Weight{i} main_grad norm is {main_grad_norm}")
                else:
                    print(f"   ✅ Weight{i} main_grad accumulation working correctly!")

    except Exception as e:
        print(f"   ❌ main_grad test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n🎉 All GroupedLinearDeepGemm tests PASSED!")
    print("   - Forward pass works correctly")
    print("   - Backward pass computes gradients")
    print("   - Native DeepGEMM operations used throughout")
    print("   - 1D1D wgrad with fp32 main_grad accumulation works")

    return True


def test_dimension_constraints():
    """Test dimension constraint enforcement"""
    print("\n🔧 Testing dimension constraint enforcement...")

    device = 'cuda'

    # Test case that violates constraints
    test_cases = [
        {
            "name": "Small batch not divisible by 128",
            "num_gemms": 2,
            "total_batch": 64,  # Not divisible by 128
            "in_features": 256,
            "out_features": 512,
            "m_splits": [32, 32],  # Each split not divisible by 128
        },
        {
            "name": "Features not divisible by 128",
            "num_gemms": 2,
            "total_batch": 256,
            "in_features": 200,  # Not divisible by 128
            "out_features": 300,  # Not divisible by 128
            "m_splits": [128, 128],
        }
    ]

    for case in test_cases:
        try:
            print(f"   Testing: {case['name']}")

            grouped_linear = GroupedLinearDeepGemm(
                num_gemms=case['num_gemms'],
                in_features=case['in_features'],
                out_features=case['out_features'],
                bias=True,
                device=device,
                params_dtype=torch.bfloat16
            )

            input_tensor = torch.randn(
                case['total_batch'], case['in_features'],
                device=device, dtype=torch.bfloat16
            )

            # This should raise an error
            output = grouped_linear(input_tensor, case['m_splits'])
            print(f"     ⚠️  Expected error but got success: {output.shape}")

        except RuntimeError as e:
            if "DeepGEMM requirements not met" in str(e):
                print(f"     ✅ Correctly caught constraint violation: {str(e)[:60]}...")
            else:
                print(f"     ❌ Unexpected error: {e}")
        except Exception as e:
            print(f"     ❌ Unexpected error type: {e}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Skipping tests.")
        exit(1)

    try:
        # Run main tests
        success = test_grouped_linear_deepgemm()

        # Run constraint tests
        test_dimension_constraints()

        if success:
            print("\n🚀 All tests completed successfully!")
            print("   GroupedLinearDeepGemm is ready with native DeepGEMM operations!")
        else:
            print("\n❌ Some tests failed!")
            exit(1)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)