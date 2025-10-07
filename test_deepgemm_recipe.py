#!/usr/bin/env python3

"""
Test script demonstrating DeepGEMMFP8Recipe usage and functionality
"""

import torch
import warnings
from transformer_engine.pytorch import fp8_autocast
from transformer_engine.pytorch.module.linear_deepgemm import LinearDeepGemm
from transformer_engine.pytorch.module.layernorm_linear_deepgemm import LayerNormLinearDeepGemm
from transformer_engine.pytorch.module.grouped_linear_deepgemm import GroupedLinearDeepGemm

# Import our DeepGEMM recipe
import sys
sys.path.append('/data/congkai/TransformerEngine')
from deepgemm_recipe import (
    DeepGEMMFP8Recipe,
    deepgemm_precision_recipe,
    deepgemm_performance_recipe,
    deepgemm_megatron_recipe
)

def test_recipe_basic_usage():
    """Test basic DeepGEMMFP8Recipe usage"""
    print("🧪 Testing DeepGEMMFP8Recipe Basic Usage...")

    # Test parameters
    batch_size = 256  # Divisible by 128
    in_features = 512  # Divisible by 128
    out_features = 768  # Not divisible by 128 - should be padded to 896
    device = 'cuda'

    # Create recipe
    recipe = DeepGEMMFP8Recipe()
    print(f"   ✓ Created recipe: {recipe}")

    # Verify recipe compatibility
    if recipe.is_compatible_with_deepgemm():
        print(f"   ✓ Recipe is compatible with DeepGEMM operations")
    else:
        print(f"   ❌ Recipe not compatible with DeepGEMM operations")
        return False

    # Create model - pad out_features to be divisible by 128
    out_features_padded = ((out_features + 127) // 128) * 128  # 896
    print(f"   📏 Padded out_features from {out_features} to {out_features_padded}")

    model = LinearDeepGemm(
        in_features=in_features,
        out_features=out_features_padded,
        accumulate_into_main_grad=True,
        use_deepgemm=True,
        device=device,
        dtype=torch.bfloat16
    )

    # Set up main_grad for fp32 accumulation
    model.weight.main_grad = torch.zeros_like(model.weight, dtype=torch.float32)

    # Create input
    input_tensor = torch.randn(batch_size, in_features, device=device, dtype=torch.bfloat16, requires_grad=True)

    try:
        # Test with recipe
        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            output = model(input_tensor)
            loss = output.sum()
            loss.backward()

        print(f"   ✓ Forward/backward pass successful")
        print(f"   ✓ Output shape: {output.shape}")
        print(f"   ✓ main_grad shape: {model.weight.main_grad.shape}")
        print(f"   ✓ main_grad dtype: {model.weight.main_grad.dtype}")
        print(f"   ✓ main_grad norm: {model.weight.main_grad.norm().item():.6f}")

        return True

    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_recipe_configurations():
    """Test different recipe configurations"""
    print("\n🔧 Testing Recipe Configurations...")

    configs = [
        ("Precision Recipe", deepgemm_precision_recipe()),
        ("Performance Recipe", deepgemm_performance_recipe()),
        ("Megatron Recipe", deepgemm_megatron_recipe()),
    ]

    for name, recipe in configs:
        print(f"\n   Testing {name}:")
        print(f"     Config: {recipe}")
        print(f"     Compatible: {recipe.is_compatible_with_deepgemm()}")

        # Test configuration retrieval
        config = recipe.get_deepgemm_config()
        print(f"     DeepGEMM Config: {config}")

        # Test quantizer kwargs
        input_kwargs = recipe.get_quantizer_kwargs('input')
        weight_kwargs = recipe.get_quantizer_kwargs('weight')
        print(f"     Input quantizer: {input_kwargs['rowwise']}/{input_kwargs['columnwise']}")
        print(f"     Weight quantizer: {weight_kwargs['rowwise']}/{weight_kwargs['columnwise']}")


def test_recipe_with_all_modules():
    """Test recipe with all DeepGEMM modules"""
    print("\n🚀 Testing Recipe with All DeepGEMM Modules...")

    # Common parameters (all divisible by 128)
    batch_size = 256
    seq_length = 128
    in_features = 512
    out_features = 768  # Will be padded to 896
    device = 'cuda'

    # Use precision recipe for comprehensive testing
    recipe = deepgemm_precision_recipe()

    modules_to_test = [
        {
            "name": "LinearDeepGemm",
            "module": LinearDeepGemm(
                in_features=in_features,
                out_features=896,  # Pre-padded
                accumulate_into_main_grad=True,
                device=device,
                dtype=torch.bfloat16
            ),
            "input_shape": (batch_size, in_features)
        },
        {
            "name": "LayerNormLinearDeepGemm",
            "module": LayerNormLinearDeepGemm(
                in_features=in_features,
                out_features=896,  # Pre-padded
                use_deepgemm=True,
                device=device,
                dtype=torch.bfloat16
            ),
            "input_shape": (batch_size, seq_length, in_features)
        },
        {
            "name": "GroupedLinearDeepGemm",
            "module": GroupedLinearDeepGemm(
                num_gemms=2,
                in_features=in_features,
                out_features=896,  # Pre-padded
                accumulate_into_main_grad=True,
                device=device,
                params_dtype=torch.bfloat16
            ),
            "input_shape": (batch_size, in_features),
            "m_splits": [128, 128]  # Each split divisible by 128
        }
    ]

    for module_info in modules_to_test:
        print(f"\n   Testing {module_info['name']}...")

        module = module_info['module']
        input_shape = module_info['input_shape']

        # Set up main_grad for compatible modules
        if hasattr(module, 'weight') and hasattr(module.weight, '__setattr__'):
            module.weight.main_grad = torch.zeros_like(module.weight, dtype=torch.float32)
        elif hasattr(module, 'weight0'):  # GroupedLinearDeepGemm
            for i in range(getattr(module, 'num_gemms', 1)):
                weight = getattr(module, f'weight{i}')
                weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)

        # Create input
        input_tensor = torch.randn(*input_shape, device=device, dtype=torch.bfloat16, requires_grad=True)

        try:
            with fp8_autocast(enabled=True, fp8_recipe=recipe):
                if module_info['name'] == 'GroupedLinearDeepGemm':
                    output = module(input_tensor, module_info['m_splits'])
                else:
                    output = module(input_tensor)

                loss = output.sum()
                loss.backward()

            print(f"     ✅ {module_info['name']}: Forward/backward successful")
            print(f"     ✅ Output shape: {output.shape}")

            # Check main_grad accumulation
            if hasattr(module, 'weight') and hasattr(module.weight, 'main_grad'):
                if module.weight.main_grad is not None:
                    norm = module.weight.main_grad.norm().item()
                    print(f"     ✅ main_grad norm: {norm:.6f}")
            elif hasattr(module, 'weight0'):  # GroupedLinearDeepGemm
                for i in range(getattr(module, 'num_gemms', 1)):
                    weight = getattr(module, f'weight{i}')
                    if hasattr(weight, 'main_grad') and weight.main_grad is not None:
                        norm = weight.main_grad.norm().item()
                        print(f"     ✅ weight{i} main_grad norm: {norm:.6f}")

        except Exception as e:
            print(f"     ❌ {module_info['name']} failed: {e}")
            return False

    return True


def test_recipe_error_handling():
    """Test recipe error handling for invalid configurations"""
    print("\n⚠️ Testing Recipe Error Handling...")

    # Test with invalid dimensions
    device = 'cuda'

    try:
        # Create model with invalid dimensions
        model = LinearDeepGemm(
            in_features=100,  # Not divisible by 128
            out_features=200,  # Not divisible by 128
            device=device,
            dtype=torch.bfloat16
        )

        input_tensor = torch.randn(50, 100, device=device, dtype=torch.bfloat16, requires_grad=True)

        recipe = DeepGEMMFP8Recipe(enforce_dim_constraints=True)

        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            output = model(input_tensor)  # This should fail

        print(f"   ⚠️ Expected error but operation succeeded")
        return False

    except RuntimeError as e:
        if "DeepGEMM requirements not met" in str(e):
            print(f"   ✅ Correctly caught dimension constraint error: {str(e)[:60]}...")
            return True
        else:
            print(f"   ❌ Unexpected error: {e}")
            return False

    except Exception as e:
        print(f"   ❌ Unexpected error type: {e}")
        return False


def main():
    """Run all recipe tests"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Skipping tests.")
        return

    print("🧪 DeepGEMMFP8Recipe Test Suite")
    print("=" * 50)

    tests = [
        ("Basic Usage", test_recipe_basic_usage),
        ("Recipe Configurations", test_recipe_configurations),
        ("All DeepGEMM Modules", test_recipe_with_all_modules),
        ("Error Handling", test_recipe_error_handling),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: FAILED with exception: {e}")

    print(f"\n🎯 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🚀 All DeepGEMMFP8Recipe tests PASSED!")
        print("   - Recipe creation and configuration ✅")
        print("   - DeepGEMM module compatibility ✅")
        print("   - fp32 main_grad accumulation ✅")
        print("   - Dimension constraint enforcement ✅")
        print("   - Error handling ✅")
    else:
        print("⚠️ Some recipe tests failed!")


if __name__ == "__main__":
    main()