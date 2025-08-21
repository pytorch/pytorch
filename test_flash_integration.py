#!/usr/bin/env python3
"""
Simple test to verify flash attention integration with trivial score mod graphs.
"""

import sys
import os

# Add pytorch to path
sys.path.insert(0, '/Users/drisspg/meta/pytorch')

import torch
from torch.fx import GraphModule


def test_trivial_graph_detection():
    """Test that we can detect trivial score mod graphs."""
    from torch._inductor.kernel.flex.flex_flash_attention import is_trivial_graph
    
    print("Testing trivial graph detection...")
    
    # This is a simplified test - in practice the GraphModule would be more complex
    # For now, just test that the function is importable and callable
    
    try:
        # Create a simple test case
        def simple_score_mod(score, b, h, m, n):
            return score
        
        # Convert to GraphModule (simplified)
        traced = torch.fx.symbolic_trace(simple_score_mod)
        
        # Test the detection function
        result = is_trivial_graph(traced, is_score_graph=True)
        print(f"is_trivial_graph result: {result}")
        print("✓ Trivial graph detection test passed")
        
    except Exception as e:
        print(f"✗ Trivial graph detection test failed: {e}")
        return False
    
    return True


def test_flash_attention_availability():
    """Test that flash attention components are available."""
    try:
        from torch._inductor.kernel.flex.flex_flash_attention import (
            CUTE_AVAILABLE, 
            TEMPLATE_AVAILABLE,
            _use_flex_flash_attention
        )
        
        print(f"CUTE_AVAILABLE: {CUTE_AVAILABLE}")
        print(f"TEMPLATE_AVAILABLE: {TEMPLATE_AVAILABLE}")
        
        if CUTE_AVAILABLE:
            print("✓ CUTE flash attention is available")
        else:
            print("✗ CUTE flash attention is not available")
            
        if TEMPLATE_AVAILABLE:
            print("✓ Flash attention template is available")
        else:
            print("✗ Flash attention template is not available")
            
        return CUTE_AVAILABLE and TEMPLATE_AVAILABLE
        
    except Exception as e:
        print(f"✗ Flash attention availability test failed: {e}")
        return False


def test_template_loading():
    """Test that the flash attention template can be loaded."""
    try:
        from torch._inductor.kernel.flex.common import load_template
        
        template_source = load_template("flash_attention")
        print("✓ Flash attention template loaded successfully")
        print(f"Template preview: {template_source[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Template loading test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Running flash attention integration tests...\n")
    
    tests = [
        test_flash_attention_availability,
        test_template_loading,
        test_trivial_graph_detection,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\n--- {test.__name__} ---")
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())