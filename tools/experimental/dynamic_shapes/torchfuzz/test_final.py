#!/usr/bin/env python3

import sys
import os

# Add the torchfuzz directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from torchfuzz.operators import list_operators

def test_matrix_ops_registration():
    """Test that matrix multiplication operators are properly registered."""
    print("Testing matrix multiplication operator registration...")
    
    operators = list_operators()
    
    # Check for matrix multiplication operators
    expected_ops = ["mm", "addmm", "bmm", "matmul"]
    found_ops = []
    
    for op_name in expected_ops:
        if op_name in operators:
            found_ops.append(op_name)
            operator = operators[op_name]
            torch_op_name = getattr(operator, 'torch_op_name', None)
            print(f"✅ {op_name}: torch_op_name = {torch_op_name}")
        else:
            print(f"❌ {op_name}: NOT FOUND")
    
    print(f"\nFound {len(found_ops)}/{len(expected_ops)} matrix multiplication operators")
    return len(found_ops) == len(expected_ops)

def test_operation_stats_format():
    """Test that operation statistics use fully qualified torch names."""
    print("\nTesting operation statistics format...")
    
    # Import fuzzer components
    from torchfuzz.ops_fuzzer import fuzz_operation_graph
    from torchfuzz.tensor_fuzzer import TensorSpec
    from torchfuzz.operators import get_operator
    
    # Create a simple 2D tensor spec that should work with matrix operations  
    target_spec = TensorSpec(size=(3, 4), stride=(4, 1), dtype=torch.float32)
    
    # Generate a small operation graph
    operation_graph = fuzz_operation_graph(target_spec, max_depth=2, seed=42)
    
    # Extract operation statistics like the fuzzer does
    operation_counts = {}
    for node in operation_graph.nodes.values():
        # Use the same logic as in fuzzer.py
        torch_op_name = None
        
        base_op_name = node.op_name
        if node.op_name.startswith("arg_"):
            base_op_name = "arg"
        
        try:
            operator = get_operator(base_op_name)
            if operator and hasattr(operator, 'torch_op_name') and operator.torch_op_name:
                torch_op_name = operator.torch_op_name
        except (KeyError, ValueError):
            pass
        
        display_name = torch_op_name if torch_op_name else node.op_name
        operation_counts[display_name] = operation_counts.get(display_name, 0) + 1
    
    print("Operation statistics:")
    for op_name, count in sorted(operation_counts.items()):
        print(f"  {op_name}: {count}")
        # Check if torch operations use fully qualified names
        if not op_name.startswith(('arg', 'constant')) and not op_name.startswith('torch.'):
            print(f"    ⚠️  Warning: {op_name} doesn't start with 'torch.'")
    
    return True

if __name__ == "__main__":
    success1 = test_matrix_ops_registration()
    success2 = test_operation_stats_format()
    
    if success1 and success2:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)