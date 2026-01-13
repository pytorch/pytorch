# Minimal OpInfo template for new operators

from torch.testing._internal.common_methods_invocations import OpInfo, SkipInfo
from torch.testing._internal.common_dtype import all_types_and
import torch


# Template for a new OpInfo entry
# Use this as a starting point when adding a new operator to test

example_op_info = OpInfo(
    # Required: unique name of the operation
    name="my_new_op",
    
    # Optional: supported data types
    # Common options:
    #   - all_types()           : all standard types (float32, float64, int32, int64, etc.)
    #   - all_types_and(...)    : specific types to add (e.g., float16, complex)
    #   - (torch.float32, torch.float64)  : explicit tuple
    dtypes=all_types_and(torch.float16),
    
    # Optional: whether operation supports out= argument
    supports_out=True,
    
    # Optional: whether operation has in-place variant (e.g., add_)
    supports_inplace=True,
    
    # Optional: whether operation supports autograd (backward)
    supports_autograd=True,
    
    # Optional: list of skip conditions
    skips=(
        # Skip this op on CPU for unspecified reasons (can be more specific)
        SkipInfo("CPU", "Not optimized for CPU", {}),
        
        # Skip on OpenReg (PrivateUse1 is the dispatch key for custom backends)
        SkipInfo("PrivateUse1", "Not yet implemented on OpenReg backend"),
        
        # Skip with specific dtype filter
        SkipInfo("CUDA", "float16 not supported on this GPU", {"dtype": torch.float16}),
    ),
)

# Example of how to use this in a test file:
# from torch.testing._internal.common_methods_invocations import op_db
# 
# # Add this OpInfo to the global op_db
# op_db.append(example_op_info)
# 
# # Now tests using @ops(op_db) will include this operation:
# @ops(op_db)
# def test_numeric_output(self, device, dtype, op):
#     """Parametrized test that runs for each OpInfo in op_db"""
#     samples = op.sample_inputs(device, dtype)
#     for sample in samples:
#         result = op(sample.input, *sample.args, **sample.kwargs)
#         expected = ...  # CPU reference
#         self.assertTrue(torch.allclose(result, expected))

print("OpInfo template created. Modify as needed and add to op_db.")
