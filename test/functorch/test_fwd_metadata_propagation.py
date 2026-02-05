# Owner(s): ["module: functorch"]
import torch
from functorch.compile import default_partition
from torch._inductor.custom_graph_pass import CustomPartitionerFn, get_hash_for_files
from torch.testing._internal.common_utils import run_tests, TestCase


class MetadataCapturingPartitioner(CustomPartitionerFn):
    """Partition function that captures fw_metadata if provided."""
    
    def __init__(self):
        super().__init__()
        # Avoid storing the fw_metadata to prevent deepcopy failure
        self.read_fw_metadata = False
        self.fw_metadata_has_mutation_info = False
        self.called = False
    
    def __call__(self, gm, joint_inputs, *, num_fwd_outputs, 
                 static_lifetime_input_indices=None, fw_metadata=None, **kwargs):
        self.called = True
        if fw_metadata is not None:
            self.read_fw_metadata = True
            self.fw_metadata_has_mutation_info = any(
                info.mutates_data or info.mutates_metadata
                for info in fw_metadata.input_info
            )
        return default_partition(
            gm, joint_inputs,
            num_fwd_outputs=num_fwd_outputs,
            static_lifetime_input_indices=static_lifetime_input_indices
        )
    
    def uuid(self):
        return get_hash_for_files((__file__,))


class TestPartitionFwMetadata(TestCase):
    def test_partition_receives_fw_metadata(self):
        """Test that custom partition function receives fw_metadata."""
        partitioner = MetadataCapturingPartitioner()
        
        @torch.compile
        def fn(x, y):
            return (x + y).cos()
        
        with torch._inductor.config.patch({"custom_partitioner_fn":partitioner}):
            x = torch.randn(2, 2, requires_grad=True)
            y = torch.randn(2, 2, requires_grad=True)
            out = fn(x, y)
            out.sum().backward()
        
        self.assertTrue(partitioner.called)
        self.assertTrue(partitioner.read_fw_metadata)
    
    def test_partition_with_mutation(self):
        """Test fw_metadata contains mutation information."""
        partitioner = MetadataCapturingPartitioner()
        
        @torch.compile
        def fn_with_mutation(x):
            x.add_(1)
            return x.cos()
        
        with torch._inductor.config.patch({"custom_partitioner_fn": partitioner}):
            x = torch.randn(2, 2, requires_grad=True)
            out = fn_with_mutation(x.clone())
            out.sum().backward()
        
        self.assertTrue(partitioner.called)
        self.assertTrue(partitioner.read_fw_metadata)
        # Check that mutation info is present
        self.assertTrue(partitioner.fw_metadata_has_mutation_info)
    
    def test_backward_compatibility_with_default_partition(self):
        """Test that existing partition functions still work."""
        
        @torch.compile
        def fn(x, y):
            return (x + y).sin()

        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2, requires_grad=True)
        out = fn(x, y)
        out.sum().backward()

if __name__ == "__main__":
    run_tests()
