# Owner(s): ["module: functorch"]
import torch
from functorch.compile import default_partition
from torch._inductor.custom_graph_pass import CustomPartitionerFn, get_hash_for_files
from torch.testing._internal.common_utils import run_tests, TestCase


class MetadataCapturingPartitioner(CustomPartitionerFn):
    """Partition function that captures fw_metadata if provided."""
    
    def __init__(self):
        super().__init__()
        self.fw_metadata = None
        self.called = False
    
    def __call__(self, gm, joint_inputs, *, num_fwd_outputs, 
                 static_lifetime_input_indices=None, fw_metadata=None, **kwargs):
        self.called = True
        self.fw_metadata = fw_metadata
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
        
        @torch.compile(backend="aot_eager")
        def fn(x, y):
            return (x + y).cos()
        
        with torch._inductor.config.patch(custom_partitioner_fn=partitioner):
            x = torch.randn(2, 2, requires_grad=True)
            y = torch.randn(2, 2, requires_grad=True)
            out = fn(x, y)
            out.sum().backward()
        
        self.assertTrue(partitioner.called)
        self.assertIsNotNone(partitioner.fw_metadata)
        self.assertGreater(len(partitioner.fw_metadata.input_info), 0)
    
    def test_partition_with_mutation(self):
        """Test fw_metadata contains mutation information."""
        partitioner = MetadataCapturingPartitioner()
        
        @torch.compile(backend="aot_eager")
        def fn_with_mutation(x):
            x.add_(1)
            return x.cos()
        
        with torch._inductor.config.patch(custom_partitioner_fn=partitioner):
            x = torch.randn(2, 2, requires_grad=True)
            out = fn_with_mutation(x.clone())
            out.sum().backward()
        
        self.assertTrue(partitioner.called)
        self.assertIsNotNone(partitioner.fw_metadata)
        # Check that mutation info is present
        has_mutation = any(
            info.mutates_data or info.mutates_metadata
            for info in partitioner.fw_metadata.input_info
        )
        self.assertTrue(has_mutation)
    
    def test_backward_compatibility_without_fw_metadata(self):
        """Test that partition functions without fw_metadata parameter still work."""
        
        def old_style_partition(gm, joint_inputs, *, num_fwd_outputs, 
                               static_lifetime_input_indices=None):
            # Old partition function without fw_metadata parameter
            return default_partition(
                gm, joint_inputs,
                num_fwd_outputs=num_fwd_outputs,
                static_lifetime_input_indices=static_lifetime_input_indices
            )
        
        @torch.compile(backend="aot_eager")
        def fn(x, y):
            return (x + y).sin()
        
        # Should not raise even though partition_fn doesn't accept fw_metadata
        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2, requires_grad=True)
        out = fn(x, y)
        out.sum().backward()
    
    def test_fw_metadata_fields(self):
        """Test that fw_metadata contains expected fields."""
        partitioner = MetadataCapturingPartitioner()
        
        @torch.compile(backend="aot_eager")
        def fn(x, y):
            return x @ y
        
        with torch._inductor.config.patch(custom_partitioner_fn=partitioner):
            x = torch.randn(3, 4, requires_grad=True)
            y = torch.randn(4, 5, requires_grad=True)
            out = fn(x, y)
            out.sum().backward()
        
        metadata = partitioner.fw_metadata
        self.assertIsNotNone(metadata)
        self.assertIsNotNone(metadata.input_info)
        self.assertIsNotNone(metadata.output_info)
        self.assertIsNotNone(metadata.subclass_inp_meta)
        self.assertIsNotNone(metadata.subclass_fw_graph_out_meta)


if __name__ == "__main__":
    run_tests()
