import unittest
import torch
import os
import torch.distributed as dist
from torch.distributed.distributed_c10d import all_gather_into_tensor  

class TestAllGatherIntoTensor(unittest.TestCase):

    def setUp(self):
        # Set up distributed environment for testing
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        dist.init_process_group("nccl", rank=0, world_size=1)

    def tearDown(self):
        # Clean up distributed environment after testing
        dist.destroy_process_group()

    def test_all_gather_into_tensor_without_kwargs(self):
        # Set up input tensors and other necessary variables
        output_tensor = torch.tensor([1, 2, 3])
        input_tensor = torch.tensor([4, 5, 6])
        group = None
        async_op = False

        # Call the function without additional keyword arguments
        result = all_gather_into_tensor(output_tensor, input_tensor, group, async_op)

        # Add assertions based on the expected behavior of your function
        self.assertIsNone(result) 

        # Assert that the kwargs are not present in output_tensor and input_tensor
        with self.assertRaises(AttributeError):
            _ = output_tensor.kwargs

        with self.assertRaises(AttributeError):
            _ = input_tensor.kwargs

if __name__ == '__main__':
    unittest.main()
