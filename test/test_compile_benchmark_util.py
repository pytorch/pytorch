import torch
import torch._dynamo as torchdynamo
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils.benchmark.utils.compile import bench_all

torchdynamo.reset()

class TestCompileBenchmarkUtil(TestCase):
    def test_training_and_inference(self):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super(ToyModel, self).__init__()
                self.weight = torch.nn.Parameter(torch.Tensor(2, 2))

            def forward(self, x):
                return x * self.weight

        model = ToyModel().cuda()

        print("===== Inference =====")
        inference_table = bench_all(model, torch.ones(1024, 2, 2).cuda(), 5)
        assert(inference_table.rows[0][0] == "Inference")
        assert(inference_table.rows[0][1] == "Eager")
        assert(inference_table.rows[0][2] == "-")
        print(inference_table)
        print("\n===== Training =====")
        training_table = bench_all(model, torch.ones(1024, 2, 2).cuda(), 5, is_training=True, optimizer=torch.optim.SGD(model.parameters(), lr=0.01))
        assert(training_table.rows[0][0] == "Training")
        assert(training_table.rows[0][1] == "Eager")
        assert(training_table.rows[0][2] == "-")
        print(training_table)

if __name__ == '__main__':
    run_tests()
