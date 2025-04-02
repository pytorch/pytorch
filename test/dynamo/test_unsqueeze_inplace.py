# Owner(s): ["module: dynamo"]
import torch
import torch._dynamo
import torch._dynamo.test_case


class TestUnsqueezeInplace(torch._dynamo.test_case.TestCase):
    """
    Test that torch methods for view/reshape/unsqueeze/squeeze work consistently
    with torch.compile
    """
    def test_unsqueeze_inplace(self):
        inputs: dict[str, torch.Tensor] = {'v0_0': torch.rand([3], device='cpu')}
        class Model(torch.nn.Module):
            def forward(self, v0_0):
                v1_0 = v0_0.unsqueeze_(dim=1)
                return v1_0

        class TorchTensorModel(torch.nn.Module):
            def forward(self, v0_0):
                v1_0 = torch.Tensor.unsqueeze_(v0_0, dim=1)
                return v1_0

        model = Model().to(torch.device("cpu"))
        torch_tensor_model = TorchTensorModel().to(torch.device("cpu"))

        # Should not crash when compiled, instead inserts
        # graph break
        model_exported = torch.compile(model)(**inputs)
        torch_tensor_model_exported = torch.compile(torch_tensor_model)(**inputs)




if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
