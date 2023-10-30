from torch.dict import TensorDict
from torch.testing._internal.common_utils import TestCase, run_tests, parametrize, instantiate_parametrized_tests
import torch.cuda


class TestTensorDicts(TestCase):
    @property
    def td_device(self):
        # A typical tensordict, on device
        if torch.cuda.device_count():
            device = "cuda:0"
        else:
            device = "cpu"
        return TensorDict(
            source={
                "a": torch.randn(4, 3, 2, 1, 5),
                "b": torch.randn(4, 3, 2, 1, 10),
                "c": torch.randint(10, (4, 3, 2, 1, 3)),
            },
            batch_size=[4, 3, 2, 1],
            device=device,
        )

    @property
    def td_no_device(self):
        # A typical tensordict, on device
        if torch.cuda.device_count():
            device = "cuda:0"
        else:
            device = "cpu"
        return TensorDict(
            source={
                "a": torch.randn(4, 3, 2, 1, 5, device=device),
                "b": torch.randn(4, 3, 2, 1, 10, device=device),
                "c": torch.randint(10, (4, 3, 2, 1, 3)),
            },
            batch_size=[4, 3, 2, 1],
        )

    @property
    def td_nested(self):
        # A typical tensordict, on device
        if torch.cuda.device_count():
            device = "cuda:0"
        else:
            device = "cpu"
        return TensorDict(
            source={
                "a": torch.randn(4, 3, 2, 1, 5, device=device),
                "b": torch.randn(4, 3, 2, 1, 10, device=device),
                "c": torch.randint(10, (4, 3, 2, 1, 3)),
                "d": TensorDict({
                    "e": torch.randn(4, 3, 2, 1, 2)
                }, batch_size=[4, 3, 2, 1, 2])
            },
            batch_size=[4, 3, 2, 1],
        )

    @parametrize("td_type", ["td_device", "td_no_device", "td_nested"])
    def test_creation(self, td_type):
        assert getattr(self, td_type) is not None

    @parametrize("td_type", ["td_device", "td_no_device", "td_nested"])
    def test_squeeze_unsqueeze(self, td_type):
        data = getattr(self, td_type)
        data_u = data.unsqueeze(-1)
        assert data_u.shape == torch.Size([4, 3, 2, 1, 1])
        assert data_u.squeeze().shape == torch.Size([4, 3, 2])
        assert data_u.squeeze(0).shape == torch.Size([4, 3, 2, 1, 1])
        assert data_u.squeeze(-1).shape == torch.Size([4, 3, 2, 1])
        data_u = data.unsqueeze(-3)
        assert data_u.shape == torch.Size([4, 3, 1, 2, 1])
        assert data_u.squeeze().shape == torch.Size([4, 3, 2])
        assert data_u.squeeze(0).shape == torch.Size([4, 3, 1, 2, 1])
        assert data_u.squeeze(-3).shape == torch.Size([4, 3, 2, 1])
        data_u = data.unsqueeze(0)
        assert data_u.shape == torch.Size([1, 4, 3, 2, 1])
        assert data_u.squeeze().shape == torch.Size([4, 3, 2])
        assert data_u.squeeze(0).shape == torch.Size([4, 3, 2, 1])
        assert data_u.squeeze(-3).shape == torch.Size([1, 4, 3, 2, 1])
        data_u = data.unsqueeze(2)
        assert data_u.shape == torch.Size([4, 3, 1, 2, 1])
        assert data_u.squeeze().shape == torch.Size([4, 3, 2])
        assert data_u.squeeze(0).shape == torch.Size([4, 3, 1, 2, 1])
        assert data_u.squeeze(2).shape == torch.Size([4, 3, 2, 1])
        for item in data_u.values(include_nested=True):
            assert item.shape[:5] == torch.Size([4, 3, 1, 2, 1])

instantiate_parametrized_tests(TestTensorDicts)


if __name__ == '__main__':
    run_tests()
