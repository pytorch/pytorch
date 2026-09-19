import torch
from torch.testing._internal.common_utils import TestCase


class TestOptimizerStateIsolation(TestCase):
    def test_state_dict_clones_tensor_state(self):
        p = torch.nn.Parameter(torch.ones(2))
        opt = torch.optim.SGD([p], lr=0.1, momentum=0.9)
        p.grad = torch.ones_like(p)
        opt.step()

        state = opt.state_dict()
        saved = state["state"][0]["momentum_buffer"]
        internal = opt.state[p]["momentum_buffer"]
        self.assertNotEqual(saved.data_ptr(), internal.data_ptr())

        saved.add_(10)
        torch.testing.assert_close(internal, torch.ones(2))

    def test_load_state_dict_clones_tensor_state(self):
        p1 = torch.nn.Parameter(torch.ones(2))
        p2 = torch.nn.Parameter(torch.ones(2))
        opt1 = torch.optim.SGD([p1], lr=0.1, momentum=0.9)
        opt2 = torch.optim.SGD([p2], lr=0.1, momentum=0.9)

        p1.grad = torch.ones_like(p1)
        opt1.step()
        opt2.load_state_dict(opt1.state_dict())

        state1 = opt1.state[p1]["momentum_buffer"]
        state2 = opt2.state[p2]["momentum_buffer"]
        self.assertNotEqual(state1.data_ptr(), state2.data_ptr())

        before = state1.clone()
        p2.grad = torch.full_like(p2, 2.0)
        opt2.step()
        torch.testing.assert_close(state1, before)

    def test_load_state_dict_preserves_non_tensor_state(self):
        p1 = torch.nn.Parameter(torch.ones(2))
        p2 = torch.nn.Parameter(torch.ones(2))
        opt1 = torch.optim.SGD([p1], lr=0.1)
        opt2 = torch.optim.SGD([p2], lr=0.1)
        opt1.state[p1]["tag"] = "cpu"
        opt1.state[p1]["step_count"] = 3

        opt2.load_state_dict(opt1.state_dict())

        self.assertEqual(opt2.state[p2]["tag"], "cpu")
        self.assertEqual(opt2.state[p2]["step_count"], 3)


if __name__ == "__main__":
    torch.testing._internal.common_utils.run_tests()
