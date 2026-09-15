import torch


def test_optimizer_state_dict_preserves_alias_structure_without_sharing_storage():
    p = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([p], lr=0.1, momentum=0.9)
    p.grad = torch.tensor(1.0)
    optimizer.step()

    state = optimizer.state[p]
    state["alias"] = state["momentum_buffer"]
    exported = optimizer.state_dict()["state"][0]

    assert exported["momentum_buffer"].data_ptr() == exported["alias"].data_ptr()
    assert exported["momentum_buffer"].data_ptr() != state["momentum_buffer"].data_ptr()


if __name__ == "__main__":
    torch.testing._internal.common_utils.run_tests()
