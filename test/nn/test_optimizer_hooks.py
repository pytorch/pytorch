# Owner(s): ["module: nn"]
import torch
from torch.optim import SGD

def test_pre_hook():

    def pre_hook(optimizer, empty_args, empty_dict):
        data.append(1)

    data = []
    empty_args = ()
    empty_dict = {}
    params = [torch.Tensor([1, 1])]

    optimizer = SGD(params, lr = 0.001)
    handles = []
    pre_hook_handle = optimizer.register_step_pre_hook(pre_hook)
    handles.append(pre_hook_handle)

    optimizer.step()
    optimizer.step()
    # check if pre hooks were registered
    assert len(data) == 2
    assert data == [1, 1]

    # remove handles, take step and verify that hook is no longer registered
    while len(handles) > 0:
        elt = handles.pop()
        elt.remove()

    optimizer.step()
    assert len(data) == 2

def test_post_hook():

    def post_hook(optimizer, empty_args, empty_dict):
        _ = post_hook_data.pop(0)

    post_hook_data = [1, 3, 5, 7, 9]
    empty_args = ()
    empty_dict = {}
    params = [torch.Tensor([1, 1])]

    optimizer2 = SGD(params, lr = 0.001)
    handles = []
    post_hook_handle = optimizer2.register_step_post_hook(post_hook)
    handles.append(post_hook_handle)

    optimizer2.step()
    optimizer2.step()

    # check if post hooks were registered
    assert len(post_hook_data) == 3
#    assert post_hook_data == [2, 2]

    # remove handles, take step and verify that hook is no longer registered
    while len(handles) > 0:
        elt = handles.pop()
        elt.remove()

    optimizer2.step()
    assert len(post_hook_data) == 3

def test_pre_and_post_hook():

    def pre_hook(optimizer, empty_args, empty_dict):
        data.append(1)

    def post_hook(optimizer, empty_args, empty_dict):
        data.append(2)

    data = []
    empty_args = ()
    empty_dict = {}
    params = [torch.Tensor([1, 1])]

    optimizer = SGD(params, lr = 0.001)

    # register pre and post hook functions
    pre_hook_handle = optimizer.register_step_pre_hook(pre_hook)
    post_hook_handle = optimizer.register_step_post_hook(post_hook)
    pre_hook_handles = []
    post_hook_handles = []
    pre_hook_handles.append(pre_hook_handle)
    post_hook_handles.append(post_hook_handle)

    optimizer.step()
    optimizer.step()
    # check if pre hooks were registered
    assert len(data) == 4
    assert data == [1, 2, 1, 2]

    # remove handles, take step and verify that hooks are no longer registered
    while len(pre_hook_handles) > 0:
        elt = pre_hook_handles.pop()
        elt.remove()

    while len(post_hook_handles) > 0:
        elt = post_hook_handles.pop()
        elt.remove()

    optimizer.step()
    assert len(data) == 4
