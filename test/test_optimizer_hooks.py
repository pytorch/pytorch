from typing import Any, Dict, Tuple
import torch
from torch.optim import Optimizer, SGD
from unittest import TestCase

class TestOptimizerHook(TestCase):
    optimizer: SGD

    @classmethod
    def setUpClass(cls):
        params = [torch.Tensor([1, 1])]
        cls.optimizer = SGD(params, lr=0.001)

    def test_post_hook(self):
        """Mutate state of a variable in a post hook
        """
        def post_hook(optimizer: Optimizer, args: Tuple[Any], kwargs: Dict[Any, Any]):
            nonlocal data
            data += 2

        data = 2
        handles = []
        hook_handle = self.optimizer.register_step_post_hook(post_hook)
        handles.append(hook_handle)

        self.optimizer.step()
        self.optimizer.step()
        # check if pre hooks were registered
        assert data == 6

        # remove handles, take step and verify that hook is no longer registered
        while len(handles) > 0:
            elt = handles.pop()
            elt.remove()

        self.optimizer.step()
        assert data == 6

    def test_pre_hook(self):
        """Mutate state of a variable in the pre hook
        """
        def pre_hook(optimizer: Optimizer, args: Tuple[Any], kwargs: Dict[Any, Any]):
            nonlocal data
            data += 2

        data = 5
        handles = []
        hook_handle = self.optimizer.register_step_pre_hook(pre_hook)
        handles.append(hook_handle)

        self.optimizer.step()
        self.optimizer.step()
        # check if pre hooks were registered
        assert data == 9

        # remove handles, take step and verify that hook is no longer registered
        while len(handles) > 0:
            elt = handles.pop()
            elt.remove()

        self.optimizer.step()
        assert data == 9

    def test_pre_and_post_hook(self):
        """Mutate state of a list in the pre hook and post hook.
        """
        def pre_hook(optimizer: Optimizer, args: Tuple[Any], kwargs: Dict[Any, Any]):
            nonlocal data
            data.append(2)

        def post_hook(optimizer: Optimizer, args: Tuple[Any], kwargs: Dict[Any, Any]):
            nonlocal data
            data.append(1)

        data = []
        pre_handles, post_handles = [], []
        pre_handle = self.optimizer.register_step_pre_hook(pre_hook)
        post_handle = self.optimizer.register_step_post_hook(post_hook)
        pre_handles.append(pre_handle)
        post_handles.append(post_handle)

        self.optimizer.step()
        assert data == [2, 1]
        self.optimizer.step()
        assert data == [2, 1, 2, 1]

        while len(pre_handles) > 0:
            elt = pre_handles.pop()
            elt.remove()

        while len(post_handles) > 0:
            elt = post_handles.pop()
            elt.remove()

        self.optimizer.step()
        assert data == [2, 1, 2, 1]
