# Owner(s): ["oncall: distributed"]

"""
Testing SsdFlatParameter and SsdTensorHandle modules.
"""

import filecmp
import functools
import os
from shutil import rmtree
import tempfile

import numpy as np
import pytest
import torch
import torch.distributed.fsdp.ssd_offload as so
from torch.utils._pytree import tree_map

def _init():
    torch.manual_seed(0)
    np.random.seed(0)


def test_write_read():
    _init()

    with tempfile.NamedTemporaryFile() as f:
        ref_tensor = torch.rand((128), dtype=torch.float32)
        test_tensor = torch.zeros_like(ref_tensor)
        assert not torch.equal(ref_tensor, test_tensor)
        so.TensorSerialization.write(ref_tensor, f.name)
        so.TensorSerialization.read(test_tensor, f.name)
        assert torch.equal(ref_tensor, test_tensor)


def test_ssd_handle_dispatch_fwd():
    _init()

    with tempfile.NamedTemporaryFile() as f:
        orig_tensor = torch.randn((128))
        ssd_handle = so.SsdTensorHandle.from_tensor(orig_tensor)
        ssd_handle.set_file_params(f.name, 0)
        ssd_handle.to_file(release_tensor_after_write=True)

        assert torch.equal(ssd_handle.to_tensor(), orig_tensor)

        # This should trigger the torch_dispatch code and write
        # back the results to the file
        ssd_handle.add_(1)
        plus1_tensor = orig_tensor.add(1)
        assert torch.equal(ssd_handle.to_tensor(), plus1_tensor)


def test_ssd_handle_dispatch_bwd():
    _init()

    with tempfile.NamedTemporaryFile() as f:
        orig_tensor = torch.randn((4, 4), requires_grad=True)
        orig_copy = orig_tensor.clone().detach().requires_grad_(True)
        ssd_handle = so.SsdTensorHandle.from_tensor(orig_tensor)
        ssd_handle.set_file_params(f.name, 0)
        ssd_handle.to_file(release_tensor_after_write=True)

        assert torch.equal(ssd_handle.to_tensor(), orig_tensor)

        y1 = ssd_handle + 1
        y2 = orig_copy + 1
        y1.sum().backward()
        y2.sum().backward()

        assert torch.equal(ssd_handle.grad, orig_copy.grad)


def test_ssd_handle_dispatch_bwd_hook():
    _init()

    def post_backward_hook(name, grad):
        print(f"BACKWARD HOOK for tensor {name} CALLED")

    with tempfile.NamedTemporaryFile() as f:
        orig_tensor = torch.randn((4, 4), requires_grad=True)
        orig_copy = orig_tensor.clone().detach().requires_grad_(True)
        ssd_handle = so.SsdTensorHandle.from_tensor(orig_tensor)
        ssd_handle.set_file_params(f.name, 0)
        ssd_handle.to_file(release_tensor_after_write=True)
        one = torch.ones((1), requires_grad=True).cuda()

        orig_copy = ssd_handle.data
        cuda_copy = ssd_handle.to("cuda").detach().requires_grad_(True)
        ssd_handle.data = cuda_copy

        ssd_handle.register_hook(functools.partial(post_backward_hook, "ssd_handle"))
        one.register_hook(functools.partial(post_backward_hook, "one"))

        y1 = ssd_handle + one
        y1.sum().backward()


def test_ssd_handle_train_simple():
    _init()

    with tempfile.NamedTemporaryFile() as f:
        orig_tensor = torch.randn((4, 4), requires_grad=True)

        with torch.no_grad():
            orig_copy = torch.empty_like(orig_tensor)
            orig_copy.copy_(orig_tensor)
            orig_copy.requires_grad = True

        ssd_handle = so.SsdTensorHandle.from_tensor(orig_tensor)
        ssd_handle.flush_on_dirty = False
        ssd_handle.set_file_params(f.name, 0)
        ssd_handle.to_file(release_tensor_after_write=True)

        assert torch.equal(ssd_handle.to_tensor(), orig_tensor)
        optimizer_ssd = torch.optim.SGD([ssd_handle], lr=0.1)
        optimizer_orig = torch.optim.SGD([orig_copy], lr=0.1)

        y1 = ssd_handle + 1
        optimizer_ssd.zero_grad()
        y1.sum().backward()
        assert ssd_handle.storage_state is so.StorageState.ON_CPU_CLEAN
        optimizer_ssd.step()
        assert ssd_handle.storage_state is so.StorageState.ON_CPU_DIRTY

        y2 = orig_copy + 1
        optimizer_orig.zero_grad()
        y2.sum().backward()
        optimizer_orig.step()

        assert torch.equal(ssd_handle.to_tensor(), orig_copy)

def test_torch_save_load_ssd_flat_param_on_disk():
    _init()
    orig_file = tempfile.NamedTemporaryFile(prefix="tensor")
    checkpoint_file = tempfile.NamedTemporaryFile(prefix="checkpoint", suffix=".pt")
    checkpoint_load_directory = tempfile.TemporaryDirectory(prefix="checkpoint_dir")

    # TENSOR_SHAPE = (1024, 1024, 2048)
    # use smaller shape for unit tests
    TENSOR_SHAPE = (1024, 321)
    ref_tensors = [torch.rand(TENSOR_SHAPE, dtype=torch.float32) for i in range(4)]
    ssd_handle = so.SsdFlatParameter.from_tensors(ref_tensors, False)
    ssd_handle.set_file_params(orig_file.name, 0)
    ssd_handle.to_file()
    ref_tensors = []

    # after deleting ref_tensor, memory usage should be very low
    # For save it shouldn't be more than 10x so.DEFAULT_CHUNK_SIZE
    with so.CheckpointPathContextManager(override_path=checkpoint_load_directory.name):
        so.torch_saver.save(ssd_handle, checkpoint_file.name)
    # below line saves file to checkpoint_load_directory/orig_file.name
    # Memory usage here should be O(1000 * so.DEFAULT_CHUNK_SIZE)
    # 1000x because that's how many elements the python unpickler
    # will buffer before passing to the SsdTensor
    test_ssd_handle = torch.load(checkpoint_file)
    head, tail = os.path.split(orig_file.name)
    assert filecmp.cmp(orig_file.name, os.path.join(checkpoint_load_directory.name, tail), shallow=False)

    orig_file.close()
    checkpoint_file.close()
    rmtree(checkpoint_load_directory.name)
    checkpoint_load_directory.cleanup()


def test_torch_save_load_ssd_flat_param_on_mem():
    _init()
    orig_file = tempfile.NamedTemporaryFile(prefix="tensor")
    checkpoint_file = tempfile.NamedTemporaryFile(prefix="checkpoint", suffix=".pt")
    checkpoint_load_directory = tempfile.TemporaryDirectory(prefix="checkpoint_dir")

    # TENSOR_SHAPE = (1024, 1024, 2048)
    # use smaller shape for unit tests
    TENSOR_SHAPE = (1024, 321)
    ref_tensors = [torch.rand(TENSOR_SHAPE, dtype=torch.float32) for i in range(4)]
    ssd_handle = so.SsdFlatParameter.from_tensors(ref_tensors, False)
    ssd_handle.set_file_params(orig_file.name, 0)
    ref_tensors = []

    # after deleting ref_tensor, memory usage should be very low
    # For save it shouldn't be more than 10x so.DEFAULT_CHUNK_SIZE
    with so.CheckpointPathContextManager(override_path=checkpoint_load_directory.name):
        so.torch_saver.save(ssd_handle, checkpoint_file.name)
    # below line saves file to checkpoint_load_directory/orig_file.name
    # Memory usage here should be O(1000 * so.DEFAULT_CHUNK_SIZE)
    # 1000x because that's how many elements the python unpickler
    # will buffer before passing to the SsdTensor
    test_ssd_handle = torch.load(checkpoint_file)
    assert torch.equal(ssd_handle, test_ssd_handle)
    orig_file.close()
    checkpoint_file.close()
    checkpoint_load_directory.cleanup()


def test_ssd_param_train_simple():
    _init()
    with tempfile.NamedTemporaryFile() as f:
        orig_tensor = torch.randn((4, 4))

        with torch.no_grad():
            orig_copy = torch.empty_like(orig_tensor)
            orig_copy.copy_(orig_tensor)
            param = torch.nn.Parameter(orig_copy)

        ssd_param = so.SsdParameter(orig_tensor.shape, orig_tensor.dtype)
        ssd_param.point_to_tensor(orig_copy)
        ssd_param.flush_on_dirty = False
        ssd_param.set_file_params(f.name, 0)
        ssd_param.to_file(release_tensor_after_write=True)

        assert torch.equal(ssd_param.to_tensor(), orig_tensor)
        optimizer_ssd = torch.optim.SGD([ssd_param], lr=0.1)
        optimizer_orig = torch.optim.SGD([param], lr=0.1)

        y1 = ssd_param + 1
        optimizer_ssd.zero_grad()
        y1.sum().backward()
        # Test to see if Dirty is being calculated correctly when optimizer modifies
        # ssd_param
        assert ssd_param.storage_state is so.StorageState.ON_CPU_CLEAN
        optimizer_ssd.step()
        assert ssd_param.storage_state is so.StorageState.ON_CPU_DIRTY

        y2 = param + 1
        optimizer_orig.zero_grad()
        y2.sum().backward()
        optimizer_orig.step()

        assert torch.equal(ssd_param.to_tensor(), param)


def test_ssd_flat_parameter_basic():
    _init()
    with tempfile.NamedTemporaryFile() as f:
        refa_param = torch.nn.Parameter(torch.rand((32, 4), dtype=torch.float32))
        refb_param = torch.nn.Parameter(torch.rand((32, 4), dtype=torch.float32))
        refc_param = torch.nn.Parameter(torch.rand((128), dtype=torch.float32))
        ssd_flat_param = so.SsdFlatParameter.from_tensors([refa_param, refb_param, refc_param], direct_to_file=False)
        ssd_flat_param.set_file_params(f.name, 0)

        param_views = list(ssd_flat_param.get_param_views())

        assert refa_param.shape == param_views[0].shape
        assert refb_param.shape == param_views[1].shape
        assert refc_param.shape == param_views[2].shape

        assert torch.equal(refa_param, param_views[0])
        assert torch.equal(refb_param, param_views[1])
        assert torch.equal(refc_param, param_views[2])
        ssd_flat_param.to_file()

        assert not ssd_flat_param.is_available()
        first_value = param_views[0][0][0].item()
        assert ssd_flat_param.is_available()
        assert first_value == refa_param[0][0].item()


def test_ssd_flat_parameter_view_modify():
    _init()
    with tempfile.NamedTemporaryFile() as f:
        refa_param = torch.nn.Parameter(torch.rand((32, 4), dtype=torch.float32), requires_grad=False)
        refb_param = torch.nn.Parameter(torch.rand((32, 4), dtype=torch.float32), requires_grad=False)
        refc_param = torch.nn.Parameter(torch.rand((128), dtype=torch.float32), requires_grad=False)
        ssd_flat_param = so.SsdFlatParameter.from_tensors([refa_param, refb_param, refc_param], direct_to_file=False)
        ssd_flat_param.set_file_params(f.name, 0)
        ssd_flat_param.flush_on_dirty = False

        param_views = list(ssd_flat_param.get_param_views())

        assert ssd_flat_param.storage_state == so.StorageState.ON_CPU_DIRTY
        ssd_flat_param.to_file()
        assert ssd_flat_param.storage_state == so.StorageState.ON_DISK
        assert param_views[0].tensor is None

        param_views[0] += 0.1
        assert ssd_flat_param.storage_state == so.StorageState.ON_CPU_DIRTY


def test_ssd_flat_parameter_view_bwd():
    _init()

    hooks_called = []

    def post_backward_hook(name, hooks_called, *grads):
        print(f"BACKWARD HOOK for tensor {name} CALLED")
        hooks_called.append(name)

    with tempfile.NamedTemporaryFile() as f:
        refa_param = (
            torch.nn.Parameter(torch.rand((32, 4), dtype=torch.float32), requires_grad=True)
            .to("cpu")
            .detach()
            .requires_grad_()
        )
        refb_param = (
            torch.nn.Parameter(torch.rand((32, 4), dtype=torch.float32), requires_grad=True)
            .to("cpu")
            .detach()
            .requires_grad_()
        )
        refc_param = (
            torch.nn.Parameter(torch.rand((128), dtype=torch.float32), requires_grad=True)
            .to("cpu")
            .detach()
            .requires_grad_()
        )
        ssd_flat_param = so.SsdFlatParameter.from_tensors(
            [refa_param, refb_param, refc_param], direct_to_file=True, filename=f.name, offset=0
        )
        orig_copy = ssd_flat_param.data
        cuda_copy = ssd_flat_param.to("cuda").detach().requires_grad_()
        cpu_copy = ssd_flat_param.to("cpu").detach().requires_grad_()

        p_tmp = ssd_flat_param.expand_as(ssd_flat_param)  # Get a grad_fn on p_tmp.
        assert p_tmp.grad_fn is not None
        grad_acc = p_tmp.grad_fn.next_functions[0][0]  # Gets its GradAccumulation object.
        grad_acc.register_hook(functools.partial(post_backward_hook, "GradAccumulation_orig", hooks_called))

        ssd_flat_param.data = cuda_copy
        one = torch.ones((1), requires_grad=True, device=ssd_flat_param.device)
        y1 = ssd_flat_param.views[0] + one
        y2 = cuda_copy + 1

        # ssd_flat_param.to_file()
        # ssd_flat_param.data = orig_copy

        p_tmp = ssd_flat_param.expand_as(ssd_flat_param)  # Get a grad_fn on p_tmp.
        assert p_tmp.grad_fn is not None
        grad_acc = p_tmp.grad_fn.next_functions[0][0]  # Gets its GradAccumulation object.
        grad_acc.register_hook(functools.partial(post_backward_hook, "GradAccumulation_cuda", hooks_called))
        ssd_flat_param.views[0].register_hook(
            functools.partial(post_backward_hook, "ssd_flat_param.views[0]", hooks_called)
        )
        ssd_flat_param.register_hook(functools.partial(post_backward_hook, "ssd_flat_param", hooks_called))
        one.register_hook(functools.partial(post_backward_hook, "one", hooks_called))

        y1.sum().backward()
        y2.sum().backward()

        assert "GradAccumulation_cuda" in hooks_called
        assert "ssd_flat_param.views[0]" in hooks_called
        assert "ssd_flat_param" in hooks_called
        assert "one" in hooks_called

def test_ssd_flat_parameter_view_bwd_propertization():
    _init()

    hooks_called = []

    def post_backward_hook(name, hooks_called, *grads):
        print(f"BACKWARD HOOK for tensor {name} CALLED")
        hooks_called.append(name)

    with tempfile.NamedTemporaryFile() as f:
        layer1 = torch.nn.Linear(32, 4, bias=False)
        layer2 = torch.nn.Linear(32, 4, bias=False)
        layer3 = torch.nn.Linear(128, 1, bias=False)
        ssd_flat_param = so.SsdFlatParameter.from_tensors(
            [layer1.weight, layer2.weight, layer3.weight], direct_to_file=False, filename=f.name, offset=0
        )
        so.PropertizeModule.register_property(layer1, "weight", so.SsdFlatParameterViewProperty(ssd_flat_param, 0))
        so.PropertizeModule.register_property(layer2, "weight", so.SsdFlatParameterViewProperty(ssd_flat_param, 1))
        so.PropertizeModule.register_property(layer3, "weight", so.SsdFlatParameterViewProperty(ssd_flat_param, 2))

        orig_copy = ssd_flat_param.data
        cuda_copy = ssd_flat_param.to("cuda").detach().requires_grad_()
        cpu_copy = ssd_flat_param.to("cpu").detach().requires_grad_()

        p_tmp = ssd_flat_param.expand_as(ssd_flat_param)  # Get a grad_fn on p_tmp.
        assert p_tmp.grad_fn is not None
        grad_acc = p_tmp.grad_fn.next_functions[0][0]  # Gets its GradAccumulation object.
        grad_acc.register_hook(functools.partial(post_backward_hook, "GradAccumulation_orig", hooks_called))

        ssd_flat_param.to_file(release_tensor_after_write=False)
        ssd_flat_param.data = cuda_copy
        one = torch.ones(layer1.weight.shape, requires_grad=True, device=ssd_flat_param.device)
        y1 = layer1.forward(one)
        y2 = cuda_copy + 1

        # ssd_flat_param.to_file()
        # ssd_flat_param.data = orig_copy

        p_tmp = ssd_flat_param.expand_as(ssd_flat_param)  # Get a grad_fn on p_tmp.
        assert p_tmp.grad_fn is not None
        grad_acc = p_tmp.grad_fn.next_functions[0][0]  # Gets its GradAccumulation object.
        grad_acc.register_hook(functools.partial(post_backward_hook, "GradAccumulation_cuda", hooks_called))
        ssd_flat_param.views[0].register_hook(
            functools.partial(post_backward_hook, "ssd_flat_param.views[0]", hooks_called)
        )
        ssd_flat_param.register_hook(functools.partial(post_backward_hook, "ssd_flat_param", hooks_called))
        one.register_hook(functools.partial(post_backward_hook, "one", hooks_called))

        y1.sum().backward()
        y2.sum().backward()

        assert "GradAccumulation_orig" not in hooks_called
        assert "GradAccumulation_cuda" in hooks_called
        assert "ssd_flat_param.views[0]" in hooks_called
        assert "ssd_flat_param" in hooks_called
        assert "one" in hooks_called

def test_ssd_flat_parameter_direct_to_file():
    _init()
    with tempfile.NamedTemporaryFile() as f:
        refa_param = torch.nn.Parameter(torch.rand((32, 4), dtype=torch.float32))
        refb_param = torch.nn.Parameter(torch.rand((32, 4), dtype=torch.float32))
        refc_param = torch.nn.Parameter(torch.rand((128), dtype=torch.float32))
        ssd_flat_param = so.SsdFlatParameter.from_tensors(
            [refa_param, refb_param, refc_param], direct_to_file=True, filename=f.name, offset=0
        )

        param_views = list(ssd_flat_param.get_param_views())

        assert refa_param.shape == param_views[0].shape
        assert refb_param.shape == param_views[1].shape
        assert refc_param.shape == param_views[2].shape

        assert torch.equal(refa_param, param_views[0])
        assert torch.equal(refb_param, param_views[1])
        assert torch.equal(refc_param, param_views[2])
        ssd_flat_param.to_file()

        assert not ssd_flat_param.is_available()
        first_value = param_views[0][0][0].item()
        assert ssd_flat_param.is_available()
        assert first_value == refa_param[0][0].item()
