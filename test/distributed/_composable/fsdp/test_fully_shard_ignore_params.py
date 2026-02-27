# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._composable.fsdp.fully_shard import FSDPModule as FSDP2
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import implicit_replication
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)


device_type = torch.device(get_devtype())

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class C(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.lin_c = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin_c(x)


class B(nn.Module):
    def __init__(self, dim: int, subtrahend: torch.Tensor) -> None:
        super().__init__()

        self.lin_b = nn.Linear(dim, dim)
        self.module_c = C(dim)
        self.subtrahend = nn.Parameter(subtrahend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c_result = self.module_c(x)
        return self.lin_b(c_result) - self.subtrahend


class A(nn.Module):
    def __init__(
        self, dim: int, addend: torch.Tensor, subtrahend: torch.Tensor
    ) -> None:
        super().__init__()

        self.module_b = B(dim, subtrahend)
        self.addend = nn.Parameter(addend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.module_b(x) + self.addend
        return result.sum()


class Y(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        p = torch.randn(10, device=device_type)
        self.p = nn.Parameter(p)


class X(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        q = torch.randn(10, device=device_type)
        self.q = nn.Parameter(q)
        self.y = Y()


def _append_prefix(prefix: str, name: str) -> str:
    if prefix != "" and name != "":
        return prefix + "." + name
    else:
        return prefix + name


def _generate_model_and_input() -> nn.Module:
    dim = 8

    torch.manual_seed(42)
    addend = torch.randn((dim, dim), device=device_type)

    torch.manual_seed(70)
    subend = torch.randn((dim, dim), device=device_type)

    model = A(dim, addend, subend).to(device_type)

    torch.manual_seed(84)
    inp = torch.randn((dim, dim), device=device_type)

    return model, inp


def _find_name_param_mappings(module: torch.nn.Module, prefix: str):
    name_to_param_map = {}
    param_to_name_map = {}
    for name, param in module.named_parameters(prefix):
        name_to_param_map[name] = param
        param_to_name_map[param] = name
    return name_to_param_map, param_to_name_map


def _discover_ddp_ignored_params(module: torch.nn.Module, prefix: str):
    ddp_ignore_parameters: list[str] = []
    if isinstance(module, FSDP2):
        ddp_ignore_parameters = [name for name, _ in module.named_parameters(prefix)]
    else:
        for name, child in list(module.named_children()):
            # post order traversal
            path = _append_prefix(prefix, name)
            ignored_params = _discover_ddp_ignored_params(child, path)
            ddp_ignore_parameters.extend(ignored_params)

    return ddp_ignore_parameters


def _modify_ddp_ignored_params(
    ddp_ignored_param_names: list[str],
    fsdp_ignored_params: set[torch.nn.Parameter],
    name_to_param_map: dict,
):
    modified_list = []
    for name in ddp_ignored_param_names:
        if name not in name_to_param_map:
            raise AssertionError(f"Expected {name} in name_to_param_map")
        param = name_to_param_map[name]
        if param not in fsdp_ignored_params:
            # DDP can ignore only if it is not ignored by FSDP
            modified_list.append(name)
    return modified_list


def _get_full_tensor(name, param):
    if isinstance(param, DTensor):
        return param.full_tensor()
    else:
        return param


def _discover_fsdp_ignored_params(
    module: torch.nn.Module, ignored_path, path: str
) -> set[torch.nn.Parameter]:
    total_ignored_params = set()

    if ignored_path == path:
        # Ignore all parameters inside module
        name_parameters = dict(module.named_parameters(path))
        total_ignored_params = set(name_parameters.values())

        for _ in module.buffers(recurse=True):
            # yet to handle ignoring buffers
            raise AssertionError("Yet to handle ignoring buffers")
    else:
        for name, sub_module in list(module.named_children()):
            child_path = _append_prefix(path, name)
            child_ignored_params = _discover_fsdp_ignored_params(
                sub_module, ignored_path, child_path
            )
            total_ignored_params = total_ignored_params | child_ignored_params

    return total_ignored_params


def _post_order_wrap_fsdp(
    module: torch.nn.Module,
    mesh,
    path: str,
    ignored_path: str,
    ignored_params: set[torch.nn.Parameter],
) -> torch.nn.Module:
    if ignored_path != path:
        for name, sub_module in list(module.named_children()):
            child_path = _append_prefix(path, name)
            _post_order_wrap_fsdp(
                sub_module, mesh, child_path, ignored_path, ignored_params
            )

        fully_shard(module, mesh=mesh, ignored_params=ignored_params)

    return module


def _find_all_fsdped_modules(module: torch.nn.Module, path) -> set[str]:
    result = set()
    for name, child in list(module.named_children()):
        child_path = _append_prefix(path, name)
        child_result = _find_all_fsdped_modules(child, child_path)
        result = result | child_result
    if isinstance(module, FSDP2):
        result.add(path)
    return result


class TestFullyShardIgnoreParams(FSDPTest):
    """Tests for fully_shard ignore params"""

    def compare_params(self, name, ref_param, test_param):
        ref_full_tensor = _get_full_tensor(name, ref_param)
        test_full_tensor = _get_full_tensor(name, test_param)
        self.assertTrue(torch.allclose(ref_full_tensor, test_full_tensor))

    def compare_ref_test_params(self, ref_name_to_param_map, test_name_to_param_map):
        for name in ref_name_to_param_map:
            self.assertTrue(name in test_name_to_param_map)

        for name in test_name_to_param_map:
            self.assertTrue(name in ref_name_to_param_map)

        for name, ref_param in ref_name_to_param_map.items():
            test_param = test_name_to_param_map[name]
            self.compare_params(name, ref_param, test_param)

    @skip_if_lt_x_gpu(2)
    def test_ddp_A_fsdp_B_ddp_C(self):
        default_pg = dist.distributed_c10d._get_default_group()
        mesh = init_device_mesh(device_type.type, mesh_shape=(default_pg.size(),))

        ref_model, ref_inp = _generate_model_and_input()

        ref_model = DDP(ref_model, process_group=default_pg)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        ref_name_to_param_map, _ = _find_name_param_mappings(ref_model, "")

        test_model, test_inp = _generate_model_and_input()

        # Computes the mappings before applying FSDP and DDP
        test_name_to_param_map, _ = _find_name_param_mappings(test_model, "")

        ignored_path = "module_b.module_c"

        fsdp_ignored_params = _discover_fsdp_ignored_params(
            test_model, ignored_path=ignored_path, path=""
        )
        test_model.module_b = _post_order_wrap_fsdp(
            test_model.module_b,
            mesh=mesh,
            path="module_b",
            ignored_path=ignored_path,
            ignored_params=fsdp_ignored_params,
        )

        fsdped_modules = _find_all_fsdped_modules(test_model, "")
        self.assertEqual(fsdped_modules, {"module_b", "module_b.lin_b"})

        ddp_ignored_param_names = _discover_ddp_ignored_params(test_model, "")
        self.assertEqual(
            set(ddp_ignored_param_names),
            {
                "module_b.subtrahend",
                "module_b.lin_b.weight",
                "module_b.lin_b.bias",
                "module_b.module_c.lin_c.weight",
                "module_b.module_c.lin_c.bias",
            },
        )

        modified_ddp_ignored_param_names = _modify_ddp_ignored_params(
            ddp_ignored_param_names, fsdp_ignored_params, test_name_to_param_map
        )
        self.assertEqual(
            set(modified_ddp_ignored_param_names),
            {"module_b.subtrahend", "module_b.lin_b.weight", "module_b.lin_b.bias"},
        )

        DDP._set_params_and_buffers_to_ignore_for_model(
            module=test_model,
            params_and_buffers_to_ignore=modified_ddp_ignored_param_names,
        )
        test_model = DDP(test_model, broadcast_buffers=False)
        test_optim = torch.optim.Adam(test_model.parameters(), lr=1e-2)

        # Recomputes the mappings after applying FSDP and DDP
        test_name_to_param_map, _ = _find_name_param_mappings(test_model, "")

        # Compare ref and test parameters before iterations
        self.compare_ref_test_params(ref_name_to_param_map, test_name_to_param_map)

        for _ in range(3):
            ref_loss = ref_model(ref_inp)
            test_loss = test_model(test_inp)

            # Compare ref and test loss at each step
            self.assertTrue(torch.allclose(ref_loss, test_loss))
            ref_loss.backward()
            test_loss.backward()

            with implicit_replication():
                ref_optim.step()
                ref_optim.zero_grad()
                test_optim.step()
                test_optim.zero_grad()

                # Compare ref and test parameters at each step
                self.compare_ref_test_params(
                    ref_name_to_param_map, test_name_to_param_map
                )


instantiate_parametrized_tests(TestFullyShardIgnoreParams)

if __name__ == "__main__":
    run_tests()
