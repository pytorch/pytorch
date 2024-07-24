# Owner(s): ["oncall: distributed"]
import io
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor import DTensor as DT, init_device_mesh, Replicate, Shard
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import (
    _get_module_fsdp_state,
    clean_tensor_name,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.distributed.tensor.parallel.fsdp import DTensorExtensions
from torch.distributed.tensor.parallel.input_reshard import input_reshard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


# Tensor-Parallel degree
TP_DEGREE = 2
LR = 3e-5


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(5, 8)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(8, 4)
        self.net3 = nn.Linear(4, 12)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        return x

    def get_input(self):
        return torch.rand(4, 5, device="cuda")


class SimpleModelUneven(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Linear(5, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 15)
        self.net3 = nn.Linear(15, 30)
        self.net4 = nn.Linear(30, 5)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        x = self.net4(x)
        return x

    def get_input(self):
        return torch.rand(4, 5, device="cuda")


# TODO: add additional tests for multi_param_group, optim_in_backward,
# and fsdp_nested.
class TestNew2dParallelTraining(DTensorTestBase):
    def _compare_params(self, m1, m2):
        with FSDP.summon_full_params(m1):
            with FSDP.summon_full_params(m2):
                for n_p1, n_p2 in zip(m1.named_parameters(), m2.named_parameters()):
                    p1 = n_p1[1]
                    p2 = n_p2[1]
                    if n_p1[0] != n_p2[0]:
                        self.assertTrue(n_p1[0] in n_p2[0])
                    name = n_p1[0]
                    if name == "net2.bias" and self.rank != 0:
                        continue
                    if type(p2) is DT:
                        p2 = p2.redistribute(p2.device_mesh, [Replicate()]).to_local()
                    self.assertTrue(torch.allclose(p1, p2), f"{p1} vs {p2}")

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_raise_invalid_tp_composition(self):
        with self.assertRaisesRegex(
            RuntimeError, r"Found TP device_mesh on the \d dimension of its parent mesh"
        ):
            mesh_2d = init_device_mesh(
                self.device_type, (2, self.world_size // 2), mesh_dim_names=("tp", "dp")
            )
            parallelize_plan = {
                "net1": ColwiseParallel(),
                "net2": RowwiseParallel(),
            }
            model_2d = parallelize_module(
                SimpleModel().cuda(), mesh_2d["tp"], parallelize_plan
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_fsdp_state_enable_extension(self):
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        model = FSDP(
            SimpleModel().cuda(),
            device_mesh=mesh_2d["dp"],
        )
        fsdp_state = _get_module_fsdp_state(model)
        self.assertTrue(isinstance(fsdp_state._fsdp_extension, DTensorExtensions))

    def _test_2d_e2e_training(
        self,
        use_orig_params=False,
        recompute_activation=False,
    ) -> None:
        torch.manual_seed(0)
        model = SimpleModel().cuda(self.rank)
        model = FSDP(model, use_orig_params=use_orig_params)
        optim = torch.optim.Adam(model.parameters(), lr=0.01)

        torch.manual_seed(0)
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        tp_mesh = mesh_2d["tp"]
        dp_mesh = mesh_2d["dp"]
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        model_2d = parallelize_module(SimpleModel().cuda(), tp_mesh, parallelize_plan)
        model_2d = FSDP(
            model_2d,
            device_mesh=dp_mesh,
            use_orig_params=use_orig_params,
        )
        optim_2d = torch.optim.Adam(model_2d.parameters(), lr=0.01)

        if recompute_activation:
            model_2d = input_reshard(model_2d, mesh_2d["tp"], 0)

        # Check named parameters are returning the same name at least.
        param_names_2d = [
            clean_tensor_name(name) for name, _ in model_2d.named_parameters()
        ]
        for name, _ in model.named_parameters():
            name = clean_tensor_name(name)
            if name not in param_names_2d:
                print(name, param_names_2d)
            self.assertTrue(name in param_names_2d)
        self._compare_params(model, model_2d)

        # TODO: add additional tests for multi_param_group and optim_in_backward.

        for i in range(5):
            # Ensure all input across TP ranks are same.
            # TODO: add a get_group_rank() to DeviceMesh.
            torch.manual_seed(i + dist.get_rank(dp_mesh.get_group(mesh_dim=0)))
            input = torch.rand(4, 5).cuda(self.rank)
            output = model(input)
            output_2d = model_2d(input)
            self.assertEqual(output, output_2d)
            output.sum().backward()
            output_2d.sum().backward()
            optim.step()
            optim_2d.step()
            self.assertEqual(model(input), model_2d(input))

        # Ensure all params are still the same after optimizer update.
        self._compare_params(model, model_2d)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_e2e_training_default(self):
        self._test_2d_e2e_training()

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_e2e_training_use_orig_params(self):
        self._test_2d_e2e_training(use_orig_params=True)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_e2e_training_not_use_orig_params(self):
        # TODO: need to revisit input_reshard API about why it failed multi-gpu tests.
        # self._test_2d_e2e_training(recompute_activation=True)
        self._test_2d_e2e_training(recompute_activation=False)


# TODO: update all state dict unit tests to use distributed.checkpoint.state_dict,
# and consolidate all the state_dict test in test.distributed.checkpoint.
class TestNew2dParallelStateDict(DTensorTestBase):
    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_fsdp_2d_extension(self):
        """
        Test whether _fsdp_extension from FSDPstate has been set correctly.
        """
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
            "net3": ColwiseParallel(),
        }
        model_2d = parallelize_module(
            SimpleModel().cuda(),
            mesh_2d["tp"],
            parallelize_plan=parallelize_plan,
        )
        model_2d = FSDP(model_2d, device_mesh=mesh_2d["dp"], use_orig_params=True)
        model_2d_fsdp_state = _get_module_fsdp_state(model_2d)
        self.assertTrue(
            isinstance(model_2d_fsdp_state._fsdp_extension, DTensorExtensions)
        )

        mesh_1d = init_device_mesh("cuda", (self.world_size,))
        model_1d = FSDP(SimpleModel().cuda(), device_mesh=mesh_1d, use_orig_params=True)
        model_1d_fsdp_state = _get_module_fsdp_state(model_1d)
        self.assertEqual(model_1d_fsdp_state._fsdp_extension, None)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("is_even_sharded_model", [True, False])
    def test_2d_state_dict(self, is_even_sharded_model):
        simple_model = SimpleModel if is_even_sharded_model else SimpleModelUneven

        # Create a model without wrapper
        torch.manual_seed(0)
        no_wrap_model = simple_model().cuda(self.rank)
        no_wrap_state_dict = no_wrap_model.state_dict()

        # Create a model and sharded it with 2D FSDP + TP
        torch.manual_seed(0)
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        tp_mesh = mesh_2d["tp"]
        dp_mesh = mesh_2d["dp"]
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        model_2d = parallelize_module(simple_model().cuda(), tp_mesh, parallelize_plan)
        model_2d = FSDP(model_2d, device_mesh=dp_mesh, use_orig_params=True)

        FSDP.set_state_dict_type(
            model_2d,
            StateDictType.SHARDED_STATE_DICT,
        )
        state_dict_2d = model_2d.state_dict()

        for no_wrap_items, two_d_items in zip(
            no_wrap_state_dict.items(), state_dict_2d.items()
        ):
            no_wrap_k, no_wrap_v = no_wrap_items
            two_d_k, two_d_v = two_d_items

            self.assertEqual(no_wrap_k, two_d_k)

            # check if all value in 2D state_dict are DTensor
            self.assertTrue(isinstance(two_d_v, DT))
            self.assertEqual(len(two_d_v.placements), 2)
            # the outer dimension is the FSDP dimension and the placement is always Shard(0)
            self.assertEqual(two_d_v.placements[0], Shard(0))
            self.assertEqual(two_d_v.device_mesh, mesh_2d)

            # check if the parameter value is the same between 2D model and the model without wrapper
            all_gather_two_d_v = two_d_v.redistribute(
                mesh_2d, (Replicate(), Replicate())
            )
            self.assertEqual(
                torch.allclose(no_wrap_v, all_gather_two_d_v.to_local()), True
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("is_even_sharded_model", [True, False])
    def test_2d_load_state_dict(self, is_even_sharded_model):
        simple_model = SimpleModel if is_even_sharded_model else SimpleModelUneven

        torch.manual_seed(0)
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        tp_mesh = mesh_2d["tp"]
        dp_mesh = mesh_2d["dp"]
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        model_2d = parallelize_module(simple_model().cuda(), tp_mesh, parallelize_plan)
        model_2d = FSDP(model_2d, device_mesh=dp_mesh, use_orig_params=True)
        optim_2d = torch.optim.Adam(model_2d.parameters(), lr=0.01)

        FSDP.set_state_dict_type(
            model_2d,
            StateDictType.SHARDED_STATE_DICT,
        )
        checkpoint = io.BytesIO()
        torch.save(model_2d.state_dict(), checkpoint)
        # Deepcopy to save current state_dict to compare with the state_dict loaded back below.
        ref_state_dict = deepcopy(model_2d.state_dict())

        # Update the parameters so model.state_dict() will be different from ref_dtensor_sd.
        model_2d(model_2d.get_input().cuda(self.rank)).sum().backward()
        optim_2d.step()

        # Load ref_state_dict back.
        checkpoint.seek(0)
        load_ref_state_dict = torch.load(checkpoint)
        model_2d.load_state_dict(load_ref_state_dict)
        new_state_dict = model_2d.state_dict()

        # Check whether new_state_dict is the same as ref_state_dict.
        for (k1, v1), (k2, v2) in zip(ref_state_dict.items(), new_state_dict.items()):
            # check whether fqn are the same
            self.assertEqual(k1, k2)

            self.assertEqual(type(v1), DT)
            self.assertEqual(type(v2), DT)
            # check whether DTensor are the same
            # TODO: 2D DTensor comparison is not supported at the time, so we are comparing the spec and the local tensor for now.
            # TODO: Update it to compare the two DTensors once 2D DTensor comparison is supported.
            self.assertEqual(v1.to_local(), v2.to_local())
            self.assertEqual(v1.device_mesh, v2.device_mesh)
            self.assertEqual(v1.placements, v2.placements)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("is_even_sharded_model", [True, False])
    def test_2d_optim_state_dict(self, is_even_sharded_model):
        simple_model = SimpleModel if is_even_sharded_model else SimpleModelUneven

        # Create a model without wrapper
        torch.manual_seed(0)
        no_wrap_model = simple_model().cuda(self.rank)
        no_wrap_state_dict = no_wrap_model.state_dict()
        no_wrap_optim = torch.optim.Adam(no_wrap_model.parameters(), lr=0.01)
        no_wrap_model(no_wrap_model.get_input().cuda(self.rank)).sum().backward()
        no_wrap_optim.step()
        no_wrap_osd = get_optimizer_state_dict(no_wrap_model, optimizers=no_wrap_optim)

        # Create a model and sharded it with 2D FSDP + TP
        torch.manual_seed(0)
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        model_2d = parallelize_module(
            simple_model().cuda(), mesh_2d["tp"], parallelize_plan
        )
        model_2d = FSDP(model_2d, device_mesh=mesh_2d["dp"], use_orig_params=True)
        FSDP.set_state_dict_type(
            model_2d,
            StateDictType.SHARDED_STATE_DICT,
        )
        optim_2d = torch.optim.Adam(model_2d.parameters(), lr=0.01)
        model_2d(model_2d.get_input().cuda(self.rank)).sum().backward()
        optim_2d.step()
        optim_2d_osd = get_optimizer_state_dict(model_2d, optimizers=optim_2d)
        ref_optim_2d_osd = deepcopy(optim_2d_osd)

        no_wrap_osd_states = no_wrap_osd["state"]
        optim_2d_osd_states = optim_2d_osd["state"]

        self.assertEqual(len(no_wrap_osd_states), len(optim_2d_osd_states))
        self.assertEqual(no_wrap_osd_states.keys(), optim_2d_osd_states.keys())
        for fqn, states in no_wrap_osd_states.items():
            dist_states = optim_2d_osd_states.get(fqn)

            for state_name, state in states.items():
                dist_state = dist_states.get(state_name)
                # If a state  is DTensor, we all gather it in both DP and TP dimension to
                # compare with no_wrap state.
                if isinstance(dist_state, DT):
                    dist_state = (
                        dist_state.cuda()
                        .redistribute(placements=(Replicate(), Replicate()))
                        .to_local()
                    )
                self.assertTrue(isinstance(dist_state, torch.Tensor))
                self.assertTrue(torch.allclose(state, dist_state))

        # Update the parameters 2d optim states will be different from ref_optim_state_dict.
        model_2d(model_2d.get_input().cuda(self.rank)).sum().backward()
        optim_2d.step()

        set_optimizer_state_dict(
            model_2d, optimizers=optim_2d, optim_state_dict=ref_optim_2d_osd
        )
        new_optim_2d_osd = get_optimizer_state_dict(model_2d, optimizers=optim_2d)

        ref_optim_2d_osd_states = ref_optim_2d_osd["state"]
        new_optim_2d_osd_states = optim_2d_osd["state"]

        # Compare the new optim state dict after load with the reference one
        self.assertEqual(len(ref_optim_2d_osd_states), len(new_optim_2d_osd_states))
        self.assertEqual(ref_optim_2d_osd_states.keys(), new_optim_2d_osd_states.keys())
        for fqn, states in ref_optim_2d_osd_states.items():
            new_states = new_optim_2d_osd_states.get(fqn)

            for state_name, state in states.items():
                new_state = new_states.get(state_name)

                if isinstance(new_state, DT):
                    self.assertEqual(new_state.placements, state.placements)
                    self.assertEqual(new_state.device_mesh, state.device_mesh)
                    self.assertTrue(
                        torch.allclose(new_state.to_local(), state.to_local())
                    )
                else:
                    self.assertEqual(new_state, state)


instantiate_parametrized_tests(TestNew2dParallelStateDict)
if __name__ == "__main__":
    run_tests()
