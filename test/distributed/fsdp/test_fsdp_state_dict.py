# Owner(s): ["oncall: distributed"]

from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
)
from torch.testing._internal.common_distributed import (
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    _zero_model,
    _get_state_dict,
    _get_full_param,
)
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)


class TestFSDPStateDict(FSDPTest):
    def _get_simple_nested_model(self, *fsdp_args, **fsdp_kwargs):
        model = FSDP(
            nn.Sequential(
                FSDP(nn.Linear(10, 10, bias=False), *fsdp_args, **fsdp_kwargs),
                nn.Linear(10, 10, bias=False),
            ),
            *fsdp_args,
            **fsdp_kwargs,
        )
        return model

    def _get_simple_model(self, *fsdp_args, **fsdp_kwargs):
        model = FSDP(nn.Linear(10, 10, bias=False), *fsdp_args, **fsdp_kwargs)
        return model

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=True), CPUOffload(offload_params=False)],
    )
    @parametrize("fp16", [True, False])
    def test_basic_save_and_load_state_dict(self, cpu_offload, fp16):
        for model_call in [
            partial(self._get_simple_nested_model, cpu_offload=cpu_offload),
            partial(self._get_simple_model, cpu_offload=cpu_offload),
        ]:
            model = model_call()
            fsdp_state_dict = _get_state_dict(model, cpu_offload.offload_params, fp16)
            if fp16:
                for tensor in fsdp_state_dict.values():
                    self.assertEqual(tensor.dtype, torch.float16)

            model_new = model_call()
            if not cpu_offload.offload_params:
                model_new = model_new.cuda()
            if fp16:
                model_new.half()

            _zero_model(model_new)

            with model._summon_full_params(), model_new._summon_full_params():
                params = list(model.parameters())
                params_new = list(model_new.parameters())
                self.assertNotEqual(params, params_new)

            model_new.load_state_dict(fsdp_state_dict)
            with model_new._summon_full_params():
                with model._summon_full_params():
                    params = list(model.parameters())
                    params_new = list(model_new.parameters())
                    self.assertEqual(params, params_new)
                    if fp16:
                        for tensor in model_new.parameters():
                            self.assertEqual(tensor.dtype, torch.float16)

    def test_save_and_load_after_forward(self):
        """
        Test that saving after some training results in params being updated as
        expected.
        """
        torch.cuda.set_device(self.rank)
        model = self._get_wrapped_model(group=torch.distributed.distributed_c10d._get_default_group())
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        initial_params = _get_full_param(model)
        for _ in range(6):
            inp = model.module.get_input(torch.device("cuda"))
            output = model(*inp)
            loss = model.module.get_loss(inp, output).cuda()
            model.module.run_backward(loss)
            optim.step()

        new_params = _get_full_param(model)
        self.assertNotEqual(initial_params, new_params)
        state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        _zero_model(model)
        zerod_params = _get_full_param(model)
        for param in zerod_params:
            self.assertEqual(0, param.sum().item())

        model.load_state_dict(state_dict)
        loaded_params = _get_full_param(model)
        self.assertEqual(loaded_params, new_params)


instantiate_parametrized_tests(TestFSDPStateDict)

if __name__ == "__main__":
    run_tests()
