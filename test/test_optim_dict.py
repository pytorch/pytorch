# Owner(s): ["module: optimizer"]

import torch
from copy import deepcopy
from torch.nn import Parameter, Linear
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
)
from torch.testing._internal.common_optimizers import (
    _get_optim_inputs_including_global_cliquey_kwargs,
    optim_db,
    optims,
)
from torch.testing._internal.common_utils import (
    markDynamoStrictTest,
    run_tests,
    parametrize,
    TestCase,
)
from torch.optim.optimizer_dict import OptimizerDict, MakeMuonWithFallback
from itertools import product
from torch.optim import (
    SGD, 
    Adam, 
    Muon, 
    Adafactor,
    RMSprop
)


optimizers = [SGD, 
    Adam, 
    Muon, 
    Adafactor,
    RMSprop]
optim_pairs = [(o1, o2) for o1, o2 in product(optimizers, repeat=2)]                                                    

@markDynamoStrictTest
class TestOptimizerDict(TestCase):
    """
    This test validates basic OptimizerDict behavior by checking the following:
    - Update Algorithm
        * One step of the OptimizerDict should be equal to stepping all sub-optimizers
        * Zero_grad should zero out all sub-optimizers
        * Defining and using a scheduler on optimizer_dict should be equal to defining
          and using the scheduler on all sub-optimizers
        * Currently does not test closures because we aren't exactly sure how to handle them
    - state_dict APIs
        * OptimizerDict should be serializable
    Additionally, it also tests the MakeMuonWithFallback function. 
    """
    @parametrize("optim_list", optim_pairs)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_steps(self, device, dtype, optim_list):
        optim_dict_optimizers = {}
        ref_impl_optimizers = {}
        for i, optimizer_type in enumerate(optim_list):
            weight = torch.randn((10, 5), device=device, dtype=dtype)
            bias = torch.randn((10), device=device, dtype=dtype)

            
            weight_grad = torch.randn((10, 5), device=device, dtype=dtype)
            bias_grad = torch.randn((10), device=device, dtype=dtype)

            if optimizer_type.__name__ != "Muon":
                ref_weight_tensor = Parameter(weight.detach().clone())
                dict_weight_tensor = Parameter(weight.detach().clone())
                ref_bias_tensor = Parameter(bias.detach().clone())
                dict_bias_tensor = Parameter(bias.detach().clone())


                ref_impl_optimizers[f"optimizer{i}"] = optimizer_type([ref_weight_tensor, ref_bias_tensor])
                optim_dict_optimizers[f"optimizer{i}"] = optimizer_type([dict_weight_tensor, dict_bias_tensor])
            else:
                ref_weight_tensor = Parameter(weight.detach().clone())
                dict_weight_tensor = Parameter(weight.detach().clone())

                ref_impl_optimizers[f"optimizer{i}"] = optimizer_type([ref_weight_tensor, ])
                optim_dict_optimizers[f"optimizer{i}"] = optimizer_type([dict_weight_tensor, ])
                
        optim_dict = OptimizerDict(optim_dict_optimizers)
        num_test_steps = 10
        for _ in range(num_test_steps):
            for key in ref_impl_optimizers.keys():
                ref_optimizer = ref_impl_optimizers[key]
                dict_optimizer = optim_dict[key]
                for ref_param_group, test_param_group in zip(ref_optimizer.param_groups, dict_optimizer.param_groups):
                    for ref_param, test_param in zip(ref_param_group["params"], test_param_group["params"]):
                        grad_tensor = torch.randn(ref_param.shape, device = ref_param.device, dtype = ref_param.dtype)
                        ref_param.grad = grad_tensor.clone().detach()
                        test_param.grad = grad_tensor.clone().detach()
            optim_dict.step()
            optim_dict.zero_grad()
            for optimizer in ref_impl_optimizers.values():
                optimizer.step()
                optimizer.zero_grad()

        for key in ref_impl_optimizers.keys():
            ref_optimizer = ref_impl_optimizers[key]
            dict_optimizer = optim_dict[key]
            for ref_param_group, test_param_group in zip(ref_optimizer.param_groups, dict_optimizer.param_groups):
                for ref_param, test_param in zip(ref_param_group["params"], test_param_group["params"]):
                    self.assertEqual(ref_param, test_param)
                    self.assertEqual(ref_param.grad, test_param.grad)

                    
    @parametrize("optim", optimizers)
    def test_state_dict(self, device, optim):
        def train_step(model, model_input, optimizer):
            loss = model(model_input).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model = Linear(5, 5, bias = False).to(device)
        optimizer = optim(model.parameters())
        optimizer_dict = OptimizerDict({"optim1": optimizer})
        train_step(model, torch.randn(5, 5, device=device), optimizer_dict)
        model_state_dict = deepcopy(model.state_dict())
        state_dict = deepcopy(optimizer_dict.state_dict())
        train_step(model, torch.ones(5, 5, device=device), optimizer_dict)

        new_model = Linear(5, 5, bias = False).to(device)
        new_model.load_state_dict(model_state_dict)
        new_optimizer = optim(new_model.parameters())
        new_optimizer_dict = OptimizerDict({"optim1" : new_optimizer})
        new_optimizer_dict.load_state_dict(state_dict)
        train_step(new_model, torch.ones(5, 5, device=device), new_optimizer_dict)

        for ref_param, test_param in zip(model.parameters(), new_model.parameters()):
            self.assertEqual(ref_param, test_param)

    @parametrize("fallback_optim", [Adam, SGD, RMSprop, Adafactor])
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_make_muon_with_fallback(self, device, dtype, fallback_optim):
        param_2d_1 = Parameter(torch.randn(10, 5, device=device, dtype=dtype))
        param_2d_2 = Parameter(torch.randn(8, 4, device=device, dtype=dtype))
        param_1d = Parameter(torch.randn(10, device=device, dtype=dtype))
        param_3d = Parameter(torch.randn(2, 3, 4, device=device, dtype=dtype))

        all_params = [param_2d_1, param_2d_2, param_1d, param_3d]

        ref_param_2d_1 = Parameter(param_2d_1.detach().clone())
        ref_param_2d_2 = Parameter(param_2d_2.detach().clone())
        ref_param_1d = Parameter(param_1d.detach().clone())
        ref_param_3d = Parameter(param_3d.detach().clone())

        ref_2d_params = [ref_param_2d_1, ref_param_2d_2]
        ref_non2d_params = [ref_param_1d, ref_param_3d]

        muon_kwargs = {"lr": 0.01}
        fallback_kwargs = {"lr": 0.01}

        opt_dict = MakeMuonWithFallback(
            all_params, fallback_optim, muon_kwargs, fallback_kwargs
        )

        ref_muon = Muon(ref_2d_params, **muon_kwargs)
        ref_fallback = fallback_optim(ref_non2d_params, **fallback_kwargs)

        num_steps = 10
        for _ in range(num_steps):
            for param, ref_param in [
                (param_2d_1, ref_param_2d_1),
                (param_2d_2, ref_param_2d_2),
                (param_1d, ref_param_1d),
                (param_3d, ref_param_3d),
            ]:
                grad = torch.randn_like(param)
                param.grad = grad.clone()
                ref_param.grad = grad.clone()

            opt_dict.step()
            opt_dict.zero_grad()

            ref_muon.step()
            ref_muon.zero_grad()
            ref_fallback.step()
            ref_fallback.zero_grad()

        self.assertEqual(param_2d_1, ref_param_2d_1)
        self.assertEqual(param_2d_2, ref_param_2d_2)

        self.assertEqual(param_1d, ref_param_1d)
        self.assertEqual(param_3d, ref_param_3d)


instantiate_device_type_tests(TestOptimizerDict, globals(), allow_mps=True)

if __name__ == "__main__":
    run_tests()
