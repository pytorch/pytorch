# Owner(s): ["module: optimizer"]

import itertools
import pickle

import torch
from torch.optim.swa_utils import (
    AveragedModel,
    get_ema_avg_fn,
    get_ema_multi_avg_fn,
    get_swa_multi_avg_fn,
    update_bn,
)
from torch.testing._internal.common_device_type import dtypes, onlyAccelerator
from torch.testing._internal.common_dtype import floating_and_complex_types_and
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    TestCase,
)


class SWATestDNN(torch.nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.n_features = 100
        self.fc1 = torch.nn.Linear(input_features, self.n_features)
        self.bn = torch.nn.BatchNorm1d(self.n_features)

    def compute_preactivation(self, x):
        return self.fc1(x)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        return x


class SWATestCNN(torch.nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.n_features = 10
        self.conv1 = torch.nn.Conv2d(
            input_channels, self.n_features, kernel_size=3, padding=1
        )
        self.bn = torch.nn.BatchNorm2d(self.n_features, momentum=0.3)

    def compute_preactivation(self, x):
        return self.conv1(x)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        return x


class TestSWAUtilsGeneric(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize("layout", ["coo", "csr", "nested", "masked"])
    def test_ema_avg_fn_layout_fallback(self, layout):
        ema = torch.tensor([[0.7, 0.5], [0.0, 0.3]])
        current = torch.tensor([[0.2, 0.1], [0.0, 0.4]])
        if layout == "coo":
            ema, current = ema.to_sparse(), current.to_sparse()
        elif layout == "csr":
            ema, current = ema.to_sparse_csr(), current.to_sparse_csr()
        elif layout == "nested":
            ema = torch.nested.nested_tensor([ema[0], ema[1, :1]])
            current = torch.nested.nested_tensor([current[0], current[1, :1]])
        else:
            mask = torch.tensor([[True, True], [False, True]])
            ema = torch.masked.masked_tensor(ema, mask)
            current = torch.masked.masked_tensor(current, mask)
        expected = 0.9999 * ema + (1 - 0.9999) * current
        actual = get_ema_avg_fn(0.9999)(ema, current, 1)
        if layout == "nested":
            actual, expected = actual.to_padded_tensor(0), expected.to_padded_tensor(0)
        elif layout == "masked":
            self.assertEqual(actual.get_mask(), expected.get_mask())
            actual, expected = actual.get_data(), expected.get_data()
        else:
            actual, expected = actual.to_dense(), expected.to_dense()
        self.assertEqual(actual, expected, rtol=0, atol=0)

    def test_averaged_model_state_dict(self):
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3), torch.nn.Linear(5, 10)
        )
        averaged_dnn = AveragedModel(dnn)
        averaged_dnn2 = AveragedModel(dnn)
        n_updates = 10
        for _ in range(n_updates):
            for p in dnn.parameters():
                p.detach().add_(torch.randn_like(p))
            averaged_dnn.update_parameters(dnn)
        averaged_dnn2.load_state_dict(averaged_dnn.state_dict())
        for p_swa, p_swa2 in zip(averaged_dnn.parameters(), averaged_dnn2.parameters()):
            self.assertEqual(p_swa, p_swa2)
        self.assertTrue(averaged_dnn.n_averaged == averaged_dnn2.n_averaged)

    def test_averaged_model_default_avg_fn_picklable(self):
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3),
            torch.nn.BatchNorm2d(5),
            torch.nn.Linear(5, 5),
        )
        averaged_dnn = AveragedModel(dnn)
        pickle.dumps(averaged_dnn)

    @parametrize("use_multi_avg_fn", [True, False])
    @parametrize("use_buffers", [True, False])
    def test_averaged_model_exponential(self, use_multi_avg_fn, use_buffers):
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3),
            torch.nn.BatchNorm2d(5, momentum=0.3),
            torch.nn.Linear(5, 10),
        )
        decay = 0.9

        if use_multi_avg_fn:
            averaged_dnn = AveragedModel(
                dnn, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=use_buffers
            )
        else:

            def avg_fn(p_avg, p, n_avg):
                return decay * p_avg + (1 - decay) * p

            averaged_dnn = AveragedModel(dnn, avg_fn=avg_fn, use_buffers=use_buffers)

        if use_buffers:
            dnn_params = list(itertools.chain(dnn.parameters(), dnn.buffers()))
        else:
            dnn_params = list(dnn.parameters())

        averaged_params = [
            torch.zeros_like(param)
            for param in dnn_params
            if param.size() != torch.Size([])
        ]

        n_updates = 10
        for i in range(n_updates):
            updated_averaged_params = []
            for p, p_avg in zip(dnn_params, averaged_params):
                if p.size() == torch.Size([]):
                    continue
                p.detach().add_(torch.randn_like(p))
                if i == 0:
                    updated_averaged_params.append(p.clone())
                else:
                    updated_averaged_params.append(
                        (p_avg * decay + p * (1 - decay)).clone()
                    )
            averaged_dnn.update_parameters(dnn)
            averaged_params = updated_averaged_params

        if use_buffers:
            for p_avg, p_swa in zip(
                averaged_params,
                itertools.chain(
                    averaged_dnn.module.parameters(), averaged_dnn.module.buffers()
                ),
            ):
                self.assertEqual(p_avg, p_swa)
        else:
            for p_avg, p_swa in zip(averaged_params, averaged_dnn.parameters()):
                self.assertEqual(p_avg, p_swa)
            for b_avg, b_swa in zip(dnn.buffers(), averaged_dnn.module.buffers()):
                self.assertEqual(b_avg, b_swa)

    def test_bn_update_eval_momentum(self):
        objects = 100
        input_channels = 3
        height, width = 5, 5
        x = torch.rand(objects, input_channels, height, width)
        ds_x = torch.utils.data.TensorDataset(x)
        dl_x = torch.utils.data.DataLoader(ds_x, batch_size=5, shuffle=True)
        cnn = SWATestCNN(input_channels=input_channels)
        cnn.eval()
        update_bn(dl_x, cnn)
        self.assertFalse(cnn.training)

        self.assertEqual(cnn.bn.momentum, 0.3)


instantiate_parametrized_tests(TestSWAUtilsGeneric)


class TestSWAUtils(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @dtypes(*floating_and_complex_types_and(torch.float16, torch.bfloat16))
    def test_ema_fixed_point(self, device, dtype):
        current = torch.full((1,), 0.7, device=device, dtype=dtype)
        single, multi = current.clone(), current.clone()
        single_fn = get_ema_avg_fn(0.9999)
        multi_fn = get_ema_multi_avg_fn(0.9999)
        for step in range(2000):
            single = single_fn(single, current, step)
            multi_fn([multi], [current], step)
        self.assertEqual(single, current, rtol=0, atol=0)
        self.assertEqual(single, multi, rtol=0, atol=0)

    @dtypes(*floating_and_complex_types_and(torch.float16, torch.bfloat16))
    @parametrize("decay", [0.0, 0.5, 0.99, 0.999, 0.9999, 1.0])
    def test_ema_avg_fn(self, device, dtype, decay):
        ema = torch.tensor([-0.75, 0.25, 1.5], device=device, dtype=dtype)
        current = torch.tensor([0.5, -0.5, 0.75], device=device, dtype=dtype)
        if dtype.is_complex:
            ema = ema + 0.25j
            current = current - 0.5j
        ema.requires_grad_()
        current.requires_grad_()
        ema_before, current_before = ema.clone(), current.clone()
        reference_dtype = torch.complex128 if dtype.is_complex else torch.float64
        expected = decay * ema.cpu().to(reference_dtype) + (1 - decay) * current.cpu().to(
            reference_dtype
        )
        result = get_ema_avg_fn(decay)(ema, current, 1)
        self.assertEqual(result, expected.to(dtype).to(device))
        self.assertEqual(ema, ema_before, rtol=0, atol=0)
        self.assertEqual(current, current_before, rtol=0, atol=0)
        self.assertNotEqual(result.data_ptr(), ema.data_ptr())
        self.assertNotEqual(result.data_ptr(), current.data_ptr())
        self.assertFalse(result.requires_grad)

    @dtypes(
        (torch.int64, torch.int64),
        (torch.bool, torch.bool),
        (torch.float32, torch.bool),
        (torch.float32, torch.float64),
        (torch.float64, torch.float32),
        (torch.float32, torch.complex64),
    )
    def test_ema_avg_fn_type_promotion(self, device, dtypes):
        ema_dtype, current_dtype = dtypes
        ema = torch.tensor([0, 1], device=device, dtype=ema_dtype)
        current = torch.tensor([1, 0], device=device, dtype=current_dtype)
        expected = 0.75 * ema + 0.25 * current
        result = get_ema_avg_fn(0.75)(ema, current, 1)
        self.assertEqual(result, expected, rtol=0, atol=0)

    @parametrize("noncontiguous", [False, True])
    def test_ema_avg_fn_broadcast(self, device, noncontiguous):
        ema = torch.full((2, 3), 0.7, device=device)
        current = torch.full((1, 3), 0.7, device=device)
        if noncontiguous:
            ema, current = ema.t(), current.t()
        result = get_ema_avg_fn(0.9999)(ema, current, 1)
        self.assertEqual(result, ema, rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, "size of tensor"):
            get_ema_avg_fn()(ema, torch.ones(5, device=device), 1)

    @dtypes(torch.float32, torch.float64)
    def test_ema_avg_fn_tensor_decay(self, device, dtype):
        ema = torch.tensor([0.7], device=device)
        current = torch.tensor([0.3], device=device)
        decay = torch.tensor([0.75], device=device, dtype=dtype)
        expected = decay * ema + (1 - decay) * current
        self.assertEqual(get_ema_avg_fn(decay)(ema, current, 1), expected)

    @dtypes(torch.float32, torch.complex64)
    @parametrize("decay", [0.0, 0.5, 0.9999, 1.0])
    def test_ema_avg_fn_extreme_values(self, device, dtype, decay):
        limit = torch.finfo(dtype).max
        ema = torch.tensor(
            [limit, float("inf"), float("nan"), 1e8], device=device, dtype=dtype
        )
        current = torch.tensor(
            [-limit, float("inf"), 1.0, 1.0], device=device, dtype=dtype
        )
        expected = decay * ema + (1 - decay) * current
        if 0 < decay < 1:
            # Interior updates use the multi-tensor helper's interpolation
            # semantics, including overflow and non-finite values.
            expected = ema.clone()
            get_ema_multi_avg_fn(decay)([expected], [current], 1)
        result = get_ema_avg_fn(decay)(ema, current, 1)
        self.assertEqual(result, expected, rtol=0, atol=0, equal_nan=True)

    @parametrize("use_buffers", [False, True])
    def test_averaged_model_ema_fixed_point(self, device, use_buffers):
        model = torch.nn.Linear(2, 2, device=device)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.7)
        model.register_buffer("floating_buffer", torch.full((2,), 0.7, device=device))
        model.register_buffer("integer_buffer", torch.tensor(3, device=device))
        averaged = AveragedModel(
            model, avg_fn=get_ema_avg_fn(0.9999), use_buffers=use_buffers
        )
        for _ in range(2001):
            averaged.update_parameters(model)
        for actual, expected in zip(
            averaged.module.state_dict().values(), model.state_dict().values()
        ):
            self.assertEqual(actual, expected, rtol=0, atol=0)
        self.assertEqual(averaged.n_averaged.item(), 2001)

    def _test_averaged_model(self, net_device, swa_device, ema):
        net_device = torch.device(net_device)
        swa_device = torch.device(swa_device)
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.BatchNorm2d(5, momentum=0.3),
            torch.nn.Conv2d(5, 2, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 10),
        ).to(net_device)

        averaged_params, averaged_dnn = self._run_averaged_steps(dnn, swa_device, ema)

        for p_avg, p_swa in zip(averaged_params, averaged_dnn.parameters()):
            self.assertEqual(p_avg, p_swa)
            # Check that AveragedModel is on the correct device
            self.assertTrue(p_swa.device == swa_device)
            self.assertTrue(p_avg.device == net_device)
        self.assertTrue(averaged_dnn.n_averaged.device == swa_device)

    def _run_averaged_steps(self, dnn, swa_device, ema):
        ema_decay = 0.999
        if ema:
            averaged_dnn = AveragedModel(
                dnn, device=swa_device, multi_avg_fn=get_ema_multi_avg_fn(ema_decay)
            )
        else:
            averaged_dnn = AveragedModel(
                dnn, device=swa_device, multi_avg_fn=get_swa_multi_avg_fn()
            )

        averaged_params = [torch.zeros_like(param) for param in dnn.parameters()]

        n_updates = 10
        for i in range(n_updates):
            for p, p_avg in zip(dnn.parameters(), averaged_params):
                p.detach().add_(torch.randn_like(p))
                if ema:
                    p_avg += (
                        p.detach()
                        * ema_decay ** (n_updates - i - 1)
                        * ((1 - ema_decay) if i > 0 else 1.0)
                    )
                else:
                    p_avg += p.detach() / n_updates
            averaged_dnn.update_parameters(dnn)

        return averaged_params, averaged_dnn

    @parametrize("ema", [True, False])
    def test_averaged_model_same_device(self, device, ema):
        self._test_averaged_model(device, device, ema)

    @onlyAccelerator
    @parametrize("ema", [True, False])
    def test_averaged_model_cross_device(self, device, ema):
        cpu = torch.device("cpu")
        self._test_averaged_model(device, cpu, ema)
        self._test_averaged_model(cpu, device, ema)

    @onlyAccelerator
    @parametrize("ema", [True, False])
    def test_averaged_model_mixed_device(self, device, ema):
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3), torch.nn.Linear(5, 10)
        )
        dnn[0].to(device)
        dnn[1].cpu()

        averaged_params, averaged_dnn = self._run_averaged_steps(dnn, None, ema)

        for p_avg, p_swa in zip(averaged_params, averaged_dnn.parameters()):
            self.assertEqual(p_avg, p_swa)
            # Check that AveragedModel is on the correct device
            self.assertTrue(p_avg.device == p_swa.device)

    def _test_update_bn(self, dnn, dl_x, dl_xy, device):
        preactivation_sum = torch.zeros(dnn.n_features, device=device)
        preactivation_squared_sum = torch.zeros(dnn.n_features, device=device)
        total_num = 0
        for x in dl_x:
            x = x[0].to(device)

            dnn.forward(x)
            preactivations = dnn.compute_preactivation(x)
            if len(preactivations.shape) == 4:
                preactivations = preactivations.transpose(1, 3)
            preactivations = preactivations.contiguous().view(-1, dnn.n_features)
            total_num += preactivations.shape[0]

            preactivation_sum += torch.sum(preactivations, dim=0)
            preactivation_squared_sum += torch.sum(preactivations**2, dim=0)

        preactivation_mean = preactivation_sum / total_num
        preactivation_var = preactivation_squared_sum / total_num
        preactivation_var = preactivation_var - preactivation_mean**2

        update_bn(dl_xy, dnn, device=device)
        self.assertEqual(preactivation_mean, dnn.bn.running_mean)
        self.assertEqual(preactivation_var, dnn.bn.running_var, atol=1e-1, rtol=0)

        def _reset_bn(module):
            if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)

        # reset batch norm and run update_bn again
        dnn.apply(_reset_bn)
        update_bn(dl_xy, dnn, device=device)
        self.assertEqual(preactivation_mean, dnn.bn.running_mean)
        self.assertEqual(preactivation_var, dnn.bn.running_var, atol=1e-1, rtol=0)
        # using the dl_x loader instead of dl_xy
        dnn.apply(_reset_bn)
        update_bn(dl_x, dnn, device=device)
        self.assertEqual(preactivation_mean, dnn.bn.running_mean)
        self.assertEqual(preactivation_var, dnn.bn.running_var, atol=1e-1, rtol=0)

    def test_update_bn_dnn(self, device):
        # Test update_bn for a fully-connected network with BatchNorm1d
        objects, input_features = 100, 5
        x = torch.rand(objects, input_features)
        y = torch.rand(objects)
        ds_x = torch.utils.data.TensorDataset(x)
        ds_xy = torch.utils.data.TensorDataset(x, y)
        dl_x = torch.utils.data.DataLoader(ds_x, batch_size=5, shuffle=True)
        dl_xy = torch.utils.data.DataLoader(ds_xy, batch_size=5, shuffle=True)
        dnn = SWATestDNN(input_features=input_features).to(device)
        dnn.train()
        self._test_update_bn(dnn, dl_x, dl_xy, device)
        self.assertTrue(dnn.training)

    def test_update_bn_cnn(self, device):
        # Test update_bn for convolutional network and BatchNorm2d
        objects = 100
        input_channels = 3
        height, width = 5, 5
        x = torch.rand(objects, input_channels, height, width)
        y = torch.rand(objects)
        ds_x = torch.utils.data.TensorDataset(x)
        ds_xy = torch.utils.data.TensorDataset(x, y)
        dl_x = torch.utils.data.DataLoader(ds_x, batch_size=5, shuffle=True)
        dl_xy = torch.utils.data.DataLoader(ds_xy, batch_size=5, shuffle=True)
        cnn = SWATestCNN(input_channels=input_channels).to(device)
        cnn.train()
        self._test_update_bn(cnn, dl_x, dl_xy, device)
        self.assertTrue(cnn.training)


if __name__ == "__main__":
    print("These tests should be run through test/test_optim.py instead")
