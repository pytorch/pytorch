# Owner(s): ["module: optimizer"]

import itertools
import pickle

import torch
from torch.optim.swa_utils import AveragedModel, update_bn, get_swa_multi_avg_fn, get_ema_multi_avg_fn
from torch.testing._internal.common_utils import (
    TestCase,
    load_tests,
    parametrize,
    instantiate_parametrized_tests,
)

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests


class TestSWAUtils(TestCase):
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

    def _test_averaged_model(self, net_device, swa_device, ema):
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
            averaged_dnn = AveragedModel(dnn, device=swa_device, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
        else:
            averaged_dnn = AveragedModel(dnn, device=swa_device, multi_avg_fn=get_swa_multi_avg_fn())

        averaged_params = [torch.zeros_like(param) for param in dnn.parameters()]

        n_updates = 10
        for i in range(n_updates):
            for p, p_avg in zip(dnn.parameters(), averaged_params):
                p.detach().add_(torch.randn_like(p))
                if ema:
                    p_avg += p.detach() * ema_decay ** (n_updates - i - 1) * ((1 - ema_decay) if i > 0 else 1.0)
                else:
                    p_avg += p.detach() / n_updates
            averaged_dnn.update_parameters(dnn)

        return averaged_params, averaged_dnn

    @parametrize("ema", [True, False])
    def test_averaged_model_all_devices(self, ema):
        cpu = torch.device("cpu")
        self._test_averaged_model(cpu, cpu, ema)
        if torch.cuda.is_available():
            cuda = torch.device(0)
            self._test_averaged_model(cuda, cpu, ema)
            self._test_averaged_model(cpu, cuda, ema)
            self._test_averaged_model(cuda, cuda, ema)

    @parametrize("ema", [True, False])
    def test_averaged_model_mixed_device(self, ema):
        if not torch.cuda.is_available():
            return
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3), torch.nn.Linear(5, 10)
        )
        dnn[0].cuda()
        dnn[1].cpu()

        averaged_params, averaged_dnn = self._run_averaged_steps(dnn, None, ema)

        for p_avg, p_swa in zip(averaged_params, averaged_dnn.parameters()):
            self.assertEqual(p_avg, p_swa)
            # Check that AveragedModel is on the correct device
            self.assertTrue(p_avg.device == p_swa.device)

    def test_averaged_model_state_dict(self):
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3), torch.nn.Linear(5, 10)
        )
        averaged_dnn = AveragedModel(dnn)
        averaged_dnn2 = AveragedModel(dnn)
        n_updates = 10
        for i in range(n_updates):
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
        # Test AveragedModel with EMA as avg_fn and use_buffers as True.
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3),
            torch.nn.BatchNorm2d(5, momentum=0.3),
            torch.nn.Linear(5, 10),
        )
        decay = 0.9

        if use_multi_avg_fn:
            averaged_dnn = AveragedModel(dnn, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=use_buffers)
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

    def _test_update_bn(self, dnn, dl_x, dl_xy, cuda):

        preactivation_sum = torch.zeros(dnn.n_features)
        preactivation_squared_sum = torch.zeros(dnn.n_features)
        if cuda:
            preactivation_sum = preactivation_sum.cuda()
            preactivation_squared_sum = preactivation_squared_sum.cuda()
        total_num = 0
        for x in dl_x:
            x = x[0]
            if cuda:
                x = x.cuda()

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

        update_bn(dl_xy, dnn, device=x.device)
        self.assertEqual(preactivation_mean, dnn.bn.running_mean)
        self.assertEqual(preactivation_var, dnn.bn.running_var, atol=1e-1, rtol=0)

        def _reset_bn(module):
            if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)

        # reset batch norm and run update_bn again
        dnn.apply(_reset_bn)
        update_bn(dl_xy, dnn, device=x.device)
        self.assertEqual(preactivation_mean, dnn.bn.running_mean)
        self.assertEqual(preactivation_var, dnn.bn.running_var, atol=1e-1, rtol=0)
        # using the dl_x loader instead of dl_xy
        dnn.apply(_reset_bn)
        update_bn(dl_x, dnn, device=x.device)
        self.assertEqual(preactivation_mean, dnn.bn.running_mean)
        self.assertEqual(preactivation_var, dnn.bn.running_var, atol=1e-1, rtol=0)

    def test_update_bn_dnn(self):
        # Test update_bn for a fully-connected network with BatchNorm1d
        objects, input_features = 100, 5
        x = torch.rand(objects, input_features)
        y = torch.rand(objects)
        ds_x = torch.utils.data.TensorDataset(x)
        ds_xy = torch.utils.data.TensorDataset(x, y)
        dl_x = torch.utils.data.DataLoader(ds_x, batch_size=5, shuffle=True)
        dl_xy = torch.utils.data.DataLoader(ds_xy, batch_size=5, shuffle=True)
        dnn = self.SWATestDNN(input_features=input_features)
        dnn.train()
        self._test_update_bn(dnn, dl_x, dl_xy, False)
        if torch.cuda.is_available():
            dnn = self.SWATestDNN(input_features=input_features)
            dnn.train()
            self._test_update_bn(dnn.cuda(), dl_x, dl_xy, True)
        self.assertTrue(dnn.training)

    def test_update_bn_cnn(self):
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
        cnn = self.SWATestCNN(input_channels=input_channels)
        cnn.train()
        self._test_update_bn(cnn, dl_x, dl_xy, False)
        if torch.cuda.is_available():
            cnn = self.SWATestCNN(input_channels=input_channels)
            cnn.train()
            self._test_update_bn(cnn.cuda(), dl_x, dl_xy, True)
        self.assertTrue(cnn.training)

    def test_bn_update_eval_momentum(self):
        # check that update_bn preserves eval mode
        objects = 100
        input_channels = 3
        height, width = 5, 5
        x = torch.rand(objects, input_channels, height, width)
        ds_x = torch.utils.data.TensorDataset(x)
        dl_x = torch.utils.data.DataLoader(ds_x, batch_size=5, shuffle=True)
        cnn = self.SWATestCNN(input_channels=input_channels)
        cnn.eval()
        update_bn(dl_x, cnn)
        self.assertFalse(cnn.training)

        # check that momentum is preserved
        self.assertEqual(cnn.bn.momentum, 0.3)


instantiate_parametrized_tests(TestSWAUtils)


if __name__ == "__main__":
    print("These tests should be run through test/test_optim.py instead")
