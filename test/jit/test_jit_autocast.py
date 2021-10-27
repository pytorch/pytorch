# Owner(s): ["oncall: jit"]

import torch
import torch.jit.frontend
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing import FileCheck, make_tensor
from jit.test_models import MnistNet

class TestJitAutocast(JitTestCase):
    def setUp(self):
        super(TestJitAutocast, self).setUp()
        self.models = [MnistNet()]
        self.inputs = [torch.randn(5, 1, 28, 28, device='cpu')]

    def tearDown(self):
        super(TestJitAutocast, self).tearDown()

    def test_generate_autocast_jit_trace_model(self):
        def test_generate_autocast_jit_trace_model(model, x):
            model.eval()
            with torch.cpu.amp.autocast(cache_enabled=False), torch.no_grad():
                traced_model = torch.jit.trace(model, x)
        for i in range(self.models.__len__()):
            test_generate_autocast_jit_trace_model(self.models[i], self.inputs[i])

    def test_nchw_autocast_jit_trace_model(self):
        def test_nchw_autocast_jit_trace_model(model, x):
            model.eval()
            with torch.cpu.amp.autocast(cache_enabled=False), torch.no_grad():
                traced_model = torch.jit.trace(model, x)
            with torch.cpu.amp.autocast(), torch.no_grad():
                y = traced_model(x.clone())
                y2 = model(x.clone())
            torch.testing.assert_allclose(y.double(), y2.double(), rtol=1e-03, atol=1e-03)
        for i in range(self.models.__len__()):
            test_nchw_autocast_jit_trace_model(self.models[i], self.inputs[i])

    def test_nhwc_autocast_jit_trace_model(self):
        def test_nhwc_autocast_jit_trace_model(model, x):
            model.eval()
            with torch.cpu.amp.autocast(cache_enabled=False), torch.no_grad():
                traced_model = torch.jit.trace(model, x.to(memory_format=torch.channels_last))
            with torch.cpu.amp.autocast(), torch.no_grad():
                y = traced_model(x.clone().to(memory_format=torch.channels_last))
                y2 = model(x.clone().to(memory_format=torch.channels_last))
            torch.testing.assert_allclose(y.double(), y2.double(), rtol=1e-03, atol=1e-03)
        for i in range(self.models.__len__()):
            if self.inputs[i].size().__len__() == 5:
                # NHWC 3D case not support yet
                continue
            test_nhwc_autocast_jit_trace_model(self.models[i], self.inputs[i])
