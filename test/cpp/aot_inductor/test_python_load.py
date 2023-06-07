import unittest
import torch
import inductor_module

class TestAoTInductorInPython(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.check_call(['python', 'setup.py', 'install'])
        
    def test_save(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(64, 10)

            def forward(self, x, y):
                return self.fc(torch.sin(x) + torch.cos(y))
            
        x = torch.randn((32, 64), device="cuda")
        y = torch.randn((32, 64), device="cuda")

        with torch.no_grad():
            from torch.fx.experimental.proxy_tensor import make_fx
            module = make_fx(Net().cuda())(x, y)
            lib_path = torch._inductor.aot_compile(module, [x, y])
    
    def test_load(self):
        net = inductor_module.InductorModule()

        x = torch.randn((32, 64), device="cuda")
        y = torch.randn((32, 64), device="cuda")

        output = net(x, y)

        self.assertEqual(output.shape, (32, 10))
