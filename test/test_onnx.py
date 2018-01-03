import io
from common import TestCase
from torch.autograd import Variable


class TestONNXExport(TestCase):
    """
    Ensures that basic pytorch Modules can be exported to onnx without errors
    """

    def test_export_linear_bias(self):
        n_feat = 16
        linear = torch.nn.Linear(n_feat, 2, bias=True)
        args = tuple([Variable(torch.randn(1, n_feat), requires_grad=True)])

        with io.BytesIO() as stream:
            torch.onnx.export(linear, args, stream, export_params=False)

    def test_export_linear_nobias(self):
        n_feat = 16
        linear = torch.nn.Linear(n_feat, 2, bias=False)
        args = tuple([Variable(torch.randn(1, n_feat), requires_grad=True)])

        with io.BytesIO() as stream:
            torch.onnx.export(linear, args, stream, export_params=False)
