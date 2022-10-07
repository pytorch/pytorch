# Owner(s): ["module: dynamo"]
from torch.dynamo.testing import make_test_cls_with_patches

from . import test_functions
from . import test_misc
from . import test_modules
from . import test_repros
from . import test_unspec


def make_no_fake_cls(cls):
    return make_test_cls_with_patches(
        cls, "NoFakeTensors", "_no_fake_tensors", ("fake_tensor_propagation", False)
    )


NoFakeTensorsFunctionTests = make_no_fake_cls(test_functions.FunctionTests)
NoFakeTensorsMiscTests = make_no_fake_cls(test_misc.MiscTests)
NoFakeTensorsReproTests = make_no_fake_cls(test_repros.ReproTests)
NoFakeTensorsNNModuleTests = make_no_fake_cls(test_modules.NNModuleTests)
NoFakeTensorsUnspecTests = make_no_fake_cls(test_unspec.UnspecTests)
