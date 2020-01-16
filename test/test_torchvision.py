import torch
from torchvision import models
from test_jit import get_execution_plan
from common_utils import enable_profiling_mode

def get_available_classification_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]

"""
def get_available_segmentation_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.segmentation.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


def get_available_detection_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.detection.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


def get_available_video_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.video.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]

    def _test_classification_model(self, name, input_shape):
        set_rng_seed(0)
        # passing num_class equal to a number other than 1000 helps in making the test
        # more enforcing in nature
        model = models.__dict__[name](num_classes=50)
        model.eval()
        x = torch.rand(input_shape)
        out = model(x)
        self.assertExpected(out, prec=0.1)
        self.assertEqual(out.shape[-1], 50)
        self.checkModule(model, name, (x,))
"""

def get_plan(model):
    ##fwd = model._c._get_method('forward')
    ##fwd.get_debug_state()
    state = model._c.get_debug_state()
    plan = get_execution_plan(state)
    num_bailouts = plan.code.num_bailouts()
    return plan

def do_test(model_name):
    with enable_profiling_mode():
        input_shape = (1, 3, 224, 224)
        if model_name in ['inception_v3']:
            input_shape = (1, 3, 299, 299)
        print("testing ", model_name)
        open("2_{}".format(model_name), 'a').close()
        torch.manual_seed(0)
        model = models.__dict__[model_name](num_classes=50)
        scripted_model = torch.jit.script(model)
        scripted_model.eval()
        x = torch.rand(input_shape)
        py_output = model(x)
        scripted_model(x)
        opt_output = scripted_model(x)
        #assert torch.allclose(py_output, opt_output)
        plan = get_plan(scripted_model)
        num_bailouts = plan.code.num_bailouts()
        print(num_bailouts)
        for i in range(0, num_bailouts):
            plan.code.request_bailout(i)
            bailout_output = scripted_model(x)
            #assert torch.allclose(bailout_output, opt_output)

def test_available_classification_models():
    for model_name in get_available_classification_models():
        # for-loop bodies don't define scopes, so we have to save the variables
        # we want to close over in some way
        if "alexnet" in model_name or "resnet" in model_name:
            print("skipping ", model_name)
        else:
            do_test(model_name)

test_available_classification_models()
