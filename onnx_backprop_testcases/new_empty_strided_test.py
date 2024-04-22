import torch
import onnxruntime as ort
import sys


class SamplerWrapper(torch.nn.Module):

    def forward(self, inputs):
        x = inputs[0]

        with torch.enable_grad():
            
            model_in = x.detach().requires_grad_()
            pred = model_in

            pred = 2.0 * pred
            pred[..., 0] += pred[...,0]#pred_r_pos[..., 0:1]
            loss_tot =  pred.sum()

            grad_outs = [loss_tot]
            grad_ins = [model_in]
            all_grad = torch.autograd.grad(grad_outs, grad_ins)
            pred_xstart = all_grad[0]

        return pred_xstart

x = torch.rand(8, 264, 1, 196)

# export onnx:
from torch import override_nonsense
wrappedmodel = SamplerWrapper()
torch.onnx.export(
    wrappedmodel,
    [x],
    "modelthing.onnx",
    dynamic_axes={
        'x': {0: 'batch_size', 3: 'num_frames'}
    },
    input_names=["x"],
    opset_version=17,
    output_names=["pred"],
    verbose=True
)