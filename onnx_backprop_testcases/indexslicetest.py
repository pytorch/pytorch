import torch
import onnxruntime as ort
import sys

from torch import override_nonsense

class Model(torch.nn.Module):

    def forward(self, inputs):
        pred = inputs[0]
        pred = pred.detach().requires_grad_()
        with torch.enable_grad():
            
            # looks like this passes a list that looks like this into the system:
            #    [None, None, [0]]
            # For some reason, my _index_put_impl override tries to sum all the elements
            # in that list (or at least a list that looks like that), and I get errors
            # from adding None to something.
            loss_tot = pred[..., [0]].sum()
            grad_outs = [loss_tot]
            grad_ins = [pred]
            return torch.autograd.grad(grad_outs, grad_ins)[0]

mdl = Model()


print("export model")
torch.onnx.export(
    Model(),
    [torch.randn([20, 2, 256])],
    "modelthing.onnx",
    input_names=["xseq"],
    opset_version=17,
    output_names=["lossgrad"],
    verbose=True
)
