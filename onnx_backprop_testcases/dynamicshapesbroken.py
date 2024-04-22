import torch
import onnxruntime as ort
import numpy as np

doBackprop = True

class Model(torch.nn.Module):

    def forward(self, inputs):

        if doBackprop:
            xseq = inputs[0].detach().requires_grad_()
            with torch.enable_grad():
                #y = torch.pow(xseq, 2).sum(dim=-1).mean(dim=-1)
                #loss = y.sum() + y.mean()
                loss = xseq.flatten(start_dim=-2).sum()
                return xseq + torch.autograd.grad([loss], [xseq])[0]
        else:
            xseq = inputs[0]
            return torch.pow(xseq, 2).sum()


# export onnx:
torch.onnx.export(
    Model(),
    [torch.randn([2, 2, 10]) ** 2],
    "modelthing.onnx",
    input_names=["x"],
    dynamic_axes={'x': {0: 'batch_size'}},
    opset_version=17,
    output_names=["lossgrad"],
    verbose=True
)

# load and evaluate with a different shape:
ort_session = ort.InferenceSession("modelthing.onnx")

#print("run onnx with export shape")
#ort_session.run(None, { "x":torch.randn([2, 2, 10]).numpy() })
#print("success")

print("run onnx with different batch size")
ort_session.run(None, { "x":torch.randn([4, 2, 10]).numpy() })
print("success")
