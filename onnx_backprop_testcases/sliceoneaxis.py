import torch
import onnxruntime as ort

from torch import override_nonsense


class Model(torch.nn.Module):

    def forward(self, inputs):

        if 0:
            x = inputs[0]
            #out = torch.zeros(x.shape[0]*2, x.shape[1], x.shape[2])
            #out[0::2] = x
            out = torch.zeros(x.shape[0]*2)
            out[0::2] = x
            return out
        
        x = inputs[0].detach().requires_grad_()
        with torch.enable_grad():

            v = x[:,1:]# - x[:,:-1]
            loss = v.sum()

            return torch.autograd.grad([loss], [x])[0]


batch_size=10
seq_len=20

mdl = Model()
from torch import override_nonsense

print("export model")
torch.onnx.export(
    mdl,
    [torch.randn([batch_size, seq_len, 256])],
    #[torch.randn([batch_size])],
    "modelthing.onnx",
    dynamic_axes={'xseq': {0: 'batch_size', 1:'seq_len'}},
    input_names=["xseq"],
    opset_version=17,
    output_names=["lossgrad"],
    verbose=True
)

# load and evaluate with a different shape:
ort_session = ort.InferenceSession("modelthing.onnx")

print("run onnx with export shape")
inputs = torch.randn([batch_size, seq_len, 256])

pyresults = mdl([inputs])
onnxresults = ort_session.run(None, { "xseq":inputs.numpy() })[0]

print("error:",torch.abs(pyresults-onnxresults).max())
print("success")

batch_size=12
seq_len=5
print("run onnx with different batch size")
inputs = torch.randn([batch_size, seq_len, 256])
pyresults = mdl([inputs])
onnxresults = ort_session.run(None, { "xseq":inputs.numpy() })[0]
print(torch.abs(pyresults-onnxresults).max())
print("success")

