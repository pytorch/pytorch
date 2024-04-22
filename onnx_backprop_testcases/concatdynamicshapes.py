import torch
import onnxruntime as ort
from torch import override_nonsense

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.noise = torch.rand(1,10000,1)


    def forward(self, inputs):

        x = inputs[0].detach().requires_grad_()
        with torch.enable_grad():

            out = torch.cat([x, x], dim=1)
            loss = out.sum()

            return torch.autograd.grad([loss], [x])[0]


batch_size=10
seq_len=20

mdl = Model()

print("export model")
torch.onnx.export(
    mdl,
    [torch.randn([batch_size, seq_len, 256])],
    "modelthing.onnx",
    dynamic_axes={'xseq': {0: 'batch_size', 1:'seq_len'}},
    input_names=["xseq"],
    opset_version=17,
    output_names=["lossgrad"],
    verbose=True,
    do_constant_folding=False
)

# load and evaluate with a different shape:
ort_session = ort.InferenceSession("modelthing.onnx")

print("run onnx with export shape")
inputs = torch.randn([batch_size, seq_len, 256])
pyresults = mdl([inputs])
onnxresults = ort_session.run(None, { "xseq":inputs.numpy() })[0]

print("error:", torch.abs(pyresults-onnxresults).max())
print("success")

batch_size=12
seq_len=5

print("run onnx with different batch size")
inputs = torch.randn([batch_size, seq_len, 256])
pyresults = mdl([inputs])
onnxresults = ort_session.run(None, { "xseq":inputs.numpy() })[0]
print("error:", torch.abs(pyresults-onnxresults).max())
print("success")

