import torch
import onnxruntime as ort

from torch import override_nonsense


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.latent_dim = 256
        self.num_heads = 4
        self.ff_size=1024
        self.dropout=0.1
        self.activation="gelu"
        self.num_layers = 4

        root_seqTransEncoderLayer = torch.nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                               nhead=self.num_heads,
                                                               dim_feedforward=self.ff_size,
                                                               dropout=self.dropout,
                                                               activation=self.activation)

        self.root_seqTransEncoder = torch.nn.TransformerEncoder(root_seqTransEncoderLayer,
                                                          num_layers=self.num_layers)

    def forward(self, inputs):
        xin = inputs[0].detach().requires_grad_()
        with torch.enable_grad():

            x = xin

            x = self.root_seqTransEncoder(x)
            loss = x.sum()

            return torch.autograd.grad([loss], [xin])[0]


batch_size=20
seq_len=2

mdl = Model()
for p in mdl.parameters():
    p.requires_grad_(False)


print("run model before overrides")

inputs = [torch.randn([batch_size, seq_len, 256])]
results_before = mdl(inputs)

from torch import override_nonsense
results_after = mdl(inputs)

print(torch.abs(results_before - results_after).max())


print("export model")
torch.onnx.export(
    mdl,
    [torch.randn([batch_size, seq_len, 256])],
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
pyresults = mdl(inputs)
onnxresults = ort_session.run(None, { "xseq":inputs.numpy() })[0]

print(torch.abs(pyresults-onnxresults).max())
print("success")

batch_size=10
seq_len=5
print("run onnx with different batch size")
inputs = torch.randn([batch_size, seq_len, 256])
pyresults = mdl(inputs)
onnxresults = ort_session.run(None, { "xseq":inputs.numpy() })[0]
print(torch.abs(pyresults-onnxresults).max())
print("success")