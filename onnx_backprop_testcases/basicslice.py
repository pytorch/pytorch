import torch

class Model(torch.nn.Module):

    def forward(self, inputs):

        x = inputs[0]
        return x[:,1:]


batch_size=10
seq_len=20

#from torch import override_nonsense
torch.onnx.export(
    Model(),
    [torch.randn([batch_size, seq_len, 256])],
    "modelthing.onnx",
    dynamic_axes={'xseq': {0: 'batch_size', 1:'seq_len'}},
    input_names=["xseq"],
    opset_version=17,
    output_names=["lossgrad"],
    verbose=True
)