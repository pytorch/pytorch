import torch


class Model(torch.nn.Module):

    def forward(self, inputs):
        xseq = inputs[0]
        xseq = xseq.detach().requires_grad_()
        with torch.enable_grad():

            loss = torch.sqrt(xseq)
            #loss = torch.pow(xseq, 0.5)
            return torch.autograd.grad([loss], [xseq])[0]

mdl = Model()

# export onnx:
print("export model")
torch.onnx.export(
    Model(),
    [torch.randn(1) ** 2],
    "modelthing.onnx",
    input_names=["xseq"],
    opset_version=17,
    output_names=["lossgrad"],
    verbose=True
)



