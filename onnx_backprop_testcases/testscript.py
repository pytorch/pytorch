import torch

class Model(torch.nn.Module):

    def forward(self, inputs):
        xseq = inputs[0].detach().requires_grad_()
        with torch.enable_grad():

            # differentiating through this "loss" fails to export:
            loss = torch.matmul(xseq, xseq.transpose(-2, -1)).sum()
            
            # differentiating through this "loss" also fails to export:
            # loss = torch.sqrt(xseq).sum()

            # differentiating through this "loss" is fine:
            # loss = torch.pow(xseq, 0.5).sum()
            
            return torch.autograd.grad([loss], [xseq])[0]


# export onnx:
torch.onnx.export(
    Model(),
    [torch.randn([2, 2, 10]) ** 2],
    "modelthing.onnx",
    input_names=["xseq"],
    opset_version=17,
    output_names=["lossgrad"],
    verbose=True
)


