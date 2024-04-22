import torch
import onnxruntime as ort

class Model(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

        self.linearlayer = torch.nn.Linear(10, 5)
        
    def forward(self, inputs):
        xseq = inputs[0].detach().requires_grad_()
        with torch.enable_grad():

            # differentiating through this "loss" fails to export:
            loss = self.linearlayer(xseq).sum()
            
            return torch.autograd.grad([loss], [xseq])[0]


batch_size = 2

# export onnx:
torch.onnx.export(
    Model(),
    [torch.randn([batch_size, 3, 2, 10])],
    "modelthing.onnx",
    input_names=["xseq"],
    dynamic_axes={'xseq': {0: 'batch_size'}},
    opset_version=17,
    output_names=["lossgrad"],
    verbose=True
)

# load and evaluate with a different shape:
ort_session = ort.InferenceSession("modelthing.onnx")

print("run onnx with export shape")
ort_session.run(None, { "xseq":torch.randn([batch_size, 3, 2, 10]).numpy() })
print("success")

# comment this out and it should work:
batch_size=10

print("run onnx with different batch size")
ort_session.run(None, { "xseq":torch.randn([batch_size, 3, 2, 10]).numpy() })
print("success")

