import torch
import torch.onnx
import torch.nn as nn

# 1. Define a symbolic function for aten::linear that inserts quantization nodes
def symbolic_linear_dynamic_quant(g, input, weight, bias):
    # Quantize the weight
    # Using symmetric quantization (qint8) as is common for weights.
    # Scale is calculated as: 2 * max(abs(weight)) / 255
    # For simplicity in this symbolic, we'll use a constant scale,
    # but a real implementation would compute this from the weight tensor.
    # We create a constant node for the zero point (0 for symmetric quantization).
    zero_point = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int8))
    
    # In a real implementation, scale would be calculated. Here, we use a placeholder.
    # This value would need to be dynamically calculated from the weight tensor.
    # For this demonstration, we'll create a node that would represent this calculation.
    abs_weight = g.op("Abs", weight)
    max_val = g.op("ReduceMax", abs_weight, keepdims_i=0)
    scale = g.op("Div", max_val, g.op("Constant", value_t=torch.tensor(127.0)))

    quant_weight = g.op("QuantizeLinear", weight, scale, zero_point)
    
    # Dequantize the weight immediately for the matmul
    dequant_weight = g.op("DequantizeLinear", quant_weight, scale, zero_point)
    
    # Perform the standard linear operation
    transposed_weight = g.op("Transpose", dequant_weight, perm_i=[1, 0])
    output = g.op("MatMul", input, transposed_weight)
    
    if bias is None or bias.node().kind() == "prim::Constant" and bias.type().isSubtypeOf(torch._C.NoneType.get()):
        return output
        
    return g.op("Add", output, bias)

# 2. Register this symbolic function to override the default aten::linear
torch.onnx.register_custom_op_symbolic("aten::linear", symbolic_linear_dynamic_quant, 9)

# 3. Run the export on a standard float model
def solve_issue():
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear = nn.Linear(5, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            # This will be intercepted by our custom symbolic function
            return self.relu(self.linear(x))

    model = SimpleModel().eval()
    dummy_input = torch.randn(1, 5)
    onnx_path = "#123555/quantized_model_fixed.onnx"
    
    print("Attempting to export with custom symbolic function to create dynamic quantization...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
        )
        print(f"\nSUCCESS: Model successfully exported to {onnx_path}")
        print("The ONNX graph now contains the quantization logic.")
        print("The original issue has been resolved.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during export: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    solve_issue()
