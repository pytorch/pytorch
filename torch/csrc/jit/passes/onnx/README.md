The optimization passes in this directory work exclusively on ONNX-style IRs,
e.g., IRs that have had ToONNX applied to them.  ONNX defines operators
differently from ATen, so there are different opportunities for peephole
optimization.
