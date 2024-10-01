categories=("Uncategorized" "lazy" "hub" "mobile" "jit" "visualization" "onnx" "caffe2" "amd" "rocm" "cuda" "cpu" "cudnn" "xla"
"benchmark" "profiler" "performance_as_product" "package" "dispatcher" "releng" "fx"" "code_coverage" "vulkan" "skip" "composability" "mps" "intel
"functorch" "gnn" "distributions" "serialization")
frontend_categories=("meta_frontend" "nn_frontend" "linalg_frontend" "cpp_frontend" "python_frontend" "complex_frontend
"vmap_frontend" "autograd_frontend" "build_frontend" "memory_format_frontend" "foreach_frontend" "dataloader_frontend" "sparse_frontend
"nested tensor_frontend" "optimizer_frontend" "dynamo" "inductor" "quantization" "distributed")
topics=("bc breaking" "deprecation" "new features" "improvements" "bug fixes" "performance" "docs" "devs" "Untopiced" "not user facing" "security")

for category in "${categories[@]}"; do
  for topic in "${topics[@]}"; do
      # Run the Python script with category option
      printf "$category\n$topic\n" | python categorize.py
  done
done
