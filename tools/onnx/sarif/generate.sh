#!/bin/bash
ROOT=$(pwd)/../../..
python -m jschema_to_python \
    --schema-path sarif-schema-2.1.0.json \
    --module-name torch.onnx.sarif_om \
    --output-directory $ROOT/torch/onnx/sarif_om \
    --root-class-name SarifLog \
    --hints-file-path code-gen-hints.json \
    --force \
    -vv

# hack to have linter not complain about generated code.
cd $ROOT
for f in $(find torch/onnx/sarif_om -name '*.py'); do
    echo "# flake8: noqa" >> $f
done
lintrunner -a torch/onnx/sarif_om/**
