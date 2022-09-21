#!/bin/bash
ROOT=$(pwd)/../../..
SARIF_DIR=torch/onnx/_internal/diagnostics/infra/sarif_om
python -m jschema_to_python \
    --schema-path sarif-schema-2.1.0.json \
    --module-name torch.onnx._internal.diagnostics.infra.sarif_om \
    --output-directory $ROOT/$SARIF_DIR \
    --root-class-name SarifLog \
    --hints-file-path code-gen-hints.json \
    --force \
    -vv

# hack to have linter not complain about generated code.
cd $ROOT
for f in $(find $SARIF_DIR -name '*.py'); do
    echo "# flake8: noqa" >> $f
done
lintrunner $SARIF_DIR/** -a
