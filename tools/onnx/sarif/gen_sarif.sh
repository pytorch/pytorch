#!/bin/bash
# Run this script inside its folder to generate the SARIF python object model files
# from the SARIF schema.
# e.g. sh gen_sarif.sh
#
# This script requires the jschema_to_python package to be installed.
# To install it, run:
#   pip install jschema_to_python
set -e -x
ROOT=$(pwd)/../../..
SARIF_DIR=torch/onnx/_internal/diagnostics/infra/sarif_om

# SARIF version
SARIF_VERSION=2.1.0
SARIF_SCHEMA_LINK=https://docs.oasis-open.org/sarif/sarif/v2.1.0/cs01/schemas/sarif-schema-2.1.0.json

# Download SARIF schema
tmp_dir=$(mktemp -d)
sarif_schema_file_path=$tmp_dir/sarif-schema-$SARIF_VERSION.json
wget -O $sarif_schema_file_path $SARIF_SCHEMA_LINK

python -m jschema_to_python \
    --schema-path $sarif_schema_file_path \
    --module-name torch.onnx._internal.diagnostics.infra.sarif_om \
    --output-directory $ROOT/$SARIF_DIR \
    --root-class-name SarifLog \
    --hints-file-path code-gen-hints.json \
    --force \
    --library dataclasses \
    -vv

# Generate SARIF version file
echo "SARIF_VERSION = \""$SARIF_VERSION"\"" > $ROOT/$SARIF_DIR/version.py
echo "SARIF_SCHEMA_LINK = \""$SARIF_SCHEMA_LINK"\"" >> $ROOT/$SARIF_DIR/version.py

# Hack to have flake8 not complain about generated code.
cd $ROOT
for f in $(find $SARIF_DIR -name '*.py'); do
    echo "# flake8: noqa" >> $f
done

lintrunner $SARIF_DIR/** -a
