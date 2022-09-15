python -m jschema_to_python \
    --schema-path sarif-schema-2.1.0.json \
    --module-name torch.onnx.sarif_om \
    --output-directory ../../../torch/onnx/sarif_om \
    --root-class-name SarifLog \
    --hints-file-path code-gen-hints.json \
    --force \
    -vv
