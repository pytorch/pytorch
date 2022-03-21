FILE_LIST=(
    # test_fx
    # test_fx_experimental
    test_jit
    # test_jit_autocast
    # test_jit_fuser_te
    # test_mkldnn
    # test_mobile_optimizer
    # test_module_init
    # test_quantization
    # test_tensor_creation_ops
    # test_tensorboard
)

for FILE_NAME in "${FILE_LIST[@]}"; do
    gdb -ex "set pagination off" \
        -ex "file python" \
        -ex "run /tmp/pytorch/test/${FILE_NAME}.py --verbose" \
        -ex "bt" \
        -ex "set confirm off" \
        -ex "q" \
        2>&1 | tee /dockerx/pytorch/${FILE_NAME}_gdb.log
done
