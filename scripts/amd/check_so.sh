# roc-obj-ls /tmp/pytorch/build/lib/libtorch_hip.so 2>&1 |tee /dockerx/pytorch/libtorch_hip.log
# roc-obj-ls /opt/conda/lib/python3.6/site-packages/mlp_cuda.cpython-36m-x86_64-linux-gnu.so 2>&1 |tee /dockerx/pytorch/mlp_cuda.log
# roc-obj-ls /root/.local/lib/python3.6/site-packages/torchvision/image.so 2>&1 |tee /dockerx/pytorch/torchvision_image.log
# roc-obj-ls /root/.local/lib/python3.6/site-packages/torchvision/_C.so 2>&1 |tee /dockerx/pytorch/torchvision_C.log
roc-obj-ls /opt/conda/lib/python3.6/site-packages/torchvision/_C.so 2>&1 |tee /dockerx/pytorch/torchvision_C.log
# find /opt/conda/lib/python3.6/site-packages/torchvision -name *.so