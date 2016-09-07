python test_torch.py
python test_autograd.py
python test_nn.py
python test_legacy_nn.py
if which nvcc >/dev/null 2>&1
then
    python test_cuda.py
fi
