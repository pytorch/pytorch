# python3 scripts/amd/Pytorch_LSTMModel.py --batch_size=32 --warm_up=2 --num_test=16 --distributed=False

PYTORCH_TEST_WITH_ROCM=1 python test/test_nn.py #--verbose TestFFTCUDA.test_batch_istft_cuda
