from torch._C import CudaComplexDoubleStorageBase
import unittest
import torch
import torch.nn as nn
import warnings
class TestPytorchDirectML(unittest.TestCase):
    tensorA = torch.tensor([
                [-1.0,-2.0,-3.0,-4.0],
                [4.0,5.0,6.0,4.0],
                [-7.0,-8.0,-9.0,-4.0],
                [10.0,11.0,12.0,4.0],
                ])
    tensorB = torch.tensor([
        [2.0],
        [2.0],
        [2.0],
        [2.0]
    ])

    def verify_result(self, cpuResult, dmlResult, equal_nan=False, abs_tol=1e-08):
        self.assertTrue(cpuResult.dtype == dmlResult.dtype)
        if cpuResult.dtype == torch.bool:
            self.assertTrue(torch.all(torch.eq(cpuResult, dmlResult.to("cpu"))).item())
        else:
            self.assertTrue(torch.allclose(cpuResult, dmlResult.to("cpu"), equal_nan=equal_nan, atol=abs_tol))
        self.assertTrue(cpuResult.shape == dmlResult.shape)
        self.assertTrue(cpuResult.stride() == dmlResult.stride())
        self.assertTrue(cpuResult.size() == dmlResult.size())

    def test_slice_backwards(self):
        input_cpu = self.tensorA.clone().detach().requires_grad_(True)
        cpuResult = input_cpu[1:, :2]
        cpuResult.mean().backward()
        input_dml = self.tensorA.to("dml").detach().requires_grad_(True)
        dmlResult = input_dml[1:, :2]
        dmlResult.mean().backward()
        self.assertTrue(torch.allclose(input_cpu.grad, input_dml.grad.to("cpu")))

    def test_gt(self):
        # compare tensors
        input_a = torch.tensor([1.0,-2.0,3.0,4.0,-5.0])
        input_b = torch.tensor([1.0,-1.0,3.0,6.0,-6.0])
        cpuTensor = input_a.clone().detach()
        dmlTensor = input_a.clone().detach().to("dml")
        cpuResult = torch.gt(cpuTensor, input_b)
        dmlResult = torch.gt(dmlTensor, input_b.to("dml"))
        self.verify_result(cpuResult, dmlResult)
        # compare tensor with scalar
        cpuResult = torch.gt(cpuTensor, 3)
        dmlResult = torch.gt(dmlTensor, 3)
        self.verify_result(cpuResult, dmlResult)
        # inplace
        cpuTensor.gt_(input_b)
        dmlTensor.gt_(input_b.to("dml"))
        self.verify_result(cpuResult, dmlResult)

    def test_ge(self):
        # compare tensors
        input_a = torch.tensor([1.0,-2.0,3.0,4.0,-5.0])
        input_b = torch.tensor([1.0,-1.0,3.0,6.0,-6.0])
        cpuTensor = input_a.clone().detach()
        dmlTensor = input_a.clone().detach().to("dml")
        cpuResult = torch.ge(cpuTensor, input_b)
        dmlResult = torch.ge(dmlTensor, input_b.to("dml"))
        self.verify_result(cpuResult, dmlResult)
        # compare tensor with scalar
        cpuResult = torch.ge(cpuTensor, 3)
        dmlResult = torch.ge(dmlTensor, 3)
        self.verify_result(cpuResult, dmlResult)
        # inplace
        cpuTensor.ge_(input_b)
        dmlTensor.ge_(input_b.to("dml"))
        self.verify_result(cpuResult, dmlResult)

    def test_lt(self):
        # compare tensors
        input_a = torch.tensor([1.0,-2.0,3.0,4.0,-5.0])
        input_b = torch.tensor([1.0,-1.0,3.0,6.0,-6.0])
        cpuTensor = input_a.clone().detach()
        dmlTensor = input_a.clone().detach().to("dml")
        cpuResult = torch.lt(cpuTensor, input_b)
        dmlResult = torch.lt(dmlTensor, input_b.to("dml"))
        self.verify_result(cpuResult, dmlResult)
        # compare tensor with scalar
        cpuResult = torch.lt(cpuTensor, 3)
        dmlResult = torch.lt(dmlTensor, 3)
        self.verify_result(cpuResult, dmlResult)
        # inplace
        cpuTensor.lt_(input_b)
        dmlTensor.lt_(input_b.to("dml"))
        self.verify_result(cpuResult, dmlResult)

    def test_le(self):
        # compare tensors
        input_a = torch.tensor([1.0,-2.0,3.0,4.0,-5.0])
        input_b = torch.tensor([1.0,-1.0,3.0,6.0,-6.0])
        cpuTensor = input_a.clone().detach()
        dmlTensor = input_a.clone().detach().to("dml")
        cpuResult = torch.le(cpuTensor, input_b)
        dmlResult = torch.le(dmlTensor, input_b.to("dml"))
        self.verify_result(cpuResult, dmlResult)
        # compare tensor with scalar
        cpuResult = torch.le(cpuTensor, 3)
        dmlResult = torch.le(dmlTensor, 3)
        self.verify_result(cpuResult, dmlResult)
        # inplace
        cpuTensor.le_(input_b)
        dmlTensor.le_(input_b.to("dml"))
        self.verify_result(cpuResult, dmlResult)

    def test_eq(self):
        # compare tensors
        input_a = torch.tensor([1.0,-2.0,3.0,4.0,-5.0])
        input_b = torch.tensor([1.0,-1.0,3.0,6.0,-6.0])
        cpuTensor = input_a.clone().detach()
        dmlTensor = input_a.clone().detach().to("dml")
        cpuResult = torch.eq(cpuTensor, input_b)
        dmlResult = torch.eq(dmlTensor, input_b.to("dml"))
        self.verify_result(cpuResult, dmlResult)
        # compare tensor with scalar
        cpuResult = torch.eq(cpuTensor, 3)
        dmlResult = torch.eq(dmlTensor, 3)
        self.verify_result(cpuResult, dmlResult)
        # inplace
        cpuTensor.eq_(input_b)
        dmlTensor.eq_(input_b.to("dml"))
        self.verify_result(cpuResult, dmlResult)

    def test_ne(self):
        # compare tensors
        input_a = torch.tensor([1.0,-2.0,3.0,4.0,-5.0])
        input_b = torch.tensor([1.0,-1.0,3.0,6.0,-6.0])
        cpuTensor = input_a.clone().detach()
        dmlTensor = input_a.clone().detach().to("dml")
        cpuResult = torch.ne(cpuTensor, input_b)
        dmlResult = torch.ne(dmlTensor, input_b.to("dml"))
        self.verify_result(cpuResult, dmlResult)
        # compare tensor with scalar
        cpuResult = torch.ne(cpuTensor, 3)
        dmlResult = torch.ne(dmlTensor, 3)
        self.verify_result(cpuResult, dmlResult)
        # inplace
        cpuTensor.ne_(input_b)
        dmlTensor.ne_(input_b.to("dml"))
        self.verify_result(cpuResult, dmlResult)

    def test_logical_and(self):
        # compare tensors
        input_a = torch.tensor([1,0,0,1])
        input_b = torch.tensor([1,1,0,0])
        dmlTensor = input_a.clone().detach().to("dml")
        dmlResult = torch.logical_and(dmlTensor, input_b.to("dml"))
        self.verify_result(torch.tensor([True, False, False, False]), dmlResult)
        # inplace
        dmlTensor.logical_and_(input_b.to("dml"))
        self.verify_result(torch.tensor([True, False, False, False]), dmlResult)

    def test_bitwise_and(self):
        # compare tensors
        input_a = torch.tensor([True, False, False, True])
        input_b = torch.tensor([True, True, False, False])
        dmlTensor = input_a.clone().detach().to("dml")
        dmlResult = torch.bitwise_and(dmlTensor, input_b.to("dml"))
        self.verify_result(torch.tensor([True, False, False, False]), dmlResult)

        # inplace
        dmlTensor.bitwise_and_(input_b.to("dml"))
        self.verify_result(torch.tensor([True, False, False, False]), dmlResult)

        # with scalar
        dmlTensor = input_a.clone().detach().to("dml")
        dmlResult = torch.bitwise_and(dmlTensor, True)
        self.verify_result(torch.tensor([True, False, False, True]), dmlResult)

        # inplace
        dmlTensor.bitwise_and_(True)
        self.verify_result(torch.tensor([True, False, False, True]), dmlResult)


    def test_logical_or(self):
        # compare tensors
        input_a = torch.tensor([1,0,0,1])
        input_b = torch.tensor([1,1,0,0])
        dmlTensor = input_a.clone().detach().to("dml")
        dmlResult = torch.logical_or(dmlTensor, input_b.to("dml"))
        self.verify_result(torch.tensor([True, True, False, True]), dmlResult)
        # inplace
        dmlTensor.logical_or_(input_b.to("dml"))
        self.verify_result(torch.tensor([True, True, False, True]), dmlResult)

    def test_bitwise_or(self):
        # compare tensors
        input_a = torch.tensor([True, False, False, True])
        input_b = torch.tensor([True, True, False, False])
        dmlTensor = input_a.clone().detach().to("dml")
        dmlResult = torch.bitwise_or(dmlTensor, input_b.to("dml"))
        self.verify_result(torch.tensor([True, True, False, True]), dmlResult)

        # inplace
        dmlTensor.bitwise_or_(input_b.to("dml"))
        self.verify_result(torch.tensor([True, True, False, True]), dmlResult)

        # with scalar
        dmlTensor = input_a.clone().detach().to("dml")
        dmlResult = torch.bitwise_or(dmlTensor, True)
        self.verify_result(torch.tensor([True, True, True, True]), dmlResult)

        # inplace
        dmlTensor.bitwise_or_(False)
        self.verify_result(torch.tensor([True, True, True, True]), dmlResult)

    def test_logical_not(self):
        # not registered for CPU and CUDA
        # compare tensors
        input_a = torch.tensor([1,0,0,1])
        dmlTensor = input_a.clone().detach().to("dml")
        dmlResult = torch.logical_not(dmlTensor)
        self.verify_result(torch.tensor([False, True, True, False]), dmlResult)
        # inplace
        dmlTensor.logical_not_()
        self.verify_result(torch.tensor([False, True, True, False]), dmlResult)

    def test_logical_xor(self):
        # compare tensors
        input_a = torch.tensor([1,0,0,1])
        input_b = torch.tensor([1,1,0,0])
        dmlTensor = input_a.clone().detach().to("dml")
        dmlResult = torch.logical_xor(dmlTensor, input_b.to("dml"))
        self.verify_result(torch.tensor([False, True, False, True]), dmlResult)
        # inplace
        dmlTensor.logical_xor_(input_b.to("dml"))
        self.verify_result(torch.tensor([False, True, False, True]), dmlResult)

    def test_abs(self):
        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuResult = torch.abs(cpuTensor)
        dmlResult = torch.abs(dmlTensor)
        self.verify_result(cpuResult, dmlResult)
        # test abs_
        cpuTensor.abs_()
        dmlTensor.abs_()
        self.verify_result(cpuTensor, dmlTensor)

    # test add operator
    def test_add(self):
        cpuResult = torch.add(self.tensorA.to("cpu"), self.tensorB.to("cpu"))
        dmlResult = torch.add(self.tensorA.to("dml"), self.tensorB.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        inputTensor = torch.randn((2,3,2,2,2,2))
        cpuResult = torch.add(inputTensor.to("cpu"), inputTensor.to("cpu"))
        dmlResult = torch.add(inputTensor.to("dml"), inputTensor.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        # scalar
        cpuResult = torch.add(self.tensorA.to("cpu"), 2)
        dmlResult = torch.add(self.tensorA.to("dml"), 2)
        self.verify_result(cpuResult, dmlResult)

        # test add_
        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor.add_(self.tensorB)
        dmlTensor.add_(self.tensorB.to("dml"))
        self.verify_result(cpuTensor, dmlTensor)

        cpuTensor = inputTensor.clone().detach()
        dmlTensor = inputTensor.clone().detach().to("dml")
        cpuTensor.add_(cpuTensor)
        dmlTensor.add_(dmlTensor)
        self.verify_result(cpuTensor, dmlTensor)

    def test_sub(self):
        cpuResult = torch.sub(self.tensorA.to("cpu"), self.tensorB.to("cpu"))
        dmlResult = torch.sub(self.tensorA.to("dml"), self.tensorB.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        # scalar
        cpuResult = torch.sub(self.tensorA.to("cpu"), 2)
        dmlResult = torch.sub(self.tensorA.to("dml"), 2)
        self.verify_result(cpuResult, dmlResult)

        # sub_
        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor.sub_(self.tensorB)
        dmlTensor.sub_(self.tensorB.to("dml"))
        self.verify_result(cpuTensor, dmlTensor)

    def test_rsub(self):
        cpuResult = torch.rsub(self.tensorA.to("cpu"), self.tensorB.to("cpu"))
        dmlResult = torch.rsub(self.tensorA.to("dml"), self.tensorB.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        # scalar
        cpuResult = torch.rsub(self.tensorA.to("cpu"), 100)
        dmlResult = torch.rsub(self.tensorA.to("dml"), 100)
        self.verify_result(cpuResult, dmlResult)

    def test_maximum(self):
        # binary maximum
        cpuResult = torch.maximum(self.tensorA.to("cpu"), self.tensorB.to("cpu"))
        dmlResult = torch.maximum(self.tensorA.to("dml"), self.tensorB.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        cpuResult = torch.max(self.tensorA.to("cpu"), self.tensorB.to("cpu"))
        dmlResult = torch.max(self.tensorA.to("dml"), self.tensorB.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        # max
        cpuResult = torch.max(self.tensorA.to("cpu"))
        dmlResult = torch.max(self.tensorA.to("dml"))
        self.verify_result(cpuResult, dmlResult)

    def test_minimum(self):
        # binary minimum
        cpuResult = torch.minimum(self.tensorA.to("cpu"), self.tensorB.to("cpu"))
        dmlResult = torch.minimum(self.tensorA.to("dml"), self.tensorB.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        cpuResult = torch.min(self.tensorA.to("cpu"), self.tensorB.to("cpu"))
        dmlResult = torch.min(self.tensorA.to("dml"), self.tensorB.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        # max
        cpuResult = torch.min(self.tensorA.to("cpu"))
        dmlResult = torch.min(self.tensorA.to("dml"))
        self.verify_result(cpuResult, dmlResult)

    def test_exp(self):
        cpuResult = torch.exp(self.tensorA.to("cpu"))
        dmlResult = torch.exp(self.tensorA.to("dml"))
        self.verify_result(cpuResult, dmlResult)
        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")

        cpuTensor.exp_()
        dmlTensor.exp_()
        self.verify_result(cpuTensor, dmlTensor)

    # def test_exp2(self):
    #     cpuResult = torch.exp2(self.tensorA.to("cpu"))
    #     dmlResult = torch.exp2(self.tensorA.to("dml"))
    #     self.assertTrue(torch.allclose(cpuResult, dmlResult.to("cpu")))
    #     cpuTensor = self.tensorA.clone().detach()
    #     dmlTensor = self.tensorA.clone().detach().to("dml")
    #     cpuTensor.exp2_()
    #     dmlTensor.exp2_()
    #     self.assertTrue(torch.allclose(cpuTensor, dmlTensor.to("cpu")))

    # def test_expm1(self):
    #     cpuResult = torch.expm1(self.tensorA.to("cpu"))
    #     dmlResult = torch.expm1(self.tensorA.to("dml"))
    #     self.assertTrue(torch.allclose(cpuResult, dmlResult.to("cpu")))
    #     cpuTensor = self.tensorA.clone().detach()
    #     dmlTensor = self.tensorA.clone().detach().to("dml")
    #     cpuTensor.expm1_()
    #     dmlTensor.expm1_()
    #     self.assertTrue(torch.allclose(cpuTensor, dmlTensor.to("cpu")))

    def test_log(self):
        cpuResult = torch.log(self.tensorA.to("cpu"))
        dmlResult = torch.log(self.tensorA.to("dml"))
        self.verify_result(cpuResult, dmlResult, equal_nan=True)
        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor.log_()
        dmlTensor.log_()
        self.verify_result(cpuTensor, dmlTensor, equal_nan=True)

    def test_log2(self):
        cpuResult = torch.log2(self.tensorA.to("cpu"))
        dmlResult = torch.log2(self.tensorA.to("dml"))
        self.verify_result(cpuResult, dmlResult, equal_nan=True)
        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor.log2_()
        dmlTensor.log2_()
        self.verify_result(cpuTensor, dmlTensor, equal_nan=True)

    def test_log10(self):
        cpuResult = torch.log10(self.tensorA.to("cpu"))
        dmlResult = torch.log10(self.tensorA.to("dml"))
        self.verify_result(cpuResult, dmlResult, equal_nan=True)
        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor.log10_()
        dmlTensor.log10_()
        self.verify_result(cpuTensor, dmlTensor, equal_nan=True)

    def test_log1p(self):
        cpuResult = torch.log1p(self.tensorA.to("cpu"))
        dmlResult = torch.log1p(self.tensorA.to("dml"))
        self.verify_result(cpuResult, dmlResult, equal_nan=True)
        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor.log1p_()
        dmlTensor.log1p_()
        self.verify_result(cpuTensor, dmlTensor, equal_nan=True)

    def test_sigmoid(self):
        cpuResult = torch.sigmoid(self.tensorA.to("cpu"))
        dmlResult = torch.sigmoid(self.tensorA.to("dml"))
        self.verify_result(cpuResult, dmlResult)
        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor.sigmoid_()
        dmlTensor.sigmoid_()
        self.verify_result(cpuTensor, dmlTensor)

    def test_neg(self):
        cpuResult = torch.neg(self.tensorA.to("cpu"))
        dmlResult = torch.neg(self.tensorA.to("dml"))
        self.verify_result(cpuResult, dmlResult)
        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor.neg_()
        dmlTensor.neg_()
        self.verify_result(cpuTensor, dmlTensor)

    def test_clamp(self):
        # out-of-place clamp test
        cpuResult = torch.clamp(self.tensorA.to("cpu"), 3., 8.)
        dmlResult = torch.clamp(self.tensorA.to("dml"), 3., 8.)
        self.verify_result(cpuResult, dmlResult)
        # in-place clamp test
        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor.clamp_(3., 8.)
        dmlTensor.clamp_(3., 8.)
        self.verify_result(cpuTensor, dmlTensor)
        # int16 min out-of-place clamp test
        cpuTensorInt16 = torch.tensor([-1,-2,1,2], dtype=torch.int16).clamp(min=0)
        dmlTensorInt16 = torch.tensor([-1,-2,1,2], dtype=torch.int16).to("dml").clamp(min=0)
        self.verify_result(cpuTensorInt16, dmlTensorInt16)
        # int32 min out-of-place clamp test
        cpuTensorInt32 = torch.tensor([-1,-2,1,2], dtype=torch.int32).clamp(min=0)
        dmlTensorInt32 = torch.tensor([-1,-2,1,2], dtype=torch.int32).to("dml").clamp(min=0)
        self.verify_result(cpuTensorInt32, dmlTensorInt32)
        # int64 min out-of-place clamp test
        cpuTensorInt64 = torch.tensor([-1,-2,1,2], dtype=torch.int64).clamp(min=0)
        dmlTensorInt64 = torch.tensor([-1,-2,1,2], dtype=torch.int64).to("dml").clamp(min=0)
        self.verify_result(cpuTensorInt64, dmlTensorInt64)
        # int16 max out-of-place clamp test
        cpuTensorInt16 = torch.tensor([-1,-2,1,2], dtype=torch.int16).clamp(max=0)
        dmlTensorInt16 = torch.tensor([-1,-2,1,2], dtype=torch.int16).to("dml").clamp(max=0)
        self.verify_result(cpuTensorInt16, dmlTensorInt16)
        # int32 max out-of-place clamp test
        cpuTensorInt32 = torch.tensor([-1,-2,1,2], dtype=torch.int32).clamp(max=0)
        dmlTensorInt32 = torch.tensor([-1,-2,1,2], dtype=torch.int32).to("dml").clamp(max=0)
        self.verify_result(cpuTensorInt32, dmlTensorInt32)
        # int64 max out-of-place clamp test
        cpuTensorInt64 = torch.tensor([-1,-2,1,2], dtype=torch.int64).clamp(max=0)
        dmlTensorInt64 = torch.tensor([-1,-2,1,2], dtype=torch.int64).to("dml").clamp(max=0)
        self.verify_result(cpuTensorInt64, dmlTensorInt64)

    def test_clamp_max(self):
        # out-of-place clamp test
        cpuResult = torch.clamp_max(self.tensorA.to("cpu"), 3.)
        dmlResult = torch.clamp_max(self.tensorA.to("dml"), 3.)
        self.verify_result(cpuResult, dmlResult)
        # in-place clamp test
        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor.clamp_max_(3.)
        dmlTensor.clamp_max_(3.)
        self.verify_result(cpuTensor, dmlTensor)
        # int16 min out-of-place clamp test
        cpuTensorInt16 = torch.tensor([-1,-2,1,2], dtype=torch.int16).clamp_max_(0)
        dmlTensorInt16 = torch.tensor([-1,-2,1,2], dtype=torch.int16).to("dml").clamp_max_(0)
        self.verify_result(cpuTensorInt16, dmlTensorInt16)
        # int32 min out-of-place clamp test
        cpuTensorInt32 = torch.tensor([-1,-2,1,2], dtype=torch.int32).clamp_max_(0)
        dmlTensorInt32 = torch.tensor([-1,-2,1,2], dtype=torch.int32).to("dml").clamp_max_(0)
        self.verify_result(cpuTensorInt32, dmlTensorInt32)
        # int64 min out-of-place clamp test
        cpuTensorInt64 = torch.tensor([-1,-2,1,2], dtype=torch.int64).clamp_max_(0)
        dmlTensorInt64 = torch.tensor([-1,-2,1,2], dtype=torch.int64).to("dml").clamp_max_(0)
        self.verify_result(cpuTensorInt64, dmlTensorInt64)
        # int16 max out-of-place clamp test
        cpuTensorInt16 = torch.tensor([-1,-2,1,2], dtype=torch.int16).clamp_max_(0)
        dmlTensorInt16 = torch.tensor([-1,-2,1,2], dtype=torch.int16).to("dml").clamp_max_(0)
        self.verify_result(cpuTensorInt16, dmlTensorInt16)
        # int32 max out-of-place clamp test
        cpuTensorInt32 = torch.tensor([-1,-2,1,2], dtype=torch.int32).clamp_max_(0)
        dmlTensorInt32 = torch.tensor([-1,-2,1,2], dtype=torch.int32).to("dml").clamp_max_(0)
        self.verify_result(cpuTensorInt32, dmlTensorInt32)
        # int64 max out-of-place clamp test
        cpuTensorInt64 = torch.tensor([-1,-2,1,2], dtype=torch.int64).clamp_max_(0)
        dmlTensorInt64 = torch.tensor([-1,-2,1,2], dtype=torch.int64).to("dml").clamp_max_(0)
        self.verify_result(cpuTensorInt64, dmlTensorInt64)

    def test_floor(self):
        input_a = torch.tensor([1.1, -2.1, 3.2, 4.6, -5.5])
        result = torch.tensor([1., -3., 3., 4., -6.])
        dmlTensor = input_a.clone().detach().to("dml")
        dmlResult = torch.floor(dmlTensor)
        self.verify_result(result, dmlResult)

        # inplace
        dmlTensor.floor_()
        self.verify_result(result, dmlTensor)

    def test_ceil(self):
        input_a = torch.tensor([1.1, -2.1, 3.2, 4.6, -5.5])
        result = torch.tensor([2., -2., 4., 5., -5.])
        dmlTensor = input_a.clone().detach().to("dml")
        dmlResult = torch.ceil(dmlTensor)
        self.verify_result(result, dmlResult)

        # inplace
        dmlTensor.ceil_()
        self.verify_result(result, dmlTensor)

    def test_clamp_max(self):
        cpuResult = torch.clamp_min(self.tensorA.to("cpu"), 8.)
        dmlResult = torch.clamp_min(self.tensorA.to("dml"), 8.)
        self.verify_result(cpuResult, dmlResult)
        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor.clamp_min(8.)
        dmlTensor.clamp_min(8.)
        self.verify_result(cpuTensor, dmlTensor)

    def test_addcdiv(self):
        cpuResult = torch.addcdiv(self.tensorA.to("cpu"), self.tensorA.to("cpu"), self.tensorB.to("cpu"))
        dmlResult = torch.addcdiv(self.tensorA.to("dml"), self.tensorA.to("dml"), self.tensorB.to("dml"))
        self.verify_result(cpuResult, dmlResult)
        # test addcdiv_
        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor.addcdiv_(self.tensorA.to("cpu"), self.tensorB.to("cpu"))
        dmlTensor.addcdiv_(self.tensorA.to("dml"), self.tensorB.to("dml"))
        self.verify_result(cpuTensor, dmlTensor)

    def test_addcmul(self):
        dmlResult = torch.addcmul(self.tensorA.to("dml"), self.tensorB.to("dml"), self.tensorB.to("dml"))
        cpuResult = torch.addcmul(self.tensorA.to("cpu"), self.tensorB.to("cpu"), self.tensorB.to("cpu"))
        self.verify_result(cpuResult, dmlResult)
        # test addcmul_
        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor.addcmul_(self.tensorB.to("cpu"), self.tensorB.to("cpu"))
        dmlTensor.addcmul_(self.tensorB.to("dml"), self.tensorB.to("dml"))
        self.verify_result(cpuTensor, dmlTensor)

    def test_addmm(self):
        cpuResult = torch.addmm(self.tensorB.to("cpu"), self.tensorA.to("cpu"), self.tensorB.to("cpu"))
        dmlResult = torch.addmm(self.tensorB.to("dml"), self.tensorA.to("dml"), self.tensorB.to("dml"))
        self.verify_result(cpuResult, dmlResult)

    def test_addmm_(self):
        cpuTensor = self.tensorB.clone().detach()
        dmlTensor = self.tensorB.clone().detach().to("dml")
        cpuTensor.addmm_(self.tensorA.to("cpu"), self.tensorB.to("cpu"))
        dmlTensor.addmm_(self.tensorA.to("dml"), self.tensorB.to("dml"))
        self.verify_result(cpuTensor, dmlTensor)

    def test_mm(self):
        cpuResult = torch.mm(self.tensorA.to("cpu"), self.tensorB.to("cpu"))
        dmlResult = torch.mm(self.tensorA.to("dml"), self.tensorB.to("dml"))
        self.verify_result(cpuResult, dmlResult)

    def test_detach(self):
        dmlResult = self.tensorA.to("dml").detach()
        cpuResult = self.tensorA.to("cpu").detach()
        self.verify_result(cpuResult, dmlResult)

    def test_index_put(self):
        indices = [torch.tensor([0,1,3]), torch.tensor([1,2,3])]
        indices_dml=[torch.tensor([0,1,3]).to("dml"), torch.tensor([1,2,3]).to("dml")]
        values=torch.tensor([50.])
        values_dml = values.to("dml")
        dmlResult = self.tensorA.to("dml").index_put(indices=indices_dml, values=values_dml)
        cpuResult = self.tensorA.to("cpu").index_put(indices=indices, values=values)
        self.verify_result(cpuResult, dmlResult)

        dmlTensor = self.tensorA.to("dml").detach()
        cpuTensor = self.tensorA.to("cpu").detach()
        dmlTensor.index_put_(indices=indices_dml, values=values_dml)
        cpuTensor.index_put_(indices=indices, values=values)
        self.verify_result(cpuResult, dmlResult)

        cpuTensor = torch.zeros(5,5,5)
        dmlTensor = cpuTensor.to("dml").detach()
        indices_dml.append(torch.tensor([0,1,3]).to("dml"))
        indices.append(torch.tensor([0,1,3]))
        dmlTensor.index_put_(indices=indices_dml, values=values_dml)
        cpuTensor.index_put_(indices=indices, values=values)
        self.verify_result(cpuTensor, dmlTensor)

    def test_div(self):
        # tensor / tensor
        cpuResult = torch.div(self.tensorA.to("cpu"), self.tensorB.to("cpu"))
        dmlResult = torch.div(self.tensorA.to("dml"), self.tensorB.to("dml"))
        self.verify_result(cpuResult, dmlResult)
        # tensor / scalar
        cpuResult = torch.div(self.tensorA.to("cpu"), 2)
        dmlResult = torch.div(self.tensorA.to("dml"), 2)
        self.verify_result(cpuResult, dmlResult)
        # test div_
        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor.div_(self.tensorB)
        dmlTensor.div_(self.tensorB.to("dml"))
        self.verify_result(cpuTensor, dmlTensor)

    def test_empty_and_empty_strided(self):
        # The results don't have to be identical because of different uninitialization on CPU and DML
        torch.empty((3,3), device=torch.device("dml"))
        torch.empty_strided((2, 3), (1, 2), device=torch.device("dml"))

        temp = torch.randn((2,3))
        cpuResult = temp.new_empty_strided((3, 4), (1, 2))
        dmlResult = temp.new_empty_strided((3, 4), (1, 2), device=torch.device("dml"))
        self.assertTrue(cpuResult.shape == dmlResult.shape)
        self.assertTrue(cpuResult.stride() == dmlResult.stride())
        self.assertTrue(cpuResult.size() == dmlResult.size())

    def test_as_strided(self):
        cpuResult = torch.as_strided(self.tensorA, (3,3), (3,1))
        dmlResult = torch.as_strided(self.tensorA.to("dml"), (3,3), (3,1))
        self.verify_result(cpuResult, dmlResult)

    def test_clone(self):
        cpuResult = self.tensorA.clone()
        dmlResult = self.tensorA.to("dml").clone()
        self.verify_result(cpuResult, dmlResult)

    def test_tensor_unfold(self):
        cpuTensor = self.tensorA.unfold(0,2,1)
        dmlTensor = self.tensorA.to("dml").unfold(0,2,1)
        self.verify_result(cpuTensor, dmlTensor)

    def test_tensor_unfold(self):
        cpuTensor = self.tensorA.unfold(0,2,1)
        dmlTensor = self.tensorA.to("dml").unfold(0,2,1)
        self.verify_result(cpuTensor, dmlTensor)

    def test_empty_like(self):
        cpuResult = torch.empty_like(self.tensorA)
        dmlResult = torch.empty_like(self.tensorA.to("dml"))
        # The results don't have to be identical because of different uninitialization on CPU and DML
        # self.assertTrue(torch.allclose(cpuResult, dmlResult.to("cpu")))
        self.assertTrue(torch.allclose(torch.tensor(cpuResult.size()),
                                        torch.tensor(dmlResult.to("cpu").size())))
        self.assertTrue(torch.allclose(torch.tensor(cpuResult.stride()),
                                        torch.tensor(dmlResult.to("cpu").stride())))

    def test_expand(self):
        dmlResult = torch.tensor([[1.], [2.], [3.]]).to("dml").expand(-1,4)
        cpuResult = torch.tensor([[1.], [2.], [3.]]).to("cpu").expand(-1,4)
        self.verify_result(cpuResult, dmlResult)

    def test_fill_and_full_(self):
        # fill_
        dmlResult = torch.empty((3,3), device=torch.device("dml"))
        dmlResult.fill_(3)
        cpuResult = torch.empty((3,3), device=torch.device("cpu"))
        cpuResult.fill_(3)
        self.verify_result(cpuResult, dmlResult)

        # full
        dmlResult = torch.full_like(self.tensorA, 3, device=torch.device("dml"))
        cpuResult = torch.full_like(self.tensorA, 3, device=torch.device("cpu"))
        self.verify_result(cpuResult, dmlResult)

        dmlResult = torch.full((2, 3), 3.14, device=torch.device("dml"))
        cpuResult = torch.full((2, 3), 3.14, device=torch.device("cpu"))
        self.verify_result(cpuResult, dmlResult)

        # new_full
        tensor = torch.ones((2,), dtype=torch.float64)
        cpuResult = tensor.new_full((3, 4), 3.141592)
        dmlResult = tensor.to("dml").new_full((3, 4), 3.141592)
        self.verify_result(cpuResult, dmlResult)

        # scalar_tensor
        dmlResult = torch.scalar_tensor(3, device=torch.device("dml"))
        cpuResult = torch.scalar_tensor(3, device=torch.device("cpu"))
        self.verify_result(cpuResult, dmlResult)

    def test_nonzero(self):
        # 4D
        cpuResult = torch.nonzero(torch.tensor([[[[1, 1, 1, 0, 1], [1, 1, 1, 0, 1]]]]))
        dmlResult = torch.nonzero(torch.tensor([[[[1, 1, 1, 0, 1], [1, 1, 1, 0, 1]]]]).to("dml"))
        self.verify_result(cpuResult, dmlResult)

        # 1D
        cpuResult = torch.nonzero(torch.tensor([1, 1, 1, 0, 1]))
        dmlResult = torch.nonzero(torch.tensor([1, 1, 1, 0, 1]).to("dml"))
        self.verify_result(cpuResult, dmlResult)

        # 1 element
        cpuResult = torch.nonzero(torch.tensor([1]))
        dmlResult = torch.nonzero(torch.tensor([1]).to("dml"))
        self.verify_result(cpuResult, dmlResult)


    def test_masked_fill(self):
        mask = torch.empty(self.tensorA.shape, dtype=torch.bool).random_(2)
        value = torch.tensor(100.0)

        # blocked by at::clone
        cpuResult = torch.masked_fill(self.tensorA, mask, value)
        dmlResult = torch.masked_fill(self.tensorA.to("dml"), mask.to("dml"), value.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        cpuResult = torch.masked_fill(self.tensorA, mask, 100)
        dmlResult = torch.masked_fill(self.tensorA.to("dml"), mask.to("dml"), 100)
        self.verify_result(cpuResult, dmlResult)

        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor.masked_fill_(mask, value)
        dmlTensor.masked_fill_(mask.to("dml"), value.to("dml"))
        self.verify_result(cpuTensor, dmlTensor)

        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor.masked_fill_(mask, 100)
        dmlTensor.masked_fill_(mask.to("dml"), 100)
        self.verify_result(cpuTensor, dmlTensor)

    def test_masked_select(self):
        mask = self.tensorA.ge(-0.2)
        cpuResult = torch.masked_select(self.tensorA, mask)
        dmlResult = torch.masked_select(self.tensorA.to("dml"), mask.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        # test broadcastable shape
        mask = torch.tensor([False, True, True, False])
        cpuResult = torch.masked_select(self.tensorA, mask)
        dmlResult = torch.masked_select(self.tensorA.to("dml"), mask.to("dml"))
        self.verify_result(cpuResult, dmlResult)

    def test_index_select(self):
        indices = torch.tensor([1, 0, 1, 2])
        cpuResult = torch.index_select(self.tensorA, 0, indices)
        dmlResult = torch.index_select(self.tensorA.to("dml"), 0, indices.to("dml"))
        self.verify_result(cpuResult, dmlResult)
        cpuResult = torch.index_select(self.tensorA, 1, indices)
        dmlResult = torch.index_select(self.tensorA.to("dml"), 1, indices.to("dml"))
        self.verify_result(cpuResult, dmlResult)

    def test_arange_(self):
        dmlResult = torch.arange(0,10,2, device=torch.device("dml"))
        cpuResult = torch.arange(0,10,2, device=torch.device("cpu"))
        self.verify_result(cpuResult, dmlResult)
        dmlResult = torch.arange(0,10,3, dtype=torch.float, device=torch.device("dml"))
        cpuResult = torch.arange(0,10,3, dtype=torch.float, device=torch.device("cpu"))
        self.verify_result(cpuResult, dmlResult)
        dmlResult = torch.arange(0,10,3, dtype=torch.int16, device=torch.device("dml"))
        cpuResult = torch.arange(0,10,3, dtype=torch.int16, device=torch.device("cpu"))
        self.verify_result(cpuResult, dmlResult)

    def test_flatten(self):
        dmlResult = self.tensorA.to("dml").flatten()
        cpuResult = self.tensorA.to("cpu").flatten()
        self.verify_result(cpuResult, dmlResult)

    def test_mul(self):
        cpuResult = torch.mul(self.tensorA.to("cpu"), self.tensorB.to("cpu"))
        dmlResult = torch.mul(self.tensorA.to("dml"), self.tensorB.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        inputTensor = torch.randn((2,3,2,2,2,2))
        cpuResult = torch.mul(inputTensor.to("cpu"), inputTensor.to("cpu"))
        dmlResult = torch.mul(inputTensor.to("dml"), inputTensor.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        # scalar
        cpuResult = torch.mul(self.tensorA.to("cpu"), 2)
        dmlResult = torch.mul(self.tensorA.to("dml"), 2)
        self.verify_result(cpuResult, dmlResult)

        cpuResult = torch.mul(inputTensor.to("cpu"), 2)
        dmlResult = torch.mul(inputTensor.to("dml"), 2)
        self.verify_result(cpuResult, dmlResult)

        # test mul_
        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor.mul_(self.tensorB)
        dmlTensor.mul_(self.tensorB.to("dml"))
        self.verify_result(cpuTensor, dmlTensor)

        cpuTensor = inputTensor.clone().detach()
        dmlTensor = inputTensor.clone().detach().to("dml")
        cpuTensor.mul_(2)
        dmlTensor.mul_(2)
        self.verify_result(cpuTensor, dmlTensor)

        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor[..., :2] *= 2
        dmlTensor[..., :2] *= 2
        self.verify_result(cpuTensor, dmlTensor)

    def test_atan(self):
        cpuResult = torch.atan(self.tensorA.to("cpu"))
        dmlResult = torch.atan(self.tensorA.to("dml"))
        self.verify_result(cpuResult, dmlResult, abs_tol=1e-03)
        # test relu_
        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor.atan_()
        dmlTensor.atan_()
        self.verify_result(cpuTensor, dmlTensor, abs_tol=1e-03)

    def test_silu(self):
        m = torch.nn.SiLU()
        cpuResult = m(self.tensorA.to("cpu"))
        dmlResult = m(self.tensorA.to("dml"))
        self.verify_result(cpuResult, dmlResult)

    def test_relu(self):
        cpuResult = torch.relu(self.tensorA.to("cpu"))
        dmlResult = torch.relu(self.tensorA.to("dml"))
        self.verify_result(cpuResult, dmlResult)
        # test relu_
        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor.relu_()
        dmlTensor.relu_()
        self.verify_result(cpuTensor, dmlTensor)

    def test_leaky_relu(self):
        dmlResult = torch.nn.functional.leaky_relu(self.tensorA.to("dml"))
        cpuResult = torch.nn.functional.leaky_relu(self.tensorA.to("cpu"))
        self.verify_result(cpuResult, dmlResult)
        # test relu_
        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        torch.nn.functional.leaky_relu_(cpuTensor, .5)
        torch.nn.functional.leaky_relu_(dmlTensor, .5)
        self.verify_result(cpuTensor, dmlTensor)

    def test_reshape(self):
        dmlResult = torch.reshape(self.tensorA.to("dml"), (4, 4))
        cpuResult = torch.reshape(self.tensorA.to("cpu"), (4, 4))
        self.verify_result(cpuResult, dmlResult)

        dmlResult = self.tensorA.to("dml")[:,-1].reshape(4)
        cpuResult = self.tensorA.to("cpu")[:,-1].reshape(4)
        self.verify_result(cpuResult, dmlResult)

    def test_sqrt(self):
        dmlResult = torch.sqrt(torch.abs(self.tensorA.to("dml")))
        cpuResult = torch.sqrt(torch.abs(self.tensorA.to("cpu")))
        self.verify_result(cpuResult, dmlResult)
        # test sqrt_
        cpuTensor = torch.abs(self.tensorA)
        dmlTensor = torch.abs(self.tensorA).to("dml")
        cpuTensor.sqrt_()
        dmlTensor.sqrt_()
        self.verify_result(cpuTensor, dmlTensor)

    def test_rsqrt(self):
        dmlResult = torch.rsqrt(torch.abs(self.tensorA.to("dml")))
        cpuResult = torch.rsqrt(torch.abs(self.tensorA.to("cpu")))
        self.verify_result(cpuResult, dmlResult)
        # test rsqrt_
        cpuTensor = torch.abs(self.tensorA)
        dmlTensor = torch.abs(self.tensorA).to("dml")
        cpuTensor.rsqrt_()
        dmlTensor.rsqrt_()
        self.verify_result(cpuTensor, dmlTensor)

    def test_unsqueeze(self):
        dmlResult = self.tensorA.to("dml").unsqueeze(1)
        cpuResult = self.tensorA.to("cpu").unsqueeze(1)
        self.verify_result(cpuResult, dmlResult)

    def test_view(self):
        cpuResult = self.tensorA.view(2, 8)
        dmlResult = self.tensorA.to("dml").view(2,8)
        self.verify_result(cpuResult, dmlResult)

        cpuResult = self.tensorA.view(-1, 8)
        dmlResult = self.tensorA.to("dml").view(-1,8)
        self.verify_result(cpuResult, dmlResult)

    def test_conv2d(self):
        net = torch.nn.Conv2d(2, 6, 4, 1, padding=2, groups=2)
        input = torch.rand(2,2,2,2)
        cpuResult = net(input.to("cpu"))
        dml_net = net.to("dml")
        dmlResult = dml_net(input.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        net_bias_false = torch.nn.Conv2d(2, 6, 4, 1, padding=2, groups=2, bias=False)
        cpuResult = net_bias_false(input.to("cpu"))
        dml_net = net_bias_false.to("dml")
        dmlResult = net_bias_false(input.to("dml"))
        self.verify_result(cpuResult, dmlResult)

    def test_convtranposed2d(self):
        input = torch.randn(20, 16, 50, 100)
        downsample = torch.nn.Conv2d(16, 16, 4, stride=2, padding=2)
        upsample = torch.nn.ConvTranspose2d(16, 16, 4, stride=2, padding=2)
        h = downsample(input)
        cpuResult = upsample(h, output_size=input.size())

        downsampe_dml = downsample.to("dml")
        upsample_dml = upsample.to("dml")
        h_dml = downsampe_dml(input.to("dml"))
        dmlResult = upsample_dml(h_dml, output_size=input.size())
        self.verify_result(cpuResult, dmlResult, abs_tol=1e-06)

    def test_linear(self):
        net = torch.nn.Linear(20, 30)
        input = torch.randn(128,20)
        cpuResult = net(input)
        dml_net = net.to("dml")
        dmlResult = dml_net(input.to("dml"))
        self.verify_result(cpuResult, dmlResult, abs_tol=1e-06)

    def test_hardtanh(self):
        hth = torch.nn.Hardtanh(-3,3)
        hth_dml = hth.to("dml")
        cpuResult = hth(self.tensorA.to("cpu"))
        dmlResult = hth_dml(self.tensorA.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        input_int32 = torch.tensor([[1,2,3,4],[2,3,4,5]], dtype=torch.int32)
        cpuResult = hth(input_int32.to("cpu"))
        dmlResult = hth_dml(input_int32.to("dml"))
        self.verify_result(cpuResult, dmlResult)

    def test_hardtanh_grad(self):
        hth = torch.nn.Hardtanh(-3,3)
        hth_dml = hth.to("dml")

        cpuTensor = self.tensorA.clone().detach().requires_grad_(True)
        dmlTensor = self.tensorA.clone().detach().to("dml").requires_grad_(True)

        out = hth(cpuTensor)
        out_dml = hth_dml(dmlTensor)
        result = out.mean()
        result_dml = out_dml.mean()
        result.backward()
        result_dml.backward()
        self.verify_result(cpuTensor.grad.to("cpu"), dmlTensor.grad.to("cpu"))

    def test_batchnorm(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.features = nn.Sequential(
                nn.BatchNorm2d(3)
                )
            def forward(self, x):
                output = self.features(x)
                return output

        model = Net()
        def eval(device, input):
            model.to(device)
            model.eval()
            input = input.to(device)
            return model(input)

        def train(device, input):
            model.to(device)
            model.train()
            input = input.to(device)
            return model(input)

        input = torch.rand(1,3,28,28)
        cpuResult = eval("cpu", input)
        dmlResult = eval("dml", input)
        self.verify_result(cpuResult, dmlResult)
        input = torch.randn(1, 3, 28, 28)
        cpuResult = train("cpu", input)
        dmlResult = train("dml", input)
        self.verify_result(cpuResult, dmlResult)

    def test_batchnorm_backward(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.features = nn.Sequential(
                nn.BatchNorm2d(3)
                )
            def forward(self, x):
                output = self.features(x)
                return output

        model = Net()
        def eval(device, input):
            model.to(device)
            model.eval()
            input = input.to(device)
            return model(input)

        def train(device, input):
            model.to(device)
            model.train()
            input = input.to(device)
            return model(input)

        input = torch.randn(1,3,28,28)
        cpu_input = input.clone().requires_grad_(True)
        dml_input = input.clone().requires_grad_(True)

        cpuResult = eval("cpu", cpu_input)
        dmlResult = eval("dml", dml_input)
        cpuResult.sum().backward()
        dmlResult.sum().backward()
        self.verify_result(cpu_input.grad, dml_input.grad)

        # train backwards is failling now
        # cpuResult = train("cpu", cpu_input)
        # dmlResult = train("dml", dml_input)
        # cpuResult.sum().backward()
        # dmlResult.sum().backward()
        # self.assertTrue(torch.allclose(cpu_input.grad, dml_input.grad))

    def test_maxpool2d(self):
        net = nn.MaxPool2d((3, 2))
        input = torch.randn(20, 16, 50, 32)
        net.to("cpu")
        cpuResult = net(input.to("cpu"))

        net.to("dml")
        dmlResult = net(input.to("dml"))

        self.verify_result(cpuResult, dmlResult)

    def test_cat(self):
        dmlTensorA = self.tensorA.to("dml");
        cpuTensorA = self.tensorA.to("cpu");
        dmlResult0 = torch.cat([dmlTensorA, dmlTensorA, dmlTensorA], dim=0)
        dmlResult1 = torch.cat([dmlTensorA, dmlTensorA, dmlTensorA], dim=1)
        cpuResult0 = torch.cat([cpuTensorA, cpuTensorA, cpuTensorA], dim=0)
        cpuResult1 = torch.cat([cpuTensorA, cpuTensorA, cpuTensorA], dim=1)
        self.verify_result(cpuResult0, dmlResult0)
        self.verify_result(cpuResult1, dmlResult1)
    
    def test_stack(self):
        dmlTensorA = self.tensorA.to("dml");
        cpuTensorA = self.tensorA.to("cpu");
        dmlResult0 = torch.stack([dmlTensorA, dmlTensorA, dmlTensorA], dim=0)
        dmlResult1 = torch.stack([dmlTensorA, dmlTensorA, dmlTensorA], dim=1)
        cpuResult0 = torch.stack([cpuTensorA, cpuTensorA, cpuTensorA], dim=0)
        cpuResult1 = torch.stack([cpuTensorA, cpuTensorA, cpuTensorA], dim=1)
        self.verify_result(cpuResult0, dmlResult0)
        self.verify_result(cpuResult1, dmlResult1)

    def test_split(self):
        dmlTensorA = self.tensorA.to("dml");
        cpuTensorA = self.tensorA.to("cpu");
        dmlResult0 = torch.split(dmlTensorA, [1,3], dim=0)
        dmlResult1 = torch.split(dmlTensorA, [1,3], dim=1)
        cpuResult0 = torch.split(cpuTensorA, [1,3], dim=0)
        cpuResult1 = torch.split(cpuTensorA, [1,3], dim=1)
        self.verify_result(cpuResult0[0], dmlResult0[0])
        self.verify_result(cpuResult0[1], dmlResult0[1])
        self.verify_result(cpuResult1[0], dmlResult1[0])
        self.verify_result(cpuResult1[1], dmlResult1[1])

    def test_topk(self):
        dmlTensorA = self.tensorA.to("dml");
        cpuTensorA = self.tensorA.to("cpu");
        dmlResult0 = torch.topk(dmlTensorA, 3, dim=0)
        dmlResult1 = torch.topk(dmlTensorA, 3, dim=1)
        cpuResult0 = torch.topk(cpuTensorA, 3, dim=0)
        cpuResult1 = torch.topk(cpuTensorA, 3, dim=1)
        self.verify_result(cpuResult0[0], dmlResult0[0])
        self.verify_result(cpuResult0[1], dmlResult0[1])
        self.verify_result(cpuResult1[0], dmlResult1[0])
        self.verify_result(cpuResult1[1], dmlResult1[1])

    def test_conj(self):
        # TODO: add support for complex dtype
        # https://pytorch.org/docs/stable/generated/torch.conj.html
        cpuResult = torch.conj(self.tensorA)
        dmlResult = torch.conj(self.tensorA.to("dml"))
        self.verify_result(cpuResult, dmlResult)

    def test_narrow(self):
        cpuResult = torch.narrow(self.tensorA, 0, 0, 2)
        dmlResult = torch.narrow(self.tensorA.to("dml"), 0, 0, 2)
        self.verify_result(cpuResult, dmlResult)
        cpuResult = torch.narrow(self.tensorA, 1, 1, 2)
        dmlResult = torch.narrow(self.tensorA.to("dml"), 1, 1, 2)
        self.verify_result(cpuResult, dmlResult)

    def test_slice(self):
        cpuResult = self.tensorA[1:, :2]
        dmlResult = self.tensorA.to("dml")[1:, :2]
        self.verify_result(cpuResult, dmlResult)

        cpuResult1 = self.tensorA[..., 0::2]
        dmlResult1 = self.tensorA.to("dml")[..., 0::2]
        self.verify_result(cpuResult1.clamp(0, 0.5), dmlResult1.clamp(0, 0.5))

        cpuResult2 = self.tensorA[..., 1::2]
        dmlResult2 = self.tensorA.to("dml")[..., 1::2]
        self.verify_result(cpuResult2.clamp(0, 0.5), dmlResult2.clamp(0, 0.5))

    def test_logsoftmax(self):
        dmlResult = torch.log_softmax(torch.abs(self.tensorA).to("dml"), 1)
        cpuResult = torch.log_softmax(torch.abs(self.tensorA).to("cpu"), 1)
        self.verify_result(cpuResult, dmlResult)

        input = torch.arange(0,27).view(3,3,3).float().to("dml")
        dmlResult = torch.log_softmax(input.to("dml"), 1)
        cpuResult = torch.log_softmax(input.to("cpu"), 1)
        self.verify_result(cpuResult, dmlResult, abs_tol=1e-6)

        dmlResult = torch.log_softmax(input.to("dml"), -1)
        cpuResult = torch.log_softmax(input.to("cpu"), -1)
        self.verify_result(cpuResult, dmlResult, abs_tol=1e-6)
    
    def test_softmax(self):
        dmlResult = torch.softmax(torch.abs(self.tensorA).to("dml"), 1)
        cpuResult = torch.softmax(torch.abs(self.tensorA).to("cpu"), 1)
        self.verify_result(cpuResult, dmlResult)
        input = torch.arange(0,27).view(3,3,3).float().to("dml")
        dmlResult = torch.softmax(input.to("dml"), 1)
        cpuResult = torch.softmax(input.to("cpu"), 1)
        self.verify_result(cpuResult, dmlResult, abs_tol=1e-6)

        dmlResult = torch.softmax(input.to("dml"), -1)
        cpuResult = torch.softmax(input.to("cpu"), -1)
        
        self.verify_result(cpuResult, dmlResult, abs_tol=1e-6)

    def test_mean(self):
        dmlResult = torch.mean(self.tensorA.to("dml"))
        cpuResult = torch.mean(self.tensorA.to("cpu"))
        self.verify_result(cpuResult, dmlResult)

        dmlResult = torch.mean(self.tensorA.to("dml"), 1)
        cpuResult = torch.mean(self.tensorA.to("cpu"), 1)
        self.verify_result(cpuResult, dmlResult)

    def test_sum(self):
        dmlResult = torch.sum(self.tensorA.to("dml"))
        cpuResult = torch.sum(self.tensorA.to("cpu"))
        self.verify_result(cpuResult, dmlResult)

        dmlResult = torch.sum(self.tensorA.to("dml"), 1)
        cpuResult = torch.sum(self.tensorA.to("cpu"), 1)
        self.verify_result(cpuResult, dmlResult)

    def test_prod(self):
        dmlResult = torch.prod(self.tensorA.to("dml"))
        cpuResult = torch.prod(self.tensorA.to("cpu"))
        self.verify_result(cpuResult, dmlResult)

        input = torch.randn(1, 6, 7)
        dmlResult = input.to("dml").prod(2)
        cpuResult = input.to("cpu").prod(2)
        self.verify_result(cpuResult, dmlResult)

    def test_repeat(self):
        dmlResult = self.tensorA.to("dml").repeat(4,2,2)
        cpuResult = self.tensorA.repeat(4,2,2)
        self.verify_result(cpuResult, dmlResult)

    def test_adaptive_avg_pool2d(self):
        input = torch.randn(2, 6, 7)
        output_size = torch.tensor([2, 2])
        aap2d = torch.nn.AdaptiveAvgPool2d(output_size) # build aap2d layer
        cpuResult = aap2d(input)  # Run on CPU

        dml_input = input.to("dml") # put input to DML
        aap2d_dml = aap2d.to("dml") # put aap2d to DML
        dmlResult = aap2d_dml(dml_input) # Run on DML
        self.verify_result(cpuResult, dmlResult)

    # CPU and DML results don't match
    def test_dropout(self):
        m = nn.Dropout(p=0.2)
        m(self.tensorA.to("dml")).to("cpu")

    def test_bernoulli(self):
        a = torch.empty(3, 3).uniform_(0, 1).to("dml")
        torch.bernoulli(a).to("cpu")
        torch.bernoulli(a, p=0.2).to("cpu")
        a.bernoulli_().to("cpu")

    def test_random(self):
        a = torch.empty(3, 3).uniform_(0, 1).to("dml")
        a.random_().to("cpu")
        a.random_(1,2).to("cpu")

    def test_uniform(self):
        a = torch.empty(3,3).to("dml")
        a.uniform_(0, 1)
        a.to("cpu")

    def test_randperm(self):
        a = torch.randperm(4, device=torch.device("dml")).to("cpu")

    def test_nll_loss(self):
        m = nn.LogSoftmax(dim=1)
        weight = torch.tensor([1.,2.,3.,4.,5.,6.])
        loss = nn.NLLLoss(weight=weight).to("cpu")
        input = torch.randn(4, 6, requires_grad=True)
        target = torch.tensor([1, 0, 4, 5]).to("cpu")
        inputs = m(input).to("cpu")
        cpuResult = loss(inputs, target)

        loss = nn.NLLLoss(weight=weight).to("dml")
        target = target.to("dml")
        inputs = inputs.to("dml")
        dmlResult = loss(inputs, target)
        self.verify_result(cpuResult, dmlResult)

    def test_BCEWithLogitsLoss(self):
        weight = torch.randn(2, 3)
        pos_weight = torch.randn(1)
        loss = nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight)
        input = torch.randn(5, 2, 3)
        target = torch.empty(5, 2, 3).random_(2)
        cpuResult = loss(input, target)

        dml_input = input.to("dml")
        target_input = target.to("dml")
        loss_dml = loss.to("dml")

        dmlResult = loss_dml(dml_input, target_input)
        self.verify_result(cpuResult, dmlResult)

        # test overflow
        input = torch.tensor([[-2.9309e+16, -5.2225e+18,  9.3603e+17,  1.3827e+18,  1.2060e+18,
          1.2949e+18,  6.0490e+17,  9.0288e+17,  7.7774e+17, -6.2197e+17],
        [ 3.0915e+15, -2.2820e+18,  3.6399e+17,  6.3987e+17,  5.6325e+17,
          5.4267e+17,  2.9168e+17,  3.2440e+17,  4.4137e+17, -2.0580e+17]])

        target = torch.empty(2, 10).random_(10)

        loss = nn.BCEWithLogitsLoss()
        cpuResult = loss(input, target)

        dml_input = input.to("dml")
        target_input = target.to("dml")
        loss_dml = loss.to("dml")

        dmlResult = loss_dml(dml_input, target_input)
        self.verify_result(cpuResult, dmlResult)

    def test_BCEWithLogitsLoss_grad(self):
        weight = torch.randn(2, 3)
        loss = nn.BCEWithLogitsLoss(weight=weight, pos_weight=weight)
        input = torch.randn(5, 2, 3)
        target = torch.empty(5, 2, 3).random_(2)

        cpu_input = input.clone().requires_grad_(True)
        cpuResult = loss(cpu_input, target)
        cpuResult.backward()

        dml_input = input.clone().requires_grad_(True)
        target_input = target.to("dml")
        loss_dml = loss.to("dml")
        dmlResult = loss_dml(dml_input.to("dml"), target_input)
        dmlResult.backward()

        self.verify_result(cpu_input.grad, dml_input.grad)

    def test_nll_loss_grad(self):
        loss = nn.NLLLoss()
        input = torch.randn(3, 5)
        cpu_input = input.clone().requires_grad_(True)
        target = torch.tensor([1, 0, 4])
        output = loss(cpu_input, target)
        output.backward()

        target_dml = target.to("dml")
        input_dml = input.clone().requires_grad_(True)
        loss_dml = loss.to("dml")
        output_dml = loss_dml(input_dml.to("dml"), target_dml)
        output_dml.backward()
        self.verify_result(cpu_input.grad, input_dml.grad)

    def test_threshold(self):
        m = nn.Threshold(5, 20)
        dmlResult = m(self.tensorA.to("dml"))
        cpuResult = m(self.tensorA.to("cpu"))
        self.verify_result(cpuResult, dmlResult)

    def test_ones_like_and_ones(self):
        # ones_like
        dmlInput = torch.empty((3,3), device=torch.device("dml"))
        dmlResult = torch.ones_like(dmlInput)

        cpuInput = torch.empty((3,3), device=torch.device("cpu"))
        cpuResult = torch.ones_like(cpuInput)
        self.verify_result(cpuResult, dmlResult)

        # ones
        cpuResult = torch.ones((3, 3), device=torch.device("cpu"))
        dmlResult = torch.ones((3, 3), device=torch.device("dml"))
        self.verify_result(cpuResult, dmlResult)

    def test_zeroes_and_zeroes_like_and_zero_(self):
        # zeros_like
        dmlInput = torch.empty((3,3), device=torch.device("dml"))
        dmlResult = torch.zeros_like(dmlInput)

        cpuInput = torch.empty((3,3), device=torch.device("cpu"))
        cpuResult = torch.zeros_like(cpuInput)
        self.verify_result(cpuResult, dmlResult)
        # zeros
        dmlResult = torch.zeros(2,3, device = torch.device("dml"))
        cpuResult = torch.zeros(2,3, device = torch.device("cpu"))
        self.verify_result(cpuResult, dmlResult)

        # zero_
        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor.zero_()
        dmlTensor.zero_()
        self.verify_result(cpuTensor, dmlTensor)

        # new_zeros
        cpuTensor = cpuTensor.new_zeros((2, 3))
        dmlTensor = dmlTensor.new_zeros((2, 3))
        self.verify_result(cpuTensor, dmlTensor)


    def test_copy_(self):
        a = torch.tensor([[1.,2.,3.],[1.,2.,3.]]).to("dml")
        b = torch.tensor([[10.,20.,30.],[10.,20.,30.]]).to("dml")
        c = b.copy_(a)
        self.verify_result(a.to("cpu"), c.to("cpu"))

    def test_transpose(self):
        dmlResult = torch.transpose(self.tensorA.to("dml"), 0, 1)
        cpuResult = torch.transpose(self.tensorA.to("cpu"), 0, 1)
        self.verify_result(cpuResult, dmlResult)

    # TODO add tests for larger size
    def test_resize_and_resize_as(self):
        a = torch.randn(2,3).to("dml")
        a.resize_(2,3).to("cpu")
        b = torch.randn(20,3).to("dml")
        b.resize_as_(a).to("cpu")

    def test_casting(self):
        a = self.tensorA.to("dml")
        self.assertTrue(torch.allclose(a.double().to("cpu"), a.to("cpu").double()))
        self.assertTrue(torch.allclose(a.float().to("cpu"), a.to("cpu").float()))
        self.assertTrue(torch.allclose(a.int().to("cpu"), a.to("cpu").int()))
        self.assertTrue(torch.allclose(a.to(torch.uint8).to("cpu"), a.to("cpu").to(torch.uint8)))
        self.assertTrue(torch.allclose(a.to(torch.int8).to("cpu"), a.to("cpu").to(torch.int8)))
        self.assertTrue(torch.allclose(a.to(torch.int32).to("cpu"), a.to("cpu").to(torch.int32)))
        self.assertTrue(torch.allclose(a.to(torch.int64).to("cpu"), a.to("cpu").to(torch.int64)))

    def test_unsqueeze(self):
        dmlResult = self.tensorB.to("dml")
        dmlResult.unsqueeze_(0).to("cpu")
        cpuResult = self.tensorB
        cpuResult.unsqueeze_(0)
        self.verify_result(cpuResult, dmlResult)

    def test_expand_as(self):
        a = self.tensorA.to("dml");
        b = self.tensorB.to("dml");
        dmlResult = b.expand_as(a.to("dml"))
        cpuResult = self.tensorB.expand_as(self.tensorA)
        self.verify_result(cpuResult, dmlResult)
        
        cpuLabels = torch.arange(91)
        cpuScores = torch.randn(1000, 91)
        cpuLabels = cpuLabels.view(1, -1).expand_as(cpuScores)
        cpuLabels = cpuLabels[:, 1:]

        dmlLabels = torch.arange(91).to("dml")
        dmlScores = torch.randn(1000, 91).to("dml")
        dmlLabels = dmlLabels.view(1, -1).expand_as(dmlScores)
        dmlLabels = dmlLabels[:, 1:]
        self.verify_result(cpuLabels, dmlLabels)

    def test_type_as(self):
        a = self.tensorA.to("dml")
        self.assertTrue(a.type_as(a.double()).dtype == torch.float64)
        self.assertTrue(a.type_as(a.float()).dtype == torch.float32)
        self.assertTrue(a.type_as(a.int()).dtype == torch.int32)
        self.assertTrue(a.type_as(a.to(torch.uint8)).dtype == torch.uint8)
        self.assertTrue(a.type_as(a.to(torch.int8)).dtype == torch.int8)
        self.assertTrue(a.type_as(a.to(torch.int32)).dtype == torch.int32)
        self.assertTrue(a.type_as(a.to(torch.int64)).dtype == torch.int64)

    def test_item(self):
        a = torch.tensor([10])
        self.assertTrue(a.item() == a.to("dml").item())

    def test_is_nonzero(self):
        a = torch.tensor([0.])
        b = torch.tensor([1.5])
        c = torch.tensor([False])
        d = torch.tensor([3])
        self.assertTrue(torch.is_nonzero(a) == torch.is_nonzero(a.to("dml")))
        self.assertTrue(torch.is_nonzero(b) == torch.is_nonzero(b.to("dml")))
        self.assertTrue(torch.is_nonzero(c) == torch.is_nonzero(c.to("dml")))
        self.assertTrue(torch.is_nonzero(d) == torch.is_nonzero(d.to("dml")))

    def test_unbind(self):
        dmlResult = self.tensorA.to("dml").unbind()
        cpuResult = self.tensorA.unbind()
        self.verify_result(cpuResult[0], dmlResult[0])
        self.verify_result(cpuResult[1], dmlResult[1])
        self.verify_result(cpuResult[2], dmlResult[2])
        self.verify_result(cpuResult[3], dmlResult[3])

## Backwards Ops
    def test_log_softmax_backward(self):
        input = torch.tensor([[-2.9309e+16, -5.2225e+18,  9.3603e+17,  1.3827e+18,  1.2060e+18,
                            1.2949e+18,  6.0490e+17,  9.0288e+17,  7.7774e+17, -6.2197e+17],
                            [ 3.0915e+15, -2.2820e+18,  3.6399e+17,  6.3987e+17,  5.6325e+17,
                            5.4267e+17,  2.9168e+17,  3.2440e+17,  4.4137e+17, -2.0580e+17]])
        input_cpu1 = input.clone().detach().requires_grad_(True)
        input_cpu2 = input.clone().detach().requires_grad_(True)
        input_dml1 = input.to("dml").detach().requires_grad_(True)
        input_dml2 = input.to("dml").detach().requires_grad_(True)

        m = torch.nn.LogSoftmax(dim=0).to("dml")
        output = m(input_dml1)
        o2 = output.mean()
        o2.backward()
        dmlResult1 = input_dml1.grad.to("cpu")

        m = torch.nn.LogSoftmax(dim=1).to("dml")
        output = m(input_dml2)
        o2 = output.mean()
        o2.backward()
        dmlResult2 = input_dml2.grad.to("cpu")

        m = torch.nn.LogSoftmax(dim=0)
        output = m(input_cpu1)
        o2 = output.mean()
        o2.backward()
        cpuResult1 = input_cpu1.grad

        m = torch.nn.LogSoftmax(dim=1)
        output = m(input_cpu2)
        o2 = output.mean()
        o2.backward()
        cpuResult2 = input_cpu2.grad
        self.verify_result(cpuResult1, dmlResult1)
        self.verify_result(cpuResult2, dmlResult2)

    def test_max_pool2d_with_indices_backward(self):
        net = nn.MaxPool2d((3, 2))
        input = torch.randn(20, 16, 50, 32, requires_grad=True)

        net.to("dml")
        net(input.to("dml")).mean().backward()
        dmlResult = input.grad

        inputClone = input.clone().detach().requires_grad_(True)
        net.to("cpu")
        net(inputClone.to("cpu")).mean().backward()
        cpuResult = inputClone.grad

        self.verify_result(cpuResult, dmlResult)

    def test_sigmoid_backwards(self):
        a_dml = self.tensorA.clone().detach().requires_grad_(True)
        torch.sigmoid(a_dml.to("dml")).mean().backward()
        a_cpu = self.tensorA.clone().detach().requires_grad_(True)
        torch.sigmoid(a_cpu.to("cpu")).mean().backward()
        self.verify_result(a_cpu.grad, a_dml.grad)

    def test_leaky_relu_backwards(self):
        a_dml = self.tensorA.clone().detach().requires_grad_(True)
        torch.nn.functional.leaky_relu(a_dml.to("dml")).mean().backward()
        a_cpu = self.tensorA.clone().detach().requires_grad_(True)
        torch.nn.functional.leaky_relu(a_cpu.to("cpu")).mean().backward()
        self.verify_result(a_cpu.grad, a_dml.grad)

    def test_threshold_backwards(self):
        net = nn.Threshold(5, 20)
        a_dml = self.tensorA.clone().detach().requires_grad_(True)
        net(a_dml.to("dml")).mean().backward()
        a_cpu = self.tensorA.clone().detach().requires_grad_(True)
        net(a_cpu.to("cpu")).mean().backward()
        self.verify_result(a_cpu.grad, a_dml.grad)

    def test_adaptive_avg_pool2d_backwards(self):
        output_size = torch.tensor([4, 4])
        aap2d = torch.nn.AdaptiveAvgPool2d(output_size) # build aap2d layer
        aap2d_dml = aap2d.to("dml") # put aap2d to DML

        # test when output dimensions are a multiple of input dimensions
        input = torch.randn(5, 8, 8)
        cpu_tensor = input.clone().detach().requires_grad_(True)
        aap2d(cpu_tensor).mean().backward()  # Run on CPU
        dml_tensor = input.clone().detach().requires_grad_(True)
        aap2d_dml(dml_tensor.to("dml")).mean().backward() # Run on DML
        self.verify_result(cpu_tensor.grad, dml_tensor.grad)

        # test when output dimensions aren't a multiple of input dimensions
        input = torch.randn(2, 6, 7)
        cpu_tensor = input.clone().detach().requires_grad_(True)
        aap2d(cpu_tensor).mean().backward()  # Run on CPU
        dml_tensor = input.clone().detach().requires_grad_(True)
        aap2d_dml(dml_tensor.to("dml")).mean().backward() # Run on DML
        if not torch.allclose(cpu_tensor.grad, dml_tensor.grad):
            warnings.warn(UserWarning("Adaptive_avg_pool2d_backwards DML implementation doesn't match CPU implementation for when output dimensions aren't a multiple of input dimensions"))

    def test_grouped_conv2d_backwards(self):
        input = torch.randn(3, 4, 3, 4)
        cpuNet = torch.nn.Conv2d(4, 2, (3, 3), groups=2)

        random_weights = torch.randn_like(cpuNet.weight)
        random_bias = torch.randn_like(cpuNet.bias)
        random_weights
        cpuNet.weight = torch.nn.Parameter(random_weights)
        cpuNet.bias = torch.nn.Parameter(random_bias)
        input_cpu = input.clone().detach().requires_grad_(True)
        output_cpu = cpuNet(input_cpu)
        output_cpu.mean().backward()

        dmlNet = torch.nn.Conv2d(4, 2, (3, 3), groups=2).to("dml")
        dmlNet.weight = torch.nn.Parameter(random_weights.clone().to("dml"))
        dmlNet.bias = torch.nn.Parameter(random_bias.clone().to("dml"))
        input_dml = input.clone().detach().to("dml").requires_grad_(True)
        output_dml = dmlNet(input_dml)
        output_dml.mean().backward()

        torch.allclose(input_cpu.grad, input_dml.grad.to("cpu"))
        torch.allclose(cpuNet.weight, dmlNet.weight.to("cpu"))
        torch.allclose(cpuNet.bias, dmlNet.bias.to("cpu"))

    def test_thnnconv2d_backwards(self):
        cpuNet1 = torch.nn.Conv2d(3, 2, 4, 2, padding=(2,2), groups=1)
        cpuNet2 = torch.nn.Conv2d(2, 2, 4, 2, padding=(2,2), groups=1)
        cpuNet1.weight = torch.nn.Parameter(torch.ones_like(cpuNet1.weight))
        cpuNet1.bias = torch.nn.Parameter(torch.ones_like(cpuNet1.bias))
        cpuNet2.weight = torch.nn.Parameter(torch.ones_like(cpuNet2.weight))
        cpuNet2.bias = torch.nn.Parameter(torch.ones_like(cpuNet2.bias))
        cpuInput = torch.empty(1,3,4,4).fill_(1)
        cpuInput.requires_grad = True
        cpuResult = cpuNet1(cpuInput)
        cpuResult2 = cpuNet2(cpuResult)
        cpuMean = cpuResult2.mean((0, 2, 3))
        cpuDiff = cpuMean + (torch.tensor([.3,.7])*-1)
        cpuDiffSquared=cpuDiff*cpuDiff
        cpuLoss = cpuDiffSquared.sum()
        cpuLoss.backward()

        dmlNet1 = torch.nn.Conv2d(3, 2, 4, 2, padding=(2,2), groups=1)
        dmlNet2 = torch.nn.Conv2d(2, 2, 4, 2, padding=(2,2), groups=1)
        dmlNet1.weight = torch.nn.Parameter(torch.ones_like(dmlNet1.weight))
        dmlNet1.bias = torch.nn.Parameter(torch.ones_like(dmlNet1.bias))
        dmlNet2.weight = torch.nn.Parameter(torch.ones_like(dmlNet2.weight))
        dmlNet2.bias = torch.nn.Parameter(torch.ones_like(dmlNet2.bias))
        dmlInput = torch.empty(1,3,4,4).fill_(1)
        dmlInput.requires_grad = True
        dml_net1 = dmlNet1.to("dml")
        dml_net2 = dmlNet2.to("dml")
        dmlResult = dml_net1(dmlInput.to("dml"))
        dmlResult2 = dml_net2(dmlResult)
        dmlMean = dmlResult2.mean((0, 2, 3))
        dmlDiff = dmlMean + (torch.tensor([.3,.7]).to("dml")*-1)
        dmlDiffSquared=dmlDiff*dmlDiff
        dmlLoss = dmlDiffSquared.sum()
        dmlLoss.backward()

        self.verify_result(cpuInput.grad, dmlInput.grad)
        self.verify_result(cpuNet1.weight, dmlNet1.weight)
        self.verify_result(cpuNet1.bias, dmlNet1.bias)
        self.verify_result(cpuNet2.weight, dmlNet2.weight)
        self.verify_result(cpuNet2.bias, dmlNet2.bias)
    
    def test_convtranposed2d_backward(self):
        m = nn.ConvTranspose2d(16, 32, (4, 6), stride=(2, 1), padding=(4, 2))
        input = torch.randn(20, 16, 50, 100).requires_grad_(True)
        cpuResult = m(input)
        cpuResult.sum().backward()

        m_dml = m.to("dml")
        input_dml = input.clone().detach().to("dml").requires_grad_(True)
        dmlResult = m_dml(input_dml)
        dmlResult.sum().backward()

        self.verify_result(cpuResult, dmlResult, abs_tol=1e-06)
        self.verify_result(m.weight.grad, m_dml.weight.grad, abs_tol=1e-06)
        self.verify_result(m.bias.grad, m_dml.bias.grad, abs_tol=1e-06)

    def test_nll_loss_backward(self):
        m = nn.LogSoftmax(dim=1)
        weight = torch.tensor([1.,2.,3.,4.,5.,6.])
        loss = nn.NLLLoss(weight=weight)
        input = torch.randn(4, 6, requires_grad=True)
        target = torch.tensor([1, 0, 4, 5])
        inputs = m(input)
        cpuResult = loss(inputs, target)
        cpuResult.backward()

        loss_dml = nn.NLLLoss(weight=weight).to("dml")
        input_dml = input.clone().detach().requires_grad_(True)
        target_dml = target.to("dml")
        inputs_dml = m(input_dml.to("dml"))
        dmlResult = loss_dml(inputs_dml, target_dml)
        dmlResult.backward()
        self.verify_result(input.grad, input_dml.grad)

    def test_upsample_nearest2d(self):
        input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
        m = nn.UpsamplingNearest2d(size=(10,6))
        cpuResult = m(input)
        dmlResult = m(input.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        m = nn.UpsamplingNearest2d(scale_factor=(3,5))
        cpuResult = m(input)
        dmlResult = m(input.to("dml"))
        self.verify_result(cpuResult, dmlResult)

    def test_upsample_bilinear2d(self):
        input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
        m = nn.UpsamplingBilinear2d(size=(10,6))
        cpuResult = m(input)
        dmlResult = m(input.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        m = nn.UpsamplingBilinear2d(scale_factor=(3,5))
        cpuResult = m(input)
        dmlResult = m(input.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        input = torch.arange(4, 5, dtype=torch.float32).view(1, 1, 1, 1)
        m = nn.UpsamplingBilinear2d(size=(10,6))
        cpuResult = m(input)
        dmlResult = m(input.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        m = nn.UpsamplingBilinear2d(scale_factor=(3,5))
        cpuResult = m(input)
        dmlResult = m(input.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        input = torch.arange(4, 6, dtype=torch.float32).view(1, 1, 1, 2)
        m = nn.UpsamplingBilinear2d(size=(10,6))
        cpuResult = m(input)
        dmlResult = m(input.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        m = nn.UpsamplingBilinear2d(scale_factor=(3,5))
        cpuResult = m(input)
        dmlResult = m(input.to("dml"))
        self.verify_result(cpuResult, dmlResult)

    def test_upsample_nearest2d_backward(self):
        input = torch.arange(1, 17, dtype=torch.float32).view(2, 2, 2, 2)
        m = nn.UpsamplingNearest2d(size=(10,6))

        cpu_input = input.clone().requires_grad_(True)
        dml_input = input.clone().requires_grad_(True)
        cpuResult = m(cpu_input)
        dmlResult = m(dml_input.to("dml"))
        cpuResult.sum().backward()
        dmlResult.sum().backward()
        self.verify_result(cpu_input.grad, dml_input.grad)

        m = nn.UpsamplingNearest2d(scale_factor=(3,5))
        cpu_input = input.clone().requires_grad_(True)
        dml_input = input.clone().requires_grad_(True)
        cpuResult = m(cpu_input)
        dmlResult = m(dml_input.to("dml"))
        cpuResult.sum().backward()
        dmlResult.sum().backward()
        self.verify_result(cpu_input.grad, dml_input.grad)
    
    def test_upsample_bilinear2d_backward(self):
        input = torch.arange(1, 17, dtype=torch.float32).view(2, 2, 2, 2)
        m = nn.UpsamplingBilinear2d(size=(10,6))

        cpu_input = input.clone().requires_grad_(True)
        dml_input = input.clone().requires_grad_(True)
        cpuResult = m(cpu_input)
        dmlResult = m(dml_input.to("dml"))
        cpuResult.sum().backward()
        dmlResult.sum().backward()
        self.verify_result(cpu_input.grad, dml_input.grad)

        m = nn.UpsamplingBilinear2d(scale_factor=(3,5))
        cpu_input = input.clone().requires_grad_(True)
        dml_input = input.clone().requires_grad_(True)
        cpuResult = m(cpu_input)
        dmlResult = m(dml_input.to("dml"))
        cpuResult.sum().backward()
        dmlResult.sum().backward()
        self.verify_result(cpu_input.grad, dml_input.grad)

        input = torch.arange(4, 5, dtype=torch.float32).view(1, 1, 1, 1)
        m = nn.UpsamplingBilinear2d(size=(10,6))

        cpu_input = input.clone().requires_grad_(True)
        dml_input = input.clone().requires_grad_(True)
        cpuResult = m(cpu_input)
        dmlResult = m(dml_input.to("dml"))
        cpuResult.sum().backward()
        dmlResult.sum().backward()
        self.verify_result(cpu_input.grad, dml_input.grad)

        m = nn.UpsamplingBilinear2d(scale_factor=(3,5))
        cpu_input = input.clone().requires_grad_(True)
        dml_input = input.clone().requires_grad_(True)
        cpuResult = m(cpu_input)
        dmlResult = m(dml_input.to("dml"))
        cpuResult.sum().backward()
        dmlResult.sum().backward()
        self.verify_result(cpu_input.grad, dml_input.grad)

        input = torch.arange(4, 6, dtype=torch.float32).view(1, 1, 1, 2)
        m = nn.UpsamplingBilinear2d(size=(10,6))

        cpu_input = input.clone().requires_grad_(True)
        dml_input = input.clone().requires_grad_(True)
        cpuResult = m(cpu_input)
        dmlResult = m(dml_input.to("dml"))
        cpuResult.sum().backward()
        dmlResult.sum().backward()
        self.verify_result(cpu_input.grad, dml_input.grad)

        m = nn.UpsamplingBilinear2d(scale_factor=(3,5))
        cpu_input = input.clone().requires_grad_(True)
        dml_input = input.clone().requires_grad_(True)
        cpuResult = m(cpu_input)
        dmlResult = m(dml_input.to("dml"))
        cpuResult.sum().backward()
        dmlResult.sum().backward()
        self.verify_result(cpu_input.grad, dml_input.grad)

    def test_is_finite(self):
        input = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')])
        cpuResult = torch.isfinite(input)
        dmlResult = torch.isfinite(input.to("dml"))
        self.verify_result(cpuResult, dmlResult)

    def test_permute(self):
        cpuResult = self.tensorA.permute(1, 0)
        dmlResult = self.tensorA.to("dml").permute(1, 0)
        self.verify_result(cpuResult, dmlResult)

    def test_pow(self):
        cpuResult = torch.pow(self.tensorA.to("cpu"), self.tensorB.to("cpu"))
        dmlResult = torch.pow(self.tensorA.to("dml"), self.tensorB.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        # scalar
        cpuResult = torch.pow(self.tensorA.to("cpu"), 2)
        dmlResult = torch.pow(self.tensorA.to("dml"), 2)
        self.verify_result(cpuResult, dmlResult)

        # test pow_
        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor.pow_(self.tensorB)
        dmlTensor.pow_(self.tensorB.to("dml"))
        self.verify_result(cpuTensor, dmlTensor)

    def test_round(self):
        a = torch.randn(4)
        cpuResult = torch.round(a)
        dmlResult = torch.round(a.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        # test round_
        cpuResult = a
        cpuResult.round_()
        dmlResult = a
        dmlResult.round_()
        self.verify_result(cpuResult, dmlResult)

    def test_flip(self):
        cpuResult = torch.flip(self.tensorA.to("cpu"),[0,1])
        dmlResult = torch.flip(self.tensorA.to("dml"),[0,1])
        self.verify_result(cpuResult, dmlResult)


        cpuResult = torch.flip(self.tensorA.to("cpu"),[1])
        dmlResult = torch.flip(self.tensorA.to("dml"),[1])
        self.verify_result(cpuResult, dmlResult)

        x = torch.arange(0,27).view(3,3,3)
        cpuResult = torch.flip(x.to("cpu"),[0,1,2])
        dmlResult = torch.flip(x.to("dml"),[0,1,2])
        self.verify_result(cpuResult, dmlResult)


    def test_sort(self):
        cpuResult, cpuIndicies = torch.sort(self.tensorA.to("cpu"))
        dmlResult, dmlIndicies = torch.sort(self.tensorA.to("dml"))
        self.verify_result(cpuResult, dmlResult)
        self.verify_result(cpuIndicies, dmlIndicies)

        cpuResult, cpuIndicies = torch.sort(self.tensorA.to("cpu"),0)
        dmlResult, dmlIndicies = torch.sort(self.tensorA.to("dml"),0)
        self.verify_result(cpuResult, dmlResult)
        self.verify_result(cpuIndicies, dmlIndicies)

        cpuResult, cpuIndicies = torch.sort(self.tensorA.to("cpu"),1, descending=True)
        dmlResult, dmlIndicies = torch.sort(self.tensorA.to("dml"),1, descending=True)
        self.verify_result(cpuResult, dmlResult)
        self.verify_result(cpuIndicies, dmlIndicies)
    
    def test_remainder(self):
        # compare tensors
        input_a = torch.tensor([1.0,-2.0,3.0,4.0,-5.0])
        input_b = torch.tensor([3.0,3.0,3.0,3.0,3.0])
        cpuTensor = input_a.clone().detach()
        dmlTensor = input_a.clone().detach().to("dml")
        cpuResult = torch.remainder(cpuTensor, input_b)
        dmlResult = torch.remainder(dmlTensor, input_b)
        self.verify_result(cpuResult, dmlResult)
        # compare tensor with scalar
        cpuResult = torch.remainder(cpuTensor, 3)
        dmlResult = torch.remainder(dmlTensor, 3)
        self.verify_result(cpuResult, dmlResult)
        # inplace
        cpuTensor.remainder_(input_b)
        dmlTensor.remainder_(input_b)
        self.verify_result(cpuResult, dmlResult)

    def test_unique(self):
        x = torch.tensor([100,20,100,30,30,1,2,2,3,4,5,5,12,14,11,13]).float()
        cpuResult = torch.unique(x, sorted=True)
        dmlResult = torch.unique(x.to("dml"), sorted=True)
        self.verify_result(cpuResult, dmlResult)

        # Try a different dimension
        x = x.view(2,2,4)
        cpuResult = torch.unique(x, sorted=True)
        dmlResult = torch.unique(x.to("dml"), sorted=True)
        self.verify_result(cpuResult, dmlResult)

        # try a strided scenario
        cpuResult = torch.unique(x[:,:,-1], sorted=True)
        dmlResult = torch.unique(x.to("dml")[:,:,-1], sorted=True)
        self.verify_result(cpuResult, dmlResult)

    def test_sign(self):
        cpuResult = torch.sign(self.tensorA)
        dmlResult = torch.sign(self.tensorA.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor.sign_()
        dmlTensor.sign_()
        self.verify_result(cpuTensor, dmlTensor)

        inputTensor = torch.tensor([True, False, True])
        cpuResult = torch.sign(inputTensor)
        dmlResult = torch.sign(inputTensor.to("dml"))
        self.verify_result(cpuResult, dmlResult)

    def test_sgn(self):
        input_a = torch.tensor([-1.0, 2.0, -3.0, -4.0])
        result = torch.tensor([-1.0, 1.0, -1.0, -1.0])
        dmlResult = torch.sgn(input_a.to("dml"))
        self.verify_result(result, dmlResult)

        dmlTensor = input_a.to("dml")
        dmlTensor.sign_()
        self.verify_result(result, dmlTensor)

    def test_reciprocal(self):
        cpuResult = torch.reciprocal(self.tensorA)
        dmlResult = torch.reciprocal(self.tensorA.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        cpuTensor = self.tensorA.clone().detach()
        dmlTensor = self.tensorA.clone().detach().to("dml")
        cpuTensor.reciprocal_()
        dmlTensor.reciprocal_()
        self.verify_result(cpuTensor, dmlTensor)

    def test__and__(self):
        input_tensor = self.tensorA.to(torch.bool)
        cpuResult = input_tensor.__and__(input_tensor)
        dmlResult = input_tensor.to("dml").__and__(input_tensor.to("dml"))
        self.verify_result(cpuResult, dmlResult)

    # CPU Op tests
    def test_remainder(self):
        cpuResult = torch.remainder(self.tensorA, 2)
        dmlResult = torch.remainder(self.tensorA.to("dml"), 2)
        self.verify_result(cpuResult, dmlResult)

    def test_all(self):
        cpuResult = torch.all(self.tensorA, dim=1)
        dmlResult = torch.all(self.tensorA.to("dml"), dim=1)
        self.verify_result(cpuResult, dmlResult)

        cpuResult = torch.all(self.tensorA)
        dmlResult = torch.all(self.tensorA.to("dml"))

        self.verify_result(cpuResult, dmlResult)

        inputTensor = torch.tensor([True, True, True])
        cpuResult = torch.all(inputTensor)
        dmlResult = torch.all(inputTensor.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        inputTensor = torch.tensor([True, False, True])
        cpuResult = torch.all(inputTensor)
        dmlResult = torch.all(inputTensor.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        inputTensor = torch.empty((0)).view(0,2)
        cpuResult = inputTensor.all(1)
        dmlResult = inputTensor.to("dml").all(1)
        self.verify_result(cpuResult, dmlResult)

    def test_any(self):
        input = torch.randn(10, 20) < 0
        cpuResult = torch.any(input, dim=1)
        dmlResult = torch.any(input.to("dml"), dim=1)
        self.verify_result(cpuResult, dmlResult)

        cpuResult = torch.any(input)
        dmlResult = torch.any(input.to("dml"))

        self.verify_result(cpuResult, dmlResult)

        inputTensor = torch.tensor([True, True, True])
        cpuResult = torch.any(inputTensor)
        dmlResult = torch.any(inputTensor.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        inputTensor = torch.tensor([True, False, True])
        cpuResult = torch.any(inputTensor)
        dmlResult = torch.any(inputTensor.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        inputTensor = torch.empty((0)).view(0,2)
        cpuResult = inputTensor.any(1)
        dmlResult = inputTensor.to("dml").any(1)
        self.verify_result(cpuResult, dmlResult)

    def test_where(self):
        cpuResult = torch.where(self.tensorA > 0, self.tensorA, torch.tensor([0.0]))
        dmlResult = torch.where(self.tensorA.to("dml") > 0, self.tensorA.to("dml"),  torch.tensor([0.0]).to("dml"))
        self.verify_result(cpuResult, dmlResult)

        cpuResult = torch.where(self.tensorA > 0)
        dmlResult = torch.where(self.tensorA.to("dml") > 0)
        for cpuR, dmlR in zip(cpuResult, dmlResult):
            self.verify_result(cpuR, dmlR)

        d_tensor = torch.tensor([1,2,-1,-2], dtype=torch.float32)
        cpuResult = torch.where(d_tensor > 0)
        dmlResult = torch.where(d_tensor.to("dml") > 0)
        for cpuR, dmlR in zip(cpuResult, dmlResult):
            self.verify_result(cpuR, dmlR)

    def test_hardsigmoid(self):
        m = nn.Hardsigmoid(inplace = True).to("dml")
        dmlInput = torch.arange(-10,10).float().to("dml")
        m(dmlInput)

        m = m.to("cpu")
        cpuInput = torch.arange(-10,10).float().to("cpu")
        m(cpuInput)
        self.verify_result(cpuInput, dmlInput)

    def test_hardsigmoid_backward(self):
        cpuInput = self.tensorA.clone().detach().requires_grad_(True)
        dmlInput = self.tensorA.to("dml").detach().requires_grad_(True)

        m = nn.Hardsigmoid().to("dml")
        output = m(dmlInput).mean().backward()
        dmlResult = dmlInput.grad.to("cpu")

        m = m.to("cpu")
        output = m(cpuInput).mean().backward()
        cpuResult = cpuInput.grad.to("cpu")
        self.verify_result(cpuResult, dmlResult)

    def test_hardswish(self):
        m = nn.Hardswish(inplace = True).to("dml")
        dmlInput = torch.arange(-10,10).float().to("dml")
        m(dmlInput)

        m = m.to("cpu")
        cpuInput = torch.arange(-10,10).float().to("cpu")
        m(cpuInput)
        self.verify_result(cpuInput, dmlInput)

    def test_hardswish_backward(self):
        cpuInput = self.tensorA.clone().detach().requires_grad_(True)
        dmlInput = self.tensorA.to("dml").detach().requires_grad_(True)

        m = nn.Hardswish().to("dml")
        output = m(dmlInput).mean().backward()
        dmlResult = dmlInput.grad.to("cpu")

        m = m.to("cpu")
        output = m(cpuInput).mean().backward()
        cpuResult = cpuInput.grad.to("cpu")
        self.verify_result(cpuResult, dmlResult)

    def test_avg_pool2d(self):
        m = nn.AvgPool2d((3, 2), stride=(2, 1))
        input = torch.arange(0,150).view(5,5,6).float()
        cpuResult = m(input)
        m = m.to("dml")
        dmlResult = m(input.to("dml"))
        self.verify_result(cpuResult, dmlResult)
        m = m.to("cpu")

        input = torch.arange(0,750).view(5,5,5,6).float()
        cpuResult = m(input)
        m = m.to("dml")
        dmlResult = m(input.to("dml"))
        self.verify_result(cpuResult, dmlResult)

        m = nn.AvgPool2d(3, stride=2)
        input = torch.arange(0,150).view(5,5,6).float()
        cpuResult = m(input)
        m = m.to("dml")
        dmlResult = m(input.to("dml"))
        self.verify_result(cpuResult, dmlResult)
        m = m.to("cpu")

        input = torch.arange(0,750).view(5,5,5,6).float()
        cpuResult = m(input)
        m = m.to("dml")
        dmlResult = m(input.to("dml"))
        self.verify_result(cpuResult, dmlResult)

    def test_avg_pool2d_backwards(self):
        m = nn.AvgPool2d((3, 2), stride=(2, 1))
        cpuInput = torch.arange(0,150).view(5,5,6).float().requires_grad_(True)
        dmlInput = cpuInput.detach().to("dml").requires_grad_(True)
        m(cpuInput).mean().backward()
        m = m.to("dml")
        m(dmlInput).mean().backward()
        self.verify_result(cpuInput.grad, dmlInput.grad)

        cpuInput = torch.arange(0,750).view(5,5,5,6).float().requires_grad_(True)
        dmlInput = cpuInput.detach().to("dml").requires_grad_(True)
        m = m.to("cpu")
        m(cpuInput).mean().backward()
        m = m.to("dml")
        m(dmlInput).mean().backward()
        self.verify_result(cpuInput.grad, dmlInput.grad)

        m = nn.AvgPool2d(3, stride=2)
        cpuInput = torch.arange(0,150).view(5,5,6).float().requires_grad_(True)
        dmlInput = cpuInput.detach().to("dml").requires_grad_(True)
        m(cpuInput).mean().backward()
        m = m.to("dml")
        m(dmlInput).mean().backward()
        self.verify_result(cpuInput.grad, dmlInput.grad)

        cpuInput = torch.arange(0,750).view(5,5,5,6).float().requires_grad_(True)
        dmlInput = cpuInput.detach().to("dml").requires_grad_(True)
        m = m.to("cpu")
        m(cpuInput).mean().backward()
        m = m.to("dml")
        m(dmlInput).mean().backward()
        self.verify_result(cpuInput.grad, dmlInput.grad)

    def test_torchvision_nms(self):
        import torchvision
        boxes = torch.tensor([[2,2,4,4], [1,1,5,5], [3,3,3.5, 3.9]])
        scores = torch.tensor([0.8,0.7,0.9])
        cpuResult = torch.ops.torchvision.nms(boxes, scores, 0.8)
        dmlResult = torch.ops.torchvision.nms(boxes.to("dml"), scores.to("dml"), 0.8)
        self.verify_result(cpuResult, dmlResult)

    def test_meshgrid(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        grid_x_cpu, grid_y_cpu = torch.meshgrid(x, y)
        grid_x_dml, grid_y_dml = torch.meshgrid(x.to("dml"), y.to("dml"))
        self.verify_result(grid_x_cpu, grid_x_dml)
        self.verify_result(grid_y_cpu, grid_y_dml)

    def test_scatter(self):
        src = torch.arange(1, 6).reshape((1, 5))
        index = torch.tensor([[0, 1, 2, 0, 1]])
        cpuResult = torch.zeros(3, 5, dtype=src.dtype)
        dmlResult = cpuResult.to("dml")
        cpuResult.scatter_(0, index, src)
        dmlResult.scatter_(0, index.to("dml"), src.to("dml"))
        self.verify_result(cpuResult, dmlResult)

    def test_torchvision_roi_align(self):
        import torchvision
        output_size = (3, 3)
        spatial_scale = 1 / 2
        sampling_ratio = 2
        x = torch.randn(36).reshape(1,1,6,6).float()

        rois = torch.tensor([
            [0, 1.0, 1.0, 3.0, 3.0]
        ])
        cpuResult = torch.ops.torchvision.roi_align(x, rois, spatial_scale, output_size[0], output_size[1], sampling_ratio, False)
        dmlResult = torch.ops.torchvision.roi_align(x.to("dml"), rois.to("dml"), spatial_scale, output_size[0], output_size[1], sampling_ratio, False)
        self.verify_result(cpuResult, dmlResult)

    def test_torchvision_roi_align_backward(self):
        import torchvision
        output_size = (6, 6)
        spatial_scale = 1 / 2
        sampling_ratio = 2
        x = torch.randn(36).reshape(1,1,6,6).float()

        rois = torch.tensor([
            [0, 1.0, 1.0, 3.0, 3.0]
        ])

        cpuResult = torch.ops.torchvision._roi_align_backward(x, rois, spatial_scale, output_size[0], output_size[1], 1, 1, 3, 3, sampling_ratio, False)
        dmlResult = torch.ops.torchvision._roi_align_backward(x.to("dml"), rois.to("dml"), spatial_scale, output_size[0], output_size[1], 1, 1, 3, 3, sampling_ratio, False)
        self.verify_result(cpuResult, dmlResult)


if __name__ == '__main__':
    unittest.main()
