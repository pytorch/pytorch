import sys
import pickle
import copy
import unittest
from collections import OrderedDict

import numpy as np
from numpy.testing import *

import torch
import torch.multiprocessing as mp


def _rebuild_tensor(type_, data, requires_grad, backward_hooks):
    param = type_(data, requires_grad)
    # NB: This line exists only for backwards compatibility; the
    # general expectation is that backward_hooks is an empty
    # OrderedDict.  See Note [Don't serialize hooks]
    param._backward_hooks = backward_hooks

    return param


class A(torch.Tensor):

    def __new__(cls, data=None, requires_grad=False):
        if data is None:
            data = torch.Tensor()
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        return self

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.data.clone(), self.requires_grad)
            memo[id(self)] = result
            return result

    def __reduce_ex__(self, proto):
        # See Note [Don't serialize hooks]
        return (
            _rebuild_tensor,
            (self.__class__, self.data, self.requires_grad, OrderedDict())
        )

    def __repr__(self):
        return 'A'


class B(torch.Tensor):

    def __new__(cls, data=None, requires_grad=False):
        if data is None:
            data = torch.Tensor()
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        return self

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.data.clone(), self.requires_grad)
            memo[id(self)] = result
            return result

    def __reduce_ex__(self, proto):
        # See Note [Don't serialize hooks]
        return (
            _rebuild_tensor,
            (self.__class__, self.data, self.requires_grad, OrderedDict())
        )

    def __repr__(self):
        return 'B'


class C(A, B):

    def __repr__(self):
        return 'C'


mp.set_start_method('spawn') if sys.platform == "win32" else mp.set_start_method('fork')


def sub_process_func(queue, *args):
    result = torch.Tensor.add(*args)
    queue.put("%s = func(%s, %s)" % (type(result), type(args[0]),  type(args[1])))


class TestSubclassSerialization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a = A(torch.ones((1,), dtype=torch.int32))
        cls.b = B(torch.ones((1,), dtype=torch.int32))
        cls.c = C(torch.ones((1,), dtype=torch.int32))
        cls.out_a = A(torch.ones((1,), dtype=torch.int32))
        cls.out_b = B(torch.ones((1,), dtype=torch.int32))
        cls.out_c = C(torch.ones((1,), dtype=torch.int32))

    def setUp(self):
        pass

    def test_save(self):
        filename = "saved_state.p"
        with open(filename, "wb") as file_id:
            pickle.dump(self.c, file_id)

    def test_load(self):
        self.test_save()
        filename = "saved_state.p"
        with open(filename, "rb") as file_id:
            c = pickle.load(file_id)
        self.assertIsInstance(c, C)

    def test_deep_copy(self):
        c = copy.deepcopy(self.c)
        assert_allclose(c, self.c)
        self.assertNotEqual(id(c), id(self.c))

    def test_repr(self):
        self.assertEqual(str(self.a), "A")
        self.assertEqual(str(self.b), "B")
        self.assertEqual(str(self.c), "C")

    def test_at(self):
        c = C(torch.ones(2))
        np.add.at(c, [1], 1)
        self.assertIsInstance(c, C)

    def test_not_implemented(self):
        self.assertIs(self.c.nonsense(self.a), NotImplemented)

    def test_bool_return(self):
        self.assertIsInstance(np.equal(self.a, self.c), torch.Tensor)
        self.assertIs(np.equal(self.a, self.c).dtype, torch.uint8)
        self.assertIsInstance(torch.equal(self.a, self.c), bool)

    def test_mp(self):
        num_processes = 4
        # NOTE: this is required for the ``fork`` method to work
        self.a.share_memory_()
        self.c.share_memory_()
        processes = []
        queue = mp.Queue()
        for rank in range(num_processes):
            p = mp.Process(target=sub_process_func, args=(queue, self.a, self.c))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        for rank in range(num_processes):
            self.assertEqual(queue.get(), "%s = func(%s, %s)" % (type(self.c), type(self.a),  type(self.c)))

    def tearDown(self):
        pass


class TestSubclassFuncIn1InOut0Out1(unittest.TestCase):
    funcs = {"torch": torch.Tensor.abs, "numpy": np.abs}
    fargs = tuple()
    shape = (1,)
    sparse = False
    vector_function = False
    matrix_function = False

    @classmethod
    def setUpClass(cls):
        try:
            tensors = [torch.ones(cls.shape, dtype=torch.float64)]*6
            if cls.sparse:
                return
                for index in range(6):
                    i = torch.LongTensor([[0, 0]])
                    v = torch.FloatTensor([1.])
                    tensors[index] = torch.sparse.FloatTensor(i.t(), v, torch.Size([1, 1]))
            cls.funcs["torch"](*tensors[0:1], *cls.fargs)
        except RuntimeError:
            try:
                tensors = [torch.ones(cls.shape, dtype=torch.int64)]*6
                cls.funcs["torch"](*tensors[0:1], *cls.fargs)
            except RuntimeError:
                tensors = [torch.ones(cls.shape, dtype=torch.uint8)]*6
                cls.funcs["torch"](*tensors[0:1], *cls.fargs)
        cls.is_boolean = tensors[0].dtype == torch.uint8
        cls.a = A(tensors[0])
        cls.b = B(tensors[1])
        cls.c = C(tensors[2])
        cls.out_a = A(tensors[3])
        cls.out_b = B(tensors[4])
        cls.out_c = C(tensors[5])

    def setUp(self):
        if self.sparse:
            self.skipTest("RuntimeError: sparse tensors do not have strides")

    def test_func_a(self):
        if self.funcs["numpy"] is not None and isinstance(self.funcs["numpy"], np.ufunc):
            self.assertIsInstance(self.funcs["numpy"](self.a, *self.fargs), A)
        self.assertIsInstance(self.funcs["torch"](self.a, *self.fargs), A)

    def tearDown(self):
        pass


class TestSubclassFuncIn1InOut1Out1(TestSubclassFuncIn1InOut0Out1):

    def setUp(self):
        pass

    def test_func_a(self):
        super().test_func_a()
        if self.funcs["numpy"] is not None and isinstance(self.funcs["numpy"], np.ufunc):
            self.assertIsInstance(self.funcs["numpy"](self.a, *self.fargs, out=self.out_a), A)
        self.assertIsInstance(self.funcs["torch"](self.a, *self.fargs, out=self.out_a), A)

    def tearDown(self):
        pass


class TestSubclassFuncIn2InOut0Out1(unittest.TestCase):
    funcs = {"torch": torch.Tensor.add, "numpy": np.add}
    fargs = tuple()
    shape = (1,)
    sparse = False
    vector_function = False
    matrix_function = False

    @classmethod
    def setUpClass(cls):
        try:
            tensors = [torch.ones(cls.shape, dtype=torch.float64)]*6
            if cls.sparse:
                return
                # for index in range(6):
                #     i = torch.LongTensor([[0, 0]])
                #     v = torch.FloatTensor([1.])
                #     tensors[index] = torch.sparse.FloatTensor(i.t(), v, torch.Size([1, 1]))
            cls.funcs["torch"](*tensors[0:2], *cls.fargs)
        except RuntimeError:
            try:
                tensors = [torch.ones(cls.shape, dtype=torch.int64)]*6
                cls.funcs["torch"](*tensors[0:2], *cls.fargs)
            except RuntimeError:
                tensors = [torch.ones(cls.shape, dtype=torch.uint8)]*6
                cls.funcs["torch"](*tensors[0:2], *cls.fargs)
        cls.a = A(tensors[0])
        cls.b = B(tensors[1])
        cls.c = C(tensors[2])
        cls.out_a = A(tensors[3])
        cls.out_b = B(tensors[4])
        cls.out_c = C(tensors[5])

    def setUp(self):
        if self.sparse:
            self.skipTest("RuntimeError: sparse tensors do not have strides")

    def test_func_a_1(self):
        if not self.vector_function and not self.matrix_function:
            if self.funcs["numpy"] is not None and isinstance(self.funcs["numpy"], np.ufunc):
                self.assertIsInstance(self.funcs["numpy"](self.a, 1, *self.fargs), A)
                self.assertIsInstance(self.funcs["torch"](self.a, 1, *self.fargs), A)

    def test_func_1_a(self):
        if not self.vector_function and not self.matrix_function:
            if self.funcs["numpy"] is not None and isinstance(self.funcs["numpy"], np.ufunc):
                self.assertIsInstance(self.funcs["numpy"](1, self.a, *self.fargs), A)

    def test_func_a_b(self):
        if self.funcs["numpy"] is not None and isinstance(self.funcs["numpy"], np.ufunc):
            self.assertIsInstance(self.funcs["numpy"](self.a, self.b, *self.fargs), A)
        self.assertIsInstance(self.funcs["torch"](self.a, self.b, *self.fargs), A)

    def test_func_b_a(self):
        if self.funcs["numpy"] is not None and isinstance(self.funcs["numpy"], np.ufunc):
            self.assertIsInstance(self.funcs["numpy"](self.b, self.a, *self.fargs), B)
        self.assertIsInstance(self.funcs["torch"](self.b, self.a, *self.fargs), B)

    def test_func_c_a(self):
        if self.funcs["numpy"] is not None and isinstance(self.funcs["numpy"], np.ufunc):
            self.assertIsInstance(self.funcs["numpy"](self.c, self.a, *self.fargs), C)
        self.assertIsInstance(self.funcs["torch"](self.c, self.a, *self.fargs), C)

    def test_func_a_c(self):
        if self.funcs["numpy"] is not None and isinstance(self.funcs["numpy"], np.ufunc):
            self.assertIsInstance(self.funcs["numpy"](self.a, self.c, *self.fargs), C)
        self.assertIsInstance(self.funcs["torch"](self.a, self.c, *self.fargs), C)

    def tearDown(self):
        pass


class TestSubclassFuncIn2InOut1Out1(TestSubclassFuncIn2InOut0Out1):

    def test_func_a_1(self):
        super().test_func_a_1()
        if not self.vector_function and not self.matrix_function:
            if self.funcs["numpy"] is not None and isinstance(self.funcs["numpy"], np.ufunc):
                self.assertIsInstance(self.funcs["numpy"](self.a, 1, *self.fargs, out=self.out_a), A)
            self.assertIsInstance(self.funcs["torch"](self.a, 1, *self.fargs, out=self.out_a), A)

    def test_func_1_a(self):
        super().test_func_1_a()
        if not self.vector_function and not self.matrix_function:
            if self.funcs["numpy"] is not None and isinstance(self.funcs["numpy"], np.ufunc):
                self.assertIsInstance(self.funcs["numpy"](1, self.a, *self.fargs, out=self.out_a), A)

    def test_func_a_b(self):
        super().test_func_a_b()
        if self.funcs["numpy"] is not None and isinstance(self.funcs["numpy"], np.ufunc):
            self.assertIsInstance(self.funcs["numpy"](self.a, self.b, *self.fargs, out=self.out_a), A)
        self.assertIsInstance(self.funcs["torch"](self.a, self.b, *self.fargs, out=self.out_a), A)

    def test_func_b_a(self):
        super().test_func_b_a()
        if self.funcs["numpy"] is not None and isinstance(self.funcs["numpy"], np.ufunc):
            self.assertIsInstance(self.funcs["numpy"](self.b, self.a, *self.fargs, out=self.out_b), B)
        self.assertIsInstance(self.funcs["torch"](self.b, self.a, *self.fargs, out=self.out_b), B)

    def test_func_c_a(self):
        super().test_func_c_a()
        if self.funcs["numpy"] is not None and isinstance(self.funcs["numpy"], np.ufunc):
            self.assertIsInstance(self.funcs["numpy"](self.c, self.a, *self.fargs, out=self.out_c), C)
        self.assertIsInstance(self.funcs["torch"](self.c, self.a, *self.fargs, out=self.out_c), C)

    def test_func_a_c(self):
        super().test_func_a_c()
        if self.funcs["numpy"] is not None and isinstance(self.funcs["numpy"], np.ufunc):
            self.assertIsInstance(self.funcs["numpy"](self.a, self.c, *self.fargs, out=self.out_c), C)
        self.assertIsInstance(self.funcs["torch"](self.a, self.c, *self.fargs, out=self.out_c), C)

    def tearDown(self):
        pass


class Test___abs__(TestSubclassFuncIn1InOut1Out1): funcs = {"torch": torch.Tensor.__abs__, "numpy": np.abs}


class Test___add__(TestSubclassFuncIn2InOut1Out1): funcs = {"torch": torch.Tensor.__add__, "numpy": np.add}


class Test___and__(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.__and__, "numpy": np.bitwise_and}


class Test___div__(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.__div__, "numpy": np.divide}


class Test___eq__(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.__eq__, "numpy": np.equal}


class Test___ge__(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.__ge__, "numpy": np.greater_equal}


class Test___gt__(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.__gt__, "numpy": np.greater}


class Test___le__(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.__le__, "numpy": np.less_equal}


class Test___lt__(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.__lt__, "numpy": np.greater_equal}


class Test___ne__(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.__ne__, "numpy": np.not_equal}


class Test___neg__(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.__neg__, "numpy": np.negative}


class Test_abs(TestSubclassFuncIn1InOut1Out1): funcs = {"torch": torch.Tensor.abs, "numpy": np.abs}


class Test_add(TestSubclassFuncIn2InOut1Out1): funcs = {"torch": torch.Tensor.add, "numpy": np.add}


#Auto Test

class Test_abs(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.abs, "numpy": np.abs}



class Test_acos(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.acos, "numpy": np.arccos}



class Test_add(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.add, "numpy": np.add}



# class Test_addmv(TestSubclassFuncIn3InOut0Out1): funcs = {"torch": torch.Tensor.addmv, "numpy": None}



# class Test_addr(TestSubclassFuncIn3InOut0Out1): funcs = {"torch": torch.Tensor.addr, "numpy": None}



class Test_all(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.all, "numpy": None}



class Test_any(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.any, "numpy": None}



class Test_argmax(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.argmax, "numpy": np.argmax}



class Test_argmin(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.argmin, "numpy": np.argmin}



class Test_as_strided(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.as_strided, "numpy": None}
    fargs = ((1,), (1,))



class Test_asin(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.asin, "numpy": np.arcsin}



class Test_atan(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.atan, "numpy": np.arctan}



# class Test_baddbmm(TestSubclassFuncIn3InOut0Out1): funcs = {"torch": torch.Tensor.baddbmm, "numpy": None}



class Test_bernoulli(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.bernoulli, "numpy": None}



class Test_bincount(TestSubclassFuncIn2InOut0Out1):
    funcs = {"torch": torch.Tensor.bincount, "numpy": None}
    shape = (10,)



class Test_bmm(TestSubclassFuncIn2InOut0Out1):
    funcs = {"torch": torch.Tensor.bmm, "numpy": None}
    shape = (1, 1, 1)



class Test_ceil(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.ceil, "numpy": np.ceil}



class Test_clamp(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.clamp, "numpy": np.clip}
    fargs = (0, 100)



class Test_clamp_max(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.clamp_max, "numpy": None}
    fargs = (100,)



class Test_clamp_min(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.clamp_min, "numpy": None}
    fargs = (0,)



class Test_contiguous(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.contiguous, "numpy": None}



class Test_cos(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.cos, "numpy": np.cos}



class Test_cosh(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.cosh, "numpy": np.cosh}



class Test_cumsum(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.cumsum, "numpy": np.cumsum}
    fargs = (0,)



class Test_cumprod(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.cumprod, "numpy": np.cumprod}
    fargs = (0,)



class Test_det(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.det, "numpy": np.linalg.det}
    shape = (1, 1)



class Test_diag_embed(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.diag_embed, "numpy": None}



class Test_diagflat(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.diagflat, "numpy": np.diagflat}



class Test_diagonal(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.diagonal, "numpy": np.diagonal}
    shape = (1, 1)



class Test_div(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.div, "numpy": np.divide}



class Test_dot(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.dot, "numpy": np.dot}



class Test_resize(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.resize, "numpy": np.resize}
    shape = (2,)
    fargs = (2,)



class Test_erf(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.erf, "numpy": None}



class Test_erfc(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.erfc, "numpy": None}



class Test_exp(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.exp, "numpy": np.exp}



class Test_expm1(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.expm1, "numpy": np.expm1}



class Test_expand(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.expand, "numpy": None}
    fargs = (1, 1, 1)



class Test_expand_as(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.expand_as, "numpy": None}
    fargs = (torch.ones((1, 1),),)



class Test_flatten(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.flatten, "numpy": None}



# class Test_fill(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.fill, "numpy": np.ndarray.fill}



class Test_floor(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.floor, "numpy": np.floor}



class Test_ger(TestSubclassFuncIn2InOut0Out1):
    funcs = {"torch": torch.Tensor.ger, "numpy": None}
    vector_function = True



# class Test_gesv(TestSubclassFuncIn2InOut0Out2): funcs = {"torch": torch.Tensor.gesv, "numpy": None}



class Test_fft(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.fft, "numpy": None}
    shape = (1, 2)
    fargs = (1,)



class Test_ifft(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.ifft, "numpy": None}
    shape = (1, 2)
    fargs = (1,)



class Test_rfft(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.rfft, "numpy": None}
    shape = (1, 2)
    fargs = (1,)



class Test_irfft(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.irfft, "numpy": None}
    shape = (1, 2)
    fargs = (1,)



# class Test_index(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.index, "numpy": None}



# class Test_index_copy(TestSubclassFuncIn3InOut0Out1): funcs = {"torch": torch.Tensor.index_copy, "numpy": None}



class Test_index_put(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.index_put, "numpy": None}
    fargs = ((torch.zeros((0,), dtype=torch.long),), torch.zeros((0,), dtype=torch.double))



class Test_inverse(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.inverse, "numpy": np.linalg.inv}
    shape = (1, 1)



class Test_isclose(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.isclose, "numpy": np.isclose}



# class Test_kthvalue(TestSubclassFuncIn1InOut0Out2): funcs = {"torch": torch.Tensor.kthvalue, "numpy": None}



class Test_log(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.log, "numpy": np.log}



class Test_log10(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.log10, "numpy": np.log10}



class Test_log1p(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.log1p, "numpy": np.log1p}



class Test_log2(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.log2, "numpy": np.log2}



class Test_logdet(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.logdet, "numpy": None}
    shape = (1, 1)



class Test_log_softmax(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.log_softmax, "numpy": None}
    fargs = (0,)



class Test_logsumexp(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.logsumexp, "numpy": None}
    fargs = (0,)



class Test_matmul(TestSubclassFuncIn2InOut0Out1):
    funcs = {"torch": torch.Tensor.matmul, "numpy": np.matmul}
    shape = (1, 1)
    matrix_function = True



class Test_matrix_power(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.matrix_power, "numpy": None}
    shape = (1, 1)
    fargs = (1,)
    matrix_function = True



# class Test_max(TestSubclassFuncIn2InOut0Out2): funcs = {"torch": torch.Tensor.max, "numpy": np.max}



# class Test_max_values(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.max_values, "numpy": None}



class Test_mean(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.mean, "numpy": None}



# class Test_median(TestSubclassFuncIn1InOut0Out2): funcs = {"torch": torch.Tensor.median, "numpy": np.median}



# class Test_min(TestSubclassFuncIn2InOut0Out2): funcs = {"torch": torch.Tensor.min, "numpy": np.min}



# class Test_min_values(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.min_values, "numpy": None}



class Test_mm(TestSubclassFuncIn2InOut0Out1):
    funcs = {"torch": torch.Tensor.mm, "numpy": None}
    shape = (1, 1)



# class Test_mode(TestSubclassFuncIn1InOut0Out2): funcs = {"torch": torch.Tensor.mode, "numpy": None}



class Test_mul(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.mul, "numpy": None}



# class Test_mv(TestSubclassFuncIn2InOut0Out1):
#     funcs = {"torch": torch.Tensor.mv, "numpy": None}
#     shape = (1, 1)



class Test_mvlgamma(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.mvlgamma, "numpy": None}
    fargs = (1,)



class Test_narrow_copy(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.narrow_copy, "numpy": None}
    fargs = (0, 0, 1)



class Test_narrow(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.narrow, "numpy": None}
    fargs = (0, 0, 1)



class Test_permute(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.permute, "numpy": np.transpose}
    fargs = (0, 1)
    shape = (1, 1)



# class Test_pin_memory(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.pin_memory, "numpy": None}



class Test_pinverse(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.pinverse, "numpy": np.linalg.inv}
    shape = (1, 1)



class Test_repeat(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.repeat, "numpy": None}
    fargs = (1, )



class Test_reshape(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.reshape, "numpy": np.reshape}
    shape = (1, 1)
    fargs = (-1, )




class Test_reshape_as(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.reshape_as, "numpy": None}



class Test_round(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.round, "numpy": np.round}



class Test_relu(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.relu, "numpy": None}



class Test_prelu(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.prelu, "numpy": None}



# class Test_prelu_backward(TestSubclassFuncIn3InOut0Out2): funcs = {"torch": torch.Tensor.prelu_backward, "numpy": None}



class Test_hardshrink(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.hardshrink, "numpy": None}



# class Test_hardshrink_backward(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.hardshrink_backward, "numpy": None}



class Test_rsqrt(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.rsqrt, "numpy": None}



class Test_select(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.select, "numpy": np.select}
    fargs = (0, 0)



class Test_sigmoid(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.sigmoid, "numpy": None}



class Test_sin(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.sin, "numpy": np.sin}



class Test_sinh(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.sinh, "numpy": np.sinh}



class Test_detach(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.detach, "numpy": None}



# class Test_slice(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.slice, "numpy": None}



# class Test_slogdet(TestSubclassFuncIn1InOut0Out2): funcs = {"torch": torch.Tensor.slogdet, "numpy": None}



class Test_smm(TestSubclassFuncIn2InOut0Out1):
    funcs = {"torch": torch.Tensor.smm, "numpy": None}
    shape = (1, 1)
    sparse = True



class Test_softmax(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.softmax, "numpy": None}
    fargs = (0, )



class Test_squeeze(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.squeeze, "numpy": np.squeeze}



# class Test_sspaddmm(TestSubclassFuncIn3InOut0Out1): funcs = {"torch": torch.Tensor.sspaddmm, "numpy": None}



class Test_stft(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.stft, "numpy": None}
    shape = (100)
    fargs = (25,)



class Test_sum(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.sum, "numpy": np.sum}
    fargs = (0,)



# class Test_sum_to_size(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.sum_to_size, "numpy": None}



class Test_sqrt(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.sqrt, "numpy": np.sqrt}



class Test_std(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.std, "numpy": np.std}



class Test_prod(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.prod, "numpy": np.prod}



class Test_t(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.t, "numpy": np.ndarray.T}



class Test_tan(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.tan, "numpy": np.tan}



class Test_tanh(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.tanh, "numpy": np.tanh}



class Test_transpose(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.transpose, "numpy": np.transpose}
    shape = (1, 1)
    fargs = (0, 1)



class Test_flip(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.flip, "numpy": np.flip}
    shape = (1, 1)
    fargs = (0, 1)



class Test_roll(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.roll, "numpy": np.roll}
    fargs = (1, 0)



class Test_rot90(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.rot90, "numpy": np.rot90}
    shape = (1, 1)
    fargs = (0, (0, 1))



class Test_trunc(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.trunc, "numpy": np.trunc}



class Test_type_as(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.type_as, "numpy": np.ndarray.astype}



class Test_unsqueeze(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.unsqueeze, "numpy": None}
    fargs = (0,)



class Test_var(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.var, "numpy": np.var}



class Test_view_as(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.view_as, "numpy": np.ndarray.view}



# class Test_where(TestSubclassFuncIn3InOut0Out1): funcs = {"torch": torch.Tensor.where, "numpy": np.where}



class Test_norm(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.norm, "numpy": np.linalg.norm}



class Test_clone(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.clone, "numpy": np.copy}



class Test_resize_as(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.resize_as, "numpy": None}



class Test_pow(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.pow, "numpy": np.power}



# class Test_zero(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.zero, "numpy": None}



class Test_sub(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.sub, "numpy": None}



# class Test_addmm(TestSubclassFuncIn3InOut0Out1): funcs = {"torch": torch.Tensor.addmm, "numpy": None}



class Test_sparse_resize(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.sparse_resize, "numpy": None}
    sparse = True



class Test_sparse_resize_and_clear(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.sparse_resize_and_clear, "numpy": None}
    sparse = True



class Test_sparse_mask(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.sparse_mask, "numpy": None}
    sparse = True



class Test_to_dense(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.to_dense, "numpy": None}
    sparse = True



class Test_coalesce(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.coalesce, "numpy": None}
    sparse = True



class Test_indices(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.indices, "numpy": np.indices}
    sparse = True



class Test_values(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.values, "numpy": None}
    sparse = True



class Test_to_sparse(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.to_sparse, "numpy": None}



class Test_to(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.to, "numpy": None}



class Test_set(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.set, "numpy": None}



# class Test_masked_fill(TestSubclassFuncIn3InOut0Out1): funcs = {"torch": torch.Tensor.masked_fill, "numpy": None}



# class Test_masked_scatter(TestSubclassFuncIn3InOut0Out1): funcs = {"torch": torch.Tensor.masked_scatter, "numpy": None}



class Test_view(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.view, "numpy": np.ndarray.view}
    fargs = ((1,),)



# class Test_put(TestSubclassFuncIn3InOut0Out1): funcs = {"torch": torch.Tensor.put, "numpy": np.put}



# class Test_index_add(TestSubclassFuncIn3InOut0Out1): funcs = {"torch": torch.Tensor.index_add, "numpy": None}



# class Test_index_fill(TestSubclassFuncIn3InOut0Out1): funcs = {"torch": torch.Tensor.index_fill, "numpy": None}



# class Test_scatter(TestSubclassFuncIn3InOut0Out1): funcs = {"torch": torch.Tensor.scatter, "numpy": None}



# class Test_scatter_add(TestSubclassFuncIn3InOut0Out1): funcs = {"torch": torch.Tensor.scatter_add, "numpy": None}



class Test_lt(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.lt, "numpy": np.less}



class Test_gt(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.gt, "numpy": np.greater}



class Test_le(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.le, "numpy": np.less_equal}



class Test_ge(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.ge, "numpy": np.greater_equal}



class Test_eq(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.eq, "numpy": np.equal}



class Test_ne(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.ne, "numpy": np.not_equal}



class Test___and__(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.__and__, "numpy": np.bitwise_and}



class Test___iand__(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.__iand__, "numpy": np.bitwise_and}



class Test___or__(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.__or__, "numpy": np.bitwise_or}



class Test___ior__(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.__ior__, "numpy": np.bitwise_or}



class Test___xor__(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.__xor__, "numpy": np.bitwise_xor}



class Test___ixor__(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.__ixor__, "numpy": np.bitwise_xor}



class Test___lshift__(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.__lshift__, "numpy": None}
    fargs = (1,)



class Test___ilshift__(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.__ilshift__, "numpy": None}
    fargs = (1,)



class Test___rshift__(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.__rshift__, "numpy": None}
    fargs = (1,)



class Test___irshift__(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.__irshift__, "numpy": None}
    fargs = (1,)



class Test_lgamma(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.lgamma, "numpy": None}



class Test_atan2(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.atan2, "numpy": None}



class Test_tril(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.tril, "numpy": np.tril}
    shape = (1, 1)



class Test_triu(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.triu, "numpy": np.triu}
    shape = (1, 1)



class Test_digamma(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.digamma, "numpy": None}



class Test_polygamma(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.polygamma, "numpy": None}
    fargs = (1,)



class Test_erfinv(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.erfinv, "numpy": None}



class Test_frac(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.frac, "numpy": None}



class Test_renorm(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.renorm, "numpy": None}
    shape = (1, 1)
    fargs = (1, 0, 5)



class Test_reciprocal(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.reciprocal, "numpy": np.reciprocal}



class Test_neg(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.neg, "numpy": np.negative}



class Test_lerp(TestSubclassFuncIn2InOut0Out1):
    funcs = {"torch": torch.Tensor.lerp, "numpy": None}
    fargs = (0.5,)



class Test_sign(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.sign, "numpy": np.sign}



class Test_fmod(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.fmod, "numpy": np.fmod}



class Test_remainder(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.remainder, "numpy": np.remainder}



# class Test_addbmm(TestSubclassFuncIn3InOut0Out1): funcs = {"torch": torch.Tensor.addbmm, "numpy": None}



# class Test_addcmul(TestSubclassFuncIn3InOut0Out1): funcs = {"torch": torch.Tensor.addcmul, "numpy": None}



# class Test_addcdiv(TestSubclassFuncIn3InOut0Out1): funcs = {"torch": torch.Tensor.addcdiv, "numpy": None}



# class Test_random(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.random, "numpy": np.random}



# class Test_uniform(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.uniform, "numpy": None}



# class Test_normal(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.normal, "numpy": None}



class Test_cauchy(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.cauchy, "numpy": None}



# class Test_log_normal(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.log_normal, "numpy": None}



class Test_exponential(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.exponential, "numpy": None}



# class Test_geometric(TestSubclassFuncIn1InOut0Out1):
#     funcs = {"torch": torch.Tensor.geometric, "numpy": None}
#     fargs = (0.5,)



class Test_diag(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.diag, "numpy": np.diag}



class Test_cross(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.cross, "numpy": np.cross}
    matrix_function = True
    shape = (3,)
    fargs = (torch.ones((3,), dtype=torch.double), )



class Test_trace(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.trace, "numpy": np.trace}
    shape = (1, 1)
    matrix_function = True



class Test_take(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.take, "numpy": np.take}
    fargs = (torch.zeros((1,), dtype=torch.long))



class Test_index_select(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.index_select, "numpy": None}
    fargs = (0, torch.zeros((1,), dtype=torch.long))



class Test_masked_select(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.masked_select, "numpy": None}



class Test_nonzero(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.nonzero, "numpy": np.nonzero}



class Test_gather(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.gather, "numpy": None}
    fargs = (0,torch.zeros((1,), dtype=torch.long))



# class Test_gels(TestSubclassFuncIn2InOut0Out2): funcs = {"torch": torch.Tensor.gels, "numpy": None}



# class Test_trtrs(TestSubclassFuncIn2InOut0Out2): funcs = {"torch": torch.Tensor.trtrs, "numpy": None}



# class Test_symeig(TestSubclassFuncIn1InOut0Out2): funcs = {"torch": torch.Tensor.symeig, "numpy": None}



# class Test_eig(TestSubclassFuncIn1InOut0Out2): funcs = {"torch": torch.Tensor.eig, "numpy": None}



# class Test_svd(TestSubclassFuncIn1InOut0Out3): funcs = {"torch": torch.Tensor.svd, "numpy": None}



class Test_cholesky(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.cholesky, "numpy": None}
    shape = (1, 1)



class Test_cholesky_solve(TestSubclassFuncIn2InOut0Out1):
    funcs = {"torch": torch.Tensor.cholesky_solve, "numpy": None}
    shape = (1, 1)



class Test_potri(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.potri, "numpy": None}
    shape = (1, 1)



# class Test_pstrf(TestSubclassFuncIn1InOut0Out2): funcs = {"torch": torch.Tensor.pstrf, "numpy": None}



# class Test_qr(TestSubclassFuncIn1InOut0Out2): funcs = {"torch": torch.Tensor.qr, "numpy": None}



# class Test_geqrf(TestSubclassFuncIn1InOut0Out2): funcs = {"torch": torch.Tensor.geqrf, "numpy": None}



class Test_orgqr(TestSubclassFuncIn2InOut0Out1):
    funcs = {"torch": torch.Tensor.orgqr, "numpy": None}
    shape = (1, 1)



# class Test_ormqr(TestSubclassFuncIn3InOut0Out1): funcs = {"torch": torch.Tensor.ormqr, "numpy": None}



# class Test_btrifact(TestSubclassFuncIn1InOut0Out2): funcs = {"torch": torch.Tensor.btrifact, "numpy": None}



# class Test_btrifact_with_info(TestSubclassFuncIn1InOut0Out3): funcs = {"torch": torch.Tensor.btrifact_with_info, "numpy": None}



# class Test_btrisolve(TestSubclassFuncIn3InOut0Out1): funcs = {"torch": torch.Tensor.btrisolve, "numpy": None}



class Test_multinomial(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.multinomial, "numpy": None}
    fargs = (1, )



class Test_dist(TestSubclassFuncIn2InOut0Out1): funcs = {"torch": torch.Tensor.dist, "numpy": None}



class Test_histc(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.histc, "numpy": None}



# class Test_sort(TestSubclassFuncIn1InOut0Out2): funcs = {"torch": torch.Tensor.sort, "numpy": np.sort}



class Test_argsort(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.argsort, "numpy": np.argsort}



# class Test_topk(TestSubclassFuncIn1InOut0Out2): funcs = {"torch": torch.Tensor.topk, "numpy": None}



class Test_unfold(TestSubclassFuncIn1InOut0Out1):
    funcs = {"torch": torch.Tensor.unfold, "numpy": None}
    fargs = (0, 1, 1)



# class Test_alias(TestSubclassFuncIn1InOut0Out1): funcs = {"torch": torch.Tensor.alias, "numpy": None}



# Ufunc requirments
#  - First input is a Tensor
#  - Returns at least one Tensor


# Not supported
