import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/vec_test_all_types_DEFAULT"


class TestMemory_0(TestCase):
    cpp_name = "Memory/0"

    def test_UnAlignedLoadStore(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnAlignedLoadStore")


class TestMemory_1(TestCase):
    cpp_name = "Memory/1"

    def test_UnAlignedLoadStore(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnAlignedLoadStore")


class TestMemory_2(TestCase):
    cpp_name = "Memory/2"

    def test_UnAlignedLoadStore(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnAlignedLoadStore")


class TestMemory_3(TestCase):
    cpp_name = "Memory/3"

    def test_UnAlignedLoadStore(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnAlignedLoadStore")


class TestMemory_4(TestCase):
    cpp_name = "Memory/4"

    def test_UnAlignedLoadStore(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnAlignedLoadStore")


class TestMemory_5(TestCase):
    cpp_name = "Memory/5"

    def test_UnAlignedLoadStore(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnAlignedLoadStore")


class TestMemory_6(TestCase):
    cpp_name = "Memory/6"

    def test_UnAlignedLoadStore(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnAlignedLoadStore")


class TestMemory_7(TestCase):
    cpp_name = "Memory/7"

    def test_UnAlignedLoadStore(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnAlignedLoadStore")


class TestMemory_8(TestCase):
    cpp_name = "Memory/8"

    def test_UnAlignedLoadStore(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnAlignedLoadStore")


class TestSignManipulation_0(TestCase):
    cpp_name = "SignManipulation/0"

    def test_Absolute(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Absolute")

    def test_Negate(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Negate")


class TestSignManipulation_1(TestCase):
    cpp_name = "SignManipulation/1"

    def test_Absolute(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Absolute")

    def test_Negate(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Negate")


class TestSignManipulation_2(TestCase):
    cpp_name = "SignManipulation/2"

    def test_Absolute(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Absolute")

    def test_Negate(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Negate")


class TestSignManipulation_3(TestCase):
    cpp_name = "SignManipulation/3"

    def test_Absolute(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Absolute")

    def test_Negate(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Negate")


class TestSignManipulation_4(TestCase):
    cpp_name = "SignManipulation/4"

    def test_Absolute(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Absolute")

    def test_Negate(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Negate")


class TestSignManipulation_5(TestCase):
    cpp_name = "SignManipulation/5"

    def test_Absolute(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Absolute")

    def test_Negate(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Negate")


class TestSignManipulation_6(TestCase):
    cpp_name = "SignManipulation/6"

    def test_Absolute(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Absolute")

    def test_Negate(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Negate")


class TestRounding_0(TestCase):
    cpp_name = "Rounding/0"

    def test_Round(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Round")

    def test_Ceil(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Ceil")

    def test_Floor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Floor")

    def test_Trunc(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Trunc")


class TestRounding_1(TestCase):
    cpp_name = "Rounding/1"

    def test_Round(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Round")

    def test_Ceil(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Ceil")

    def test_Floor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Floor")

    def test_Trunc(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Trunc")


class TestSqrtAndReciprocal_0(TestCase):
    cpp_name = "SqrtAndReciprocal/0"

    def test_Sqrt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Sqrt")


class TestSqrtAndReciprocal_1(TestCase):
    cpp_name = "SqrtAndReciprocal/1"

    def test_Sqrt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Sqrt")


class TestSqrtAndReciprocal_2(TestCase):
    cpp_name = "SqrtAndReciprocal/2"

    def test_Sqrt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Sqrt")


class TestSqrtAndReciprocal_3(TestCase):
    cpp_name = "SqrtAndReciprocal/3"

    def test_Sqrt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Sqrt")


class TestSqrtAndReciprocalReal_0(TestCase):
    cpp_name = "SqrtAndReciprocalReal/0"

    def test_RSqrt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RSqrt")

    def test_Reciprocal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Reciprocal")


class TestSqrtAndReciprocalReal_1(TestCase):
    cpp_name = "SqrtAndReciprocalReal/1"

    def test_RSqrt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RSqrt")

    def test_Reciprocal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Reciprocal")


class TestFractionAndRemainderReal_0(TestCase):
    cpp_name = "FractionAndRemainderReal/0"

    def test_Frac(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Frac")

    def test_Fmod(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Fmod")


class TestFractionAndRemainderReal_1(TestCase):
    cpp_name = "FractionAndRemainderReal/1"

    def test_Frac(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Frac")

    def test_Fmod(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Fmod")


class TestTrigonometric_0(TestCase):
    cpp_name = "Trigonometric/0"

    def test_Sin(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Sin")

    def test_Cos(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Cos")

    def test_Tan(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Tan")


class TestTrigonometric_1(TestCase):
    cpp_name = "Trigonometric/1"

    def test_Sin(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Sin")

    def test_Cos(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Cos")

    def test_Tan(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Tan")


class TestHyperbolic_0(TestCase):
    cpp_name = "Hyperbolic/0"

    def test_Tanh(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Tanh")

    def test_Sinh(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Sinh")

    def test_Cosh(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Cosh")


class TestHyperbolic_1(TestCase):
    cpp_name = "Hyperbolic/1"

    def test_Tanh(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Tanh")

    def test_Sinh(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Sinh")

    def test_Cosh(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Cosh")


class TestInverseTrigonometric_0(TestCase):
    cpp_name = "InverseTrigonometric/0"

    def test_Asin(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Asin")

    def test_ACos(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ACos")

    def test_ATan(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ATan")


class TestInverseTrigonometric_1(TestCase):
    cpp_name = "InverseTrigonometric/1"

    def test_Asin(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Asin")

    def test_ACos(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ACos")

    def test_ATan(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ATan")


class TestInverseTrigonometric_2(TestCase):
    cpp_name = "InverseTrigonometric/2"

    def test_Asin(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Asin")

    def test_ACos(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ACos")

    def test_ATan(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ATan")


class TestInverseTrigonometric_3(TestCase):
    cpp_name = "InverseTrigonometric/3"

    def test_Asin(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Asin")

    def test_ACos(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ACos")

    def test_ATan(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ATan")


class TestLogarithm_0(TestCase):
    cpp_name = "Logarithm/0"

    def test_Log(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Log")

    def test_Log10(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Log10")


class TestLogarithm_1(TestCase):
    cpp_name = "Logarithm/1"

    def test_Log(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Log")

    def test_Log10(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Log10")


class TestLogarithm_2(TestCase):
    cpp_name = "Logarithm/2"

    def test_Log(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Log")

    def test_Log10(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Log10")


class TestLogarithm_3(TestCase):
    cpp_name = "Logarithm/3"

    def test_Log(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Log")

    def test_Log10(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Log10")


class TestLogarithmReals_0(TestCase):
    cpp_name = "LogarithmReals/0"

    def test_Log2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Log2")

    def test_Log1p(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Log1p")


class TestLogarithmReals_1(TestCase):
    cpp_name = "LogarithmReals/1"

    def test_Log2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Log2")

    def test_Log1p(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Log1p")


class TestExponents_0(TestCase):
    cpp_name = "Exponents/0"

    def test_Exp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Exp")

    def test_Expm1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Expm1")


class TestExponents_1(TestCase):
    cpp_name = "Exponents/1"

    def test_Exp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Exp")

    def test_Expm1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Expm1")


class TestErrorFunctions_0(TestCase):
    cpp_name = "ErrorFunctions/0"

    def test_Erf(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Erf")

    def test_Erfc(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Erfc")

    def test_Erfinv(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Erfinv")


class TestErrorFunctions_1(TestCase):
    cpp_name = "ErrorFunctions/1"

    def test_Erf(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Erf")

    def test_Erfc(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Erfc")

    def test_Erfinv(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Erfinv")


class TestNan_0(TestCase):
    cpp_name = "Nan/0"

    def test_IsNan(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IsNan")


class TestNan_1(TestCase):
    cpp_name = "Nan/1"

    def test_IsNan(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IsNan")


class TestLGamma_0(TestCase):
    cpp_name = "LGamma/0"

    def test_LGamma(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LGamma")


class TestLGamma_1(TestCase):
    cpp_name = "LGamma/1"

    def test_LGamma(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LGamma")


class TestInverseTrigonometricReal_0(TestCase):
    cpp_name = "InverseTrigonometricReal/0"

    def test_ATan2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ATan2")


class TestInverseTrigonometricReal_1(TestCase):
    cpp_name = "InverseTrigonometricReal/1"

    def test_ATan2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ATan2")


class TestPow_0(TestCase):
    cpp_name = "Pow/0"

    def test_Pow(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Pow")


class TestPow_1(TestCase):
    cpp_name = "Pow/1"

    def test_Pow(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Pow")


class TestRealTests_0(TestCase):
    cpp_name = "RealTests/0"

    def test_Hypot(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Hypot")

    def test_NextAfter(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NextAfter")


class TestRealTests_1(TestCase):
    cpp_name = "RealTests/1"

    def test_Hypot(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Hypot")

    def test_NextAfter(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NextAfter")


class TestInterleave_0(TestCase):
    cpp_name = "Interleave/0"

    def test_Interleave(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Interleave")

    def test_DeInterleave(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DeInterleave")


class TestInterleave_1(TestCase):
    cpp_name = "Interleave/1"

    def test_Interleave(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Interleave")

    def test_DeInterleave(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DeInterleave")


class TestInterleave_2(TestCase):
    cpp_name = "Interleave/2"

    def test_Interleave(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Interleave")

    def test_DeInterleave(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DeInterleave")


class TestInterleave_3(TestCase):
    cpp_name = "Interleave/3"

    def test_Interleave(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Interleave")

    def test_DeInterleave(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DeInterleave")


class TestInterleave_4(TestCase):
    cpp_name = "Interleave/4"

    def test_Interleave(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Interleave")

    def test_DeInterleave(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DeInterleave")


class TestArithmetics_0(TestCase):
    cpp_name = "Arithmetics/0"

    def test_Plus(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Plus")

    def test_Minus(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Minus")

    def test_Multiplication(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Multiplication")

    def test_Division(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Division")


class TestArithmetics_1(TestCase):
    cpp_name = "Arithmetics/1"

    def test_Plus(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Plus")

    def test_Minus(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Minus")

    def test_Multiplication(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Multiplication")

    def test_Division(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Division")


class TestArithmetics_2(TestCase):
    cpp_name = "Arithmetics/2"

    def test_Plus(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Plus")

    def test_Minus(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Minus")

    def test_Multiplication(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Multiplication")

    def test_Division(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Division")


class TestArithmetics_3(TestCase):
    cpp_name = "Arithmetics/3"

    def test_Plus(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Plus")

    def test_Minus(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Minus")

    def test_Multiplication(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Multiplication")

    def test_Division(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Division")


class TestArithmetics_4(TestCase):
    cpp_name = "Arithmetics/4"

    def test_Plus(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Plus")

    def test_Minus(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Minus")

    def test_Multiplication(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Multiplication")

    def test_Division(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Division")


class TestArithmetics_5(TestCase):
    cpp_name = "Arithmetics/5"

    def test_Plus(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Plus")

    def test_Minus(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Minus")

    def test_Multiplication(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Multiplication")

    def test_Division(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Division")


class TestArithmetics_6(TestCase):
    cpp_name = "Arithmetics/6"

    def test_Plus(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Plus")

    def test_Minus(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Minus")

    def test_Multiplication(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Multiplication")

    def test_Division(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Division")


class TestBitwise_0(TestCase):
    cpp_name = "Bitwise/0"

    def test_BitAnd(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitAnd")

    def test_BitOr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitOr")

    def test_BitXor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitXor")


class TestBitwise_1(TestCase):
    cpp_name = "Bitwise/1"

    def test_BitAnd(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitAnd")

    def test_BitOr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitOr")

    def test_BitXor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitXor")


class TestBitwise_2(TestCase):
    cpp_name = "Bitwise/2"

    def test_BitAnd(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitAnd")

    def test_BitOr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitOr")

    def test_BitXor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitXor")


class TestBitwise_3(TestCase):
    cpp_name = "Bitwise/3"

    def test_BitAnd(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitAnd")

    def test_BitOr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitOr")

    def test_BitXor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitXor")


class TestBitwise_4(TestCase):
    cpp_name = "Bitwise/4"

    def test_BitAnd(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitAnd")

    def test_BitOr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitOr")

    def test_BitXor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitXor")


class TestBitwise_5(TestCase):
    cpp_name = "Bitwise/5"

    def test_BitAnd(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitAnd")

    def test_BitOr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitOr")

    def test_BitXor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitXor")


class TestBitwise_6(TestCase):
    cpp_name = "Bitwise/6"

    def test_BitAnd(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitAnd")

    def test_BitOr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitOr")

    def test_BitXor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitXor")


class TestComparison_0(TestCase):
    cpp_name = "Comparison/0"

    def test_Equal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Equal")

    def test_NotEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NotEqual")

    def test_Greater(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Greater")

    def test_Less(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Less")

    def test_GreaterEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GreaterEqual")

    def test_LessEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LessEqual")


class TestComparison_1(TestCase):
    cpp_name = "Comparison/1"

    def test_Equal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Equal")

    def test_NotEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NotEqual")

    def test_Greater(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Greater")

    def test_Less(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Less")

    def test_GreaterEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GreaterEqual")

    def test_LessEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LessEqual")


class TestComparison_2(TestCase):
    cpp_name = "Comparison/2"

    def test_Equal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Equal")

    def test_NotEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NotEqual")

    def test_Greater(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Greater")

    def test_Less(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Less")

    def test_GreaterEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GreaterEqual")

    def test_LessEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LessEqual")


class TestComparison_3(TestCase):
    cpp_name = "Comparison/3"

    def test_Equal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Equal")

    def test_NotEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NotEqual")

    def test_Greater(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Greater")

    def test_Less(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Less")

    def test_GreaterEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GreaterEqual")

    def test_LessEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LessEqual")


class TestComparison_4(TestCase):
    cpp_name = "Comparison/4"

    def test_Equal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Equal")

    def test_NotEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NotEqual")

    def test_Greater(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Greater")

    def test_Less(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Less")

    def test_GreaterEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GreaterEqual")

    def test_LessEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LessEqual")


class TestMinMax_0(TestCase):
    cpp_name = "MinMax/0"

    def test_Minimum(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Minimum")

    def test_Maximum(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Maximum")

    def test_ClampMin(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ClampMin")

    def test_ClampMax(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ClampMax")

    def test_Clamp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Clamp")


class TestMinMax_1(TestCase):
    cpp_name = "MinMax/1"

    def test_Minimum(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Minimum")

    def test_Maximum(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Maximum")

    def test_ClampMin(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ClampMin")

    def test_ClampMax(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ClampMax")

    def test_Clamp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Clamp")


class TestMinMax_2(TestCase):
    cpp_name = "MinMax/2"

    def test_Minimum(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Minimum")

    def test_Maximum(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Maximum")

    def test_ClampMin(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ClampMin")

    def test_ClampMax(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ClampMax")

    def test_Clamp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Clamp")


class TestMinMax_3(TestCase):
    cpp_name = "MinMax/3"

    def test_Minimum(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Minimum")

    def test_Maximum(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Maximum")

    def test_ClampMin(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ClampMin")

    def test_ClampMax(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ClampMax")

    def test_Clamp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Clamp")


class TestMinMax_4(TestCase):
    cpp_name = "MinMax/4"

    def test_Minimum(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Minimum")

    def test_Maximum(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Maximum")

    def test_ClampMin(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ClampMin")

    def test_ClampMax(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ClampMax")

    def test_Clamp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Clamp")


class TestBitwiseFloatsAdditional_0(TestCase):
    cpp_name = "BitwiseFloatsAdditional/0"

    def test_ZeroMask(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ZeroMask")

    def test_Convert(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Convert")

    def test_Fmadd(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Fmadd")

    def test_Blendv(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Blendv")


class TestBitwiseFloatsAdditional_1(TestCase):
    cpp_name = "BitwiseFloatsAdditional/1"

    def test_ZeroMask(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ZeroMask")

    def test_Convert(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Convert")

    def test_Fmadd(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Fmadd")

    def test_Blendv(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Blendv")


class TestBitwiseFloatsAdditional2_0(TestCase):
    cpp_name = "BitwiseFloatsAdditional2/0"

    def test_Blend(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Blend")

    def test_Set(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Set")


class TestBitwiseFloatsAdditional2_1(TestCase):
    cpp_name = "BitwiseFloatsAdditional2/1"

    def test_Blend(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Blend")

    def test_Set(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Set")


class TestBitwiseFloatsAdditional2_2(TestCase):
    cpp_name = "BitwiseFloatsAdditional2/2"

    def test_Blend(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Blend")

    def test_Set(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Set")


class TestBitwiseFloatsAdditional2_3(TestCase):
    cpp_name = "BitwiseFloatsAdditional2/3"

    def test_Blend(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Blend")

    def test_Set(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Set")


class TestRangeFactories_0(TestCase):
    cpp_name = "RangeFactories/0"

    def test_Arange(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Arange")


class TestRangeFactories_1(TestCase):
    cpp_name = "RangeFactories/1"

    def test_Arange(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Arange")


class TestRangeFactories_2(TestCase):
    cpp_name = "RangeFactories/2"

    def test_Arange(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Arange")


class TestRangeFactories_3(TestCase):
    cpp_name = "RangeFactories/3"

    def test_Arange(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Arange")


class TestRangeFactories_4(TestCase):
    cpp_name = "RangeFactories/4"

    def test_Arange(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Arange")


class TestRangeFactories_5(TestCase):
    cpp_name = "RangeFactories/5"

    def test_Arange(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Arange")


class TestRangeFactories_6(TestCase):
    cpp_name = "RangeFactories/6"

    def test_Arange(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Arange")


class TestComplexTests(TestCase):
    cpp_name = "ComplexTests"

    def test_TestComplexFloatImagRealConj(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestComplexFloatImagRealConj")


class TestQuantizationTests_0(TestCase):
    cpp_name = "QuantizationTests/0"

    def test_Quantize(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Quantize")

    def test_DeQuantize(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DeQuantize")

    def test_ReQuantizeFromInt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReQuantizeFromInt")

    def test_WideningSubtract(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "WideningSubtract")

    def test_Relu(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Relu")

    def test_Relu6(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Relu6")


class TestQuantizationTests_1(TestCase):
    cpp_name = "QuantizationTests/1"

    def test_Quantize(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Quantize")

    def test_DeQuantize(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DeQuantize")

    def test_ReQuantizeFromInt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReQuantizeFromInt")

    def test_WideningSubtract(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "WideningSubtract")

    def test_Relu(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Relu")

    def test_Relu6(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Relu6")


class TestQuantizationTests_2(TestCase):
    cpp_name = "QuantizationTests/2"

    def test_Quantize(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Quantize")

    def test_DeQuantize(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DeQuantize")

    def test_ReQuantizeFromInt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReQuantizeFromInt")

    def test_WideningSubtract(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "WideningSubtract")

    def test_Relu(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Relu")

    def test_Relu6(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Relu6")


class TestFunctionalTests_0(TestCase):
    cpp_name = "FunctionalTests/0"

    def test_Map(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Map")


class TestFunctionalTests_1(TestCase):
    cpp_name = "FunctionalTests/1"

    def test_Map(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Map")


class TestFunctionalTests_2(TestCase):
    cpp_name = "FunctionalTests/2"

    def test_Map(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Map")


class TestFunctionalTests_3(TestCase):
    cpp_name = "FunctionalTests/3"

    def test_Map(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Map")


class TestFunctionalTests_4(TestCase):
    cpp_name = "FunctionalTests/4"

    def test_Map(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Map")


class TestFunctionalBF16Tests_0(TestCase):
    cpp_name = "FunctionalBF16Tests/0"

    def test_Reduce(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Reduce")

    def test_Map(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Map")


if __name__ == "__main__":
    run_tests()
