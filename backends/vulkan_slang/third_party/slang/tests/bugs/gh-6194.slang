//TEST:SIMPLE(filecheck=CHECK): -target spirv
//CHECK: OpEntryPoint

interface IMLP
{
    associatedtype PrecisionType : __BuiltinFloatingPointType;
}
static const int kSize = 4;
float runAndSum<Pass0 : IMLP, Pass1 : IMLP>(float base)
{
    typealias Pass0Precision = Pass0::PrecisionType;
    typealias WARCastPass0 = Pass0Precision;
    typealias WARActualTypePass0 = half;

    typealias Pass1Precision = Pass1::PrecisionType;
    typealias WARCastPass1 = Pass1Precision;
    typealias WARActualTypePass1 = half;

    Pass0Precision pass0_inputs[kSize];
    for (uint i = 0; i < kSize; ++i)
    {
        // pass0_inputs[i] = (base + float(i)); // Not working, WAR below
        pass0_inputs[i] = (WARCastPass0)(base + float(i));
    }

    Pass0Precision pass0_outputs[kSize] = pass0_inputs;

    Pass1Precision pass1_inputs[kSize];

    for (uint i = 0; i < kSize; ++i)
    {
        // Fires SLANG_ASSERT(!"Unhandled comparison"); in slang-ir.cpp, _isTypeOperandEqual
        pass1_inputs[i] = __realCast<Pass1Precision>(pass0_outputs[i]);

        // Working
        //pass1_inputs[i] = (WARCastPass1)__realCast<WARActualTypePass0>(pass0_outputs[i]);
    }

    float result = 0;
    for (uint i = 0; i < kSize; ++i)
    {
        result += __realCast<WARActualTypePass1>(pass1_inputs[i]);
    }

    return result;
};

RWStructuredBuffer<float> result;

struct Pass0Impl : IMLP
{
    typealias PrecisionType = half;
};

struct Pass1Impl : IMLP
{
    typealias PrecisionType = half;
};

[numthreads(1, 1, 1)]
void test()
{
    result[0] = runAndSum<Pass0Impl, Pass1Impl>(3.f);
}