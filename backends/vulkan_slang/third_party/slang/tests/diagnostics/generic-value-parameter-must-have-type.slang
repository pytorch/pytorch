//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target spirv

interface ITest
{
    static const uint kValue0;
    static const uint kValue1;
};

// `TValue1` does not have an explicit type, this should fail to compile.
// CHECK: error 30623: a generic value parameter
// Make sure erroneous code is printed out.
// CHECK: let TValue1>
struct TestImpl<let TValue0 : uint, let TValue1> : ITest
{
    static const uint kValue0 = TValue0;
    static const uint kValue1 = TValue1;
};

void computeMain()
{
}
