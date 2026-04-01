//TEST:SIMPLE(filecheck=CHECK): -target hlsl -entry testMain -profile cs_6_0

// Test that invoking the default constructor of a type then use the result as an existential value
// works correctly.

RWStructuredBuffer<uint> output;
RWStructuredBuffer<uint> expected;

interface ITest
{
    uint getValue();
};


//TEST_INPUT:type_conformance Test0:ITest = 0
struct Test0 : ITest
{
    uint getValue() { return 0; }
};

//TEST_INPUT:type_conformance Test1:ITest = 1
struct Test1 : ITest
{
    uint getValue() { return 1; }
};

//TEST_INPUT:type_conformance TestAny:ITest = 2
struct TestAny : ITest
{
    uint value = 5;
    __init(uint v)
    {
        value = v;
    }

    __init()
    {
        value = 0;
    }

    uint getValue() { return value; }
}

ITest makeTest0()
{
    return Test0();
}

ITest makeTest1()
{
    return Test1();
}

// CHECK: TestAny{{.*}} makeTestAny{{.*}}()
// CHECK: return TestAny_{{.*}}init{{.*}}()
ITest makeTestAny()
{
    return TestAny();
}

ITest makeTestAny(uint v)
{
    return TestAny(v);
}


[numthreads(16, 1, 1)]
void testMain(uint3 threadID: SV_DispatchThreadID)
{
    if (threadID.x != 0)
        return;

    int outputIdx = 0;

    /// Test0
    {
        Test0 test;
        output[outputIdx] = test.getValue();
        expected[outputIdx++] = 0;
    }

    {
        ITest test = Test0();
        output[outputIdx] = test.getValue();
        expected[outputIdx++] = 0;
    }

    {
        output[outputIdx] = Test0().getValue();
        expected[outputIdx++] = 0;
    }

    {
        ITest test = makeTest0();
        output[outputIdx] = test.getValue();
        expected[outputIdx++] = 0;
    }

    {
        output[outputIdx] = makeTest0().getValue();
        expected[outputIdx++] = 0;
    }

    output[outputIdx] = 1000;
    expected[outputIdx++] = 1000;

    /// Test1
    {
        Test1 test;
        output[outputIdx] = test.getValue();
        expected[outputIdx++] = 1;
    }

    {
        ITest test = Test1();
        output[outputIdx] = test.getValue();
        expected[outputIdx++] = 1;
    }

    {
        output[outputIdx] = Test1().getValue();
        expected[outputIdx++] = 1;
    }

    {
        ITest test = makeTest1();
        output[outputIdx] = test.getValue();
        expected[outputIdx++] = 1;
    }

    {
        output[outputIdx] = makeTest1().getValue();
        expected[outputIdx++] = 1;
    }

    output[outputIdx] = 2000;
    expected[outputIdx++] = 2000;

    /// TestAny
    {
        TestAny test;
        output[outputIdx] = test.getValue();
        expected[outputIdx++] = 5;
    }

    {
        ITest test = TestAny();
        output[outputIdx] = test.getValue();
        expected[outputIdx++] = 5;
    }

    {
        ITest test = TestAny(2);
        output[outputIdx] = test.getValue();
        expected[outputIdx++] = 2;
    }

    {
        output[outputIdx] = TestAny().getValue();
        expected[outputIdx++] = 5;
    }

    {
        output[outputIdx] = TestAny(2).getValue();
        expected[outputIdx++] = 2;
    }

    {
        ITest test = makeTestAny();
        output[outputIdx] = test.getValue();
        expected[outputIdx++] = 5;
    }

    {
        ITest test = makeTestAny(2);
        output[outputIdx] = test.getValue();
        expected[outputIdx++] = 2;
    }

    {
        output[outputIdx] = makeTestAny().getValue();
        expected[outputIdx++] = 5;
    }

    {
        output[outputIdx] = makeTestAny(2).getValue();
        expected[outputIdx++] = 2;
    }

    expected[outputIdx++] = uint(-1);
}
