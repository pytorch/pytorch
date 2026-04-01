// Test that the override behavior around extensions and generic extensions works as expected.

// When there are multiple ways for a type to conform to an interface, then the expected behavior
// is that:
// 1. If the type directly implements an interface, use that conformance.
// 2. Otherwise, if there is a direct extension on the type that makes it conform to the interface, use that
//    extension.
// 3. Otherwise, if there is a generic extension that makes the type conform to the interface, use that.

//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj
interface IFoo
{
    int getVal();
}

interface IBar
{
    int getValPlusOne();
}

interface IBaz
{
    int getValPlusTwo();
}

struct MyInt
{
    int v;
}

extension MyInt : IFoo
{
    int getVal() { return v; }
}

extension MyInt : IBar
{
    int getValPlusOne() { return this.getVal() + 2; }
}

extension<T: IFoo> T : IBar
{
    int getValPlusOne() { return this.getVal() + 1; }
}

int helper1<T:IBar>(T v){ return v.getValPlusOne();}
int helper2<T:IFoo>(T v){ return v.getValPlusOne();}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    MyInt v = {1};

    // CHECK: 3
    outputBuffer[0] = v.getValPlusOne(); // should call MyInt::ext::getValPlusOne();

    // CHECK: 3
    outputBuffer[1] = helper1(v); // should call MyInt::ext::getValPlusOne();

    // CHECK: 2
    outputBuffer[2] = helper2(v); // should call T::ext::getValPlusOne();
}