//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-d3d11 -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer

public interface A<T: IFloat>
{
}

public extension<T1: IFloat, a1 : A<T1>> a1
{
    void foo() {
        outputBuffer[0] = 1.0;
    }
}

RWStructuredBuffer<float> outputBuffer;
struct S : A<float> {
}

void helper<T: IFloat, a : A<T>>(a a2)
{
    a2.foo();
}

// CHECK: 1

[numthreads(1,1,1)]
void computeMain()
{
    S a;
    helper(a);
}