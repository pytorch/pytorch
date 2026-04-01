//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

interface IOperation<T : __BuiltinFloatingPointType>
{
  static T apply(T lhs, T rhs);
};
 
T applyOp<T:__BuiltinFloatingPointType, TOp:IOperation<T>>(T lhs, T rhs)
{
  return TOp::apply(lhs, rhs);
}

struct AddOp<T : __BuiltinFloatingPointType> : IOperation<T>
{
    static T apply(T lhs, T rhs)
    {
        return lhs + rhs;
    }
}

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    let result = applyOp<float, AddOp<float>>(1.0, 2.0);

    // CHECK: 3
    outputBuffer[0] = (int)result;
}
