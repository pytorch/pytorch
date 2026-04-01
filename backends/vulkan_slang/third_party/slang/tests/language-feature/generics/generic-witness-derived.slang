//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type

// Test that we can compile a generic function with a generic type constraint that is dependent on an
// outer generic type parameter.

namespace ns{

    public interface IBinaryElementWiseFunction<T>
    {
        public static T call(const in T lhs, const in T rhs);
    }
    public struct AddOp<T : IArithmetic> : IBinaryElementWiseFunction<T>
    {
        public static  T call(const in T lhs, const in T rhs)
        {
            return lhs + rhs; 
        }
    }
    public struct BinaryElementWiseInputData<T : IArithmetic>
    {
        T lhs;
        T rhs;

        // Note: `U` is constrainted by `IBinaryElementWiseFunction<T>`, which is dependent on `T`,
        // that is another generic type parameter defined on the outer type.
        // This eventually leads to a IRGeneric where one param has a type that is dependent on
        // another param.
        // In this case, the IR for `test` after generic flattening will be:
        // ```
        // %g_test = IRGeneric
        // {
        //     IRBlock
        //     {
        //          %T = IRParam : Type;
        //          %T_w = IRParam : IRWitnessTableType<IArithmetic>;
        //          %U = IRParam : Type;
        //          %U_w = IRRaram : IRWitnessTableType<%s>; // note that the type here is a forward reference to %s
        //          %s = specialize(%IBinaryElementWiseFunction, %T) // %s is dependent on %T.
        //          ...
        //     }
        // }
        // 
        public T test<U : IBinaryElementWiseFunction<T>>(U x)
        {
            return x.call(lhs ,rhs);
        }
    }
}


//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[shader("compute")]
[numthreads(1,1,1)]
void computeMain(uint3 threadId: SV_DispatchThreadID)
{
    ns::BinaryElementWiseInputData<int> cb;
    cb.lhs = threadId.x + 1;
    cb.rhs = 2;
    // CHECK: 3
    outputBuffer[0] = cb.test(ns::AddOp<int>());
}