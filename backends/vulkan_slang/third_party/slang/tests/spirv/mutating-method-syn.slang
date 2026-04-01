// mutating-method-syn.slang
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute
// Test ability to directly output SPIR-V

interface IFoo
{
    [mutating]
    int bar(inout int y);
}

struct Val : IFoo
{
    int x;

    int bar(int y)
    {
        return x + y;
    }
}

int test<T:IFoo>(inout T f, inout int y)
{
    return f.bar(y);
}

//TEST_INPUT:set result = out ubuffer(data=[0 0 0 0], stride=4)

RWStructuredBuffer<int> result;
[numthreads(1,1,1)]
void computeMain()
{    
    Val v;
    int y = 0;
    v.x = 1;
    
    // CHECK: 1
    result[0] = test(v, y);
}
