//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-vk -slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct HitInfo
{
    float val;
    float getVal() { return val; }
}
struct PathInfo
{
    HitInfo hit;
    [mutating]
    void setVal(float x)
    {
        hit.val = x;
    }
}

void someOp(HitInfo hitInfo)
{
    outputBuffer[0] = hitInfo.val;
}

void func(int x)
{
    PathInfo path = {};
    int i = 0;
    if (x < 10)
    {
        // The `elementAddr(path, hit)` is first defined in this block.
        // Note that this block dominates all remaining blocks in this
        // function, so all future references to path.hit will use
        // `elementAddr` inst defined here.
        //
        // The bug here is that when we emit code, the emit logic will
        // find that the elementAddr inst is not in a valid region for
        // future use and will try to create a local var for it. This will
        // result a pointer typed var being created, and leads to invalid
        // hlsl/glsl code.
        // 
        // The fix is to hoist all always-fold insts to earliest point
        // in the program instead of creating a var for it.
        //
        someOp(path.hit);
    }
    else
    {
        // This return makes the true block dominate the rest of the blocks
        // from a region that does not "dominate" the rest code regions.
        return;
    }

    while (i < 3)
    {
        if (i < 2)
        {
            path.setVal(1);
        }
        else
        {
            path.setVal(2);
        }

        // This is the point where we are using path.hit again.
        outputBuffer[i] = path.hit.getVal();
        i++;
    }
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    func(4);
}
