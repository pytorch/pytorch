//DISABLE_TEST:SIMPLE:-target hlsl -entry computeMain -profile cs_6_2

/* 
In C++ we are able to explicitly specialize over more than one type/value. Here we try to use a 'dummy' generic 
type such that an extension can be applied to it.

I can't just specialize a function also complicating things. 

If I try explicit function specialization in C++. In g++11.1 it will complain if there isn't a specialition visible. 
Visual studio it seems to assume it is available for import and doesn't complain.
*/

RWStructuredBuffer<float> outputBuffer;

interface IDoThing
{
    static float doThing(float v);
};

struct Combination<T, let V : int> {};

extension Combination<int, 10> : IDoThing
{
    static float doThing(float v)
    {
        return int(v) + 10;
    }
};

extension Combination<float, 20> : IDoThing
{
    static float doThing(float v)
    {
        return float(v) + 20;
    }
};


[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;

    let v = Combination<float, 20>::doThing(tid) + 
        Combination<int, 10>::doThing(tid);
    
    // Produces an error - although the error message of typeof(Combination)
    // is probably not great.
    // 
    // slang(35): error 30027: 'doThing' is not a member of 'typeof(Combination)'.
    //
    //let y = Combination<int, 20>::doThing(tid);
    
    outputBuffer[tid] = v;
}

