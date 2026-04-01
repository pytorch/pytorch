//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* Tuple experiment.

Works.
 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

struct Tuple2<T0, T1>
{
    T0 _0;
    T1 _1;
};

struct Tuple3<T0, T1, T2>
{
    T0 _0;
    T1 _1;
    T2 _2;
};

Tuple2<T0, T1> makeTuple<T0, T1>(T0 t0, T1 t1)
{
    Tuple2<T0, T1> t;
    t._0 = t0;
    t._1 = t1;
    return t;
}

Tuple3<T0, T1, T2> makeTuple<T0, T1, T2>(T0 t0, T1 t1, T2 t2)
{
    Tuple3<T0, T1, T2> t;
    t._0 = t0;
    t._1 = t1;
    t._2 = t2;
    return t;
}

int doThing(Tuple2<float, int> v)
{
    return v._0 + v._1;
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;

    let t  = makeTuple(index + 1.0f, index - 1);
    
	outputBuffer[index] = doThing(t);
}

