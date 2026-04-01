
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cpu -compute -output-using-type -shaderobj

// This is a test modified from autodiff/reverse-struct-multi-write.slang to test that
// tuple types can be autodiff'ed the same way as struct types.

//TEST_INPUT:ubuffer(data=[1 2], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typealias A = Tuple<float, Tuple<float, float>>;

[Differentiable]
A f(A a)
{
    // Read/writes to local struct variables won't be SSA'd out by default.
    // The backward diff preparation pass will kick in to create temp vars for them.
    A aout;
    aout._1._1 = 2 * a._1._0;
    aout._1._1 = aout._1._1 + 2 * a._1._0;
    aout._1._0 = aout._1._1 + 5 * a._1._0;

    // The result should be equivalent to:
    /*
    A aout;
    var tmp = 2 * a.x;
    tmp = tmp + 2 * a.x;
    aout.y = tmp;
    aout.x = tmp + 5 * a.x;
    */
    return aout;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    A a = makeTuple(1.0, makeTuple(1.0, 2.0));

    var dpa = diffPair(a);

    A.Differential dout = makeTuple(1.0, makeTuple(1.0, 1.0));
    
    bwd_diff(f)(dpa, dout);
    // CHECK: 13
    outputBuffer[0] = dpa.d._1._0; // Expect: 13
    // CHECK: 0
    outputBuffer[1] = dpa.d._1._1; // Expect: 0
}
