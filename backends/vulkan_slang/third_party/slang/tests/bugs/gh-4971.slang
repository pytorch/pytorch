//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-d3d11 -output-using-type

struct Test {
    RWStructuredBuffer<int> val;
    __subscript(int x, int y)->int
    {
        get { return val[x * 3 + y]; }
        [nonmutating] set { val[x * 3 + y] = newValue; }
    }
}
Test test;

//TEST_INPUT: set test = {out ubuffer(data=[0 0 0 0 0 0 0 0 0], stride=4)};

[numthreads(1, 1, 1)]
void computeMain()
{
    // test[0,0] should be a valid l-value here because although `test` is
    // a read-only parameter, the `set` accessor is marked as `nonmutating`,
    // which means that it can be called even when `test` is not mutable.

    // CHECK: 1
    test[0,0] = 1;
}