//TEST:SIMPLE(filecheck=CHECK): -target spirv

struct Test {
    RWStructuredBuffer<int> val;
    __subscript(int x, int y)->int
    {
        get { return val[x * 3 + y]; }
        set { val[x * 3 + y] = newValue; }
    }
}
Test test;

[numthreads(1, 1, 1)]
void computeMain()
{
    // test[0,0] is not an l-value because `test` is a read-only parameter,
    // and the `set` accessor is by-default `mutating`, which means that it is
    // only callable when `test` itself is l-value.
    
    // CHECK: ([[# @LINE+1]]): error 30011
    test[0,0] = 1;

}