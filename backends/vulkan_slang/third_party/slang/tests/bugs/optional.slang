//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-slang -compute
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-slang -compute -vk

interface IFoo
{
    void foo();
}

struct S : IFoo { int x; void foo(); }

struct P
{
    IFoo f;
}
struct Tr
{
    int test<T:IArithmetic>(T t, P p)
    {
        const IFoo hit = p.f;
        let castResult = hit as S;
        if (!castResult.hasValue)
            return 0;
        return castResult.value.x;
    }
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name buffer

RWStructuredBuffer<int> buffer;

[numthreads(1,1,1)]
void computeMain()
{
    P p;
    S s;
    s.x = 2;
    p.f = s;
    Tr tt;
    // BUF: 2
    buffer[0] = tt.test(0, p);
}

