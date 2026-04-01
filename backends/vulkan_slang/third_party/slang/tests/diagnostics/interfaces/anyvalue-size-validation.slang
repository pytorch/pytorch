// anyvalue-size-validation.slang

//DIAGNOSTIC_TEST:SIMPLE:-target cpp -stage compute -entry main -disable-specialization

[anyValueSize(8)]
interface IInterface
{
    int doSomething();
};

struct S : IInterface
{
    uint a;
    uint b;
    uint c;
    int doSomething() { return 5; }
};

T test<T:IInterface>(T s)
{
    return s;
}

RWStructuredBuffer<uint> output;

[numthreads(4, 1, 1)]
void main()
{
    S s = S(1, 2, 3);
    output[0] = test(s).a;
}
