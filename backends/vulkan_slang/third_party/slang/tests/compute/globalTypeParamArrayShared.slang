//TEST_IGNORE_FILE:
interface IBase
{
    float compute<T>(T g);
}
struct Base:IBase
{
    float b;
    float compute<T>(T g) { return b; }
};

struct Pair<T1:IBase, T2:IBase> : IBase
{
    T1 head;
    T2 tail;
    float compute<T>(T g)
    {
        return head.compute(g) + tail.compute(g);
    }
};

struct Arr<T:IBase, let N:int> : IBase
{
    T base[N];  // = 1.0
    float compute<T>(T g)
    {
        float sum = 0.0;
        for (int i = 0; i < N; i++)
            sum += base[i].compute(g);
        return sum;
    }
};
