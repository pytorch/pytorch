//TEST:INTERPRET(filecheck=CHECK):
struct Inner
{
    int header;
    int array[3];
}
struct MyType
{
    float3 value1;
    float3 value2;
    Inner inner;
}
MyType getValue()
{
    MyType t = { float3(1, 2, 3), float3(-1,-2, 3), { 4, { 5, 6, 7 } } };
    return t;
}
int main()
{
    var t = getValue();
    int sum = 0;
    for (int i = 0; i < 3; ++i)
    {
        sum += t.inner.array[i];
    }
    t.value1 += t.value2;
    for (int i = 0; i < 3; i++)
    {
        sum += (int)t.value1[i];
    }
    //CHECK: 24
    printf("%d\n", sum);
    return 0;
}
