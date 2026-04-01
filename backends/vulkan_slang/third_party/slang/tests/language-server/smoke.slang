//TEST(smoke):LANG_SERVER:
//COMPLETE:31,21
//HOVER:25,30
//SIGNATURE:25,40
interface IFoo
{
    /**
    Returns the sum of the contents.
    */
    int getSum();
}

struct MyType : IFoo
{
    int getSum() { return 0; }
}

struct Pair<T:IFoo, U: IFoo> : IFoo
{
    T first;
    U second;
    /**
    Returns the sum of the contents.
    */
    int getSum() { return first.getSum() + second.getSum(); }
}

void m()
{
    Pair<MyType, Pair<MyType, MyType>> v;
    v.first = v.second.first;
    
}
