//TEST:LANG_SERVER:

struct MyType : INotExist{};

void m()
{
//HOVER:8,9
    MyType b;
    reinterpret<MyType, MyType>(b);
}
