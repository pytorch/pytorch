//TEST:LANG_SERVER:
struct MyType{};
void m()
{
    MyType b;
//HOVER:7,19
    reinterpret<MyType, MyType>(b);
}
