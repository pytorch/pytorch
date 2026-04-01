//TEST:LANG_SERVER:

struct MyType {};

void m()
{
//HOVER:8,9
    MyType b;
    int t = 2tdf;
    float c = 3.0ug;
    reinterpret<MyType, MyType>(b);
}
