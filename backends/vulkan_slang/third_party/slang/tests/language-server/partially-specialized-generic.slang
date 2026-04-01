//TEST:LANG_SERVER:
//HOVER:9,29

struct MyType {}

void main()
{
    MyType a;
    let b = reinterpret<MyType>(a);
}
