//TEST:LANG_SERVER:
//
// tests broken subscript decl.
//HOVER:7,24
struct MyType
{
    __subscript(int index)
    {
        get(){return index;}
        se
    }
}
