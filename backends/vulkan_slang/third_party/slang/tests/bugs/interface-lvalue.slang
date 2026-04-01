//TEST:EXECUTABLE:
__target_intrinsic(cpp, "printf(\"%s\\n\", ($0).getBuffer())")
void writeln(String text);

[COM("BE18F5D2-4522-4AB0-A6EE-1D157FA2B083")]
interface IFoo
{
    int method();
};

class Impl : IFoo
{
    __init() { }
    int method()
    {
        return 0;
    }
}

void createFoo(out IFoo val)
{
    Impl resuult = new Impl();
    val = resuult;
}

export __extern_cpp int main()
{
    IFoo v;
    createFoo(v);

    if (v.method() == 0)
        writeln("succ");
    else
        writeln("fail");
    return 0;
}
