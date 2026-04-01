//TEST:EXECUTABLE:
[COM("BE18F5D2-4522-4AB0-A6EE-1D157FA2B083")]
interface IFoo
{
    int method();
};

class MyClass : IFoo
{
    int intMember;
    __init()
    {
        intMember = 0;
    }
    int method()
    {
        printf("method\n");
        return intMember;
    }
};

[DllExport]
export IFoo createFoo()
{
    IFoo result = new MyClass();
    return result;
}

// Now we import the `createFoo` function from current module.
[DllImport("", "createFoo")]
IFoo createFooImported();

export __extern_cpp int main()
{
    var obj = createFooImported();
    return obj.method();
}
