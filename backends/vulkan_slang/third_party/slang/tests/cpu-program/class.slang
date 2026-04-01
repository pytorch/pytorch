//TEST:EXECUTABLE:
class MyClass
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
}

export __extern_cpp int main()
{
    MyClass obj = new MyClass();
    return obj.method();
}
