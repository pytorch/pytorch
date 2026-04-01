//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

struct MyType
{
    private int member;
    private int func()
    {
        return member;
    }
    struct SubType
    {
        int member2;
        int func2(MyType m)
        {
            return m.member + member2; // OK.
        }
    }
    int func1() { return member; }

    private __init() { member = 0; }
    
    private __subscript(int i)->int
    {
        get { return member; }
        set { member = newValue; }
    }
    // CHECK:{{.*}}(28): error 30601:
    public void publicMethod() {} // ERROR.
}

void test()
{
    // CHECK:{{.*}}(34): error 30600:
    MyType t; // ERROR.
    // CHECK-NOT:{{.*}}error
    t.func1(); // OK.
    // CHECK:{{.*}}(38): error 30600:
    t.func(); // ERROR.
    // CHECK:{{.*}}(40): error 30600:
    t[0] = 1; // ERROR.

    // CHECK:{{.*}}(43): error 30600:
    t.member = 2;
}
