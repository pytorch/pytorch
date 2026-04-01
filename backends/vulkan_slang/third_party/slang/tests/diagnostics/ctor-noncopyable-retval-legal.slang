//TEST:SIMPLE(filecheck=CHECK):

// Test to catch compilation time errors with return values in constructors within a NonCopyable struct.
// A constructor from the callsite's point of view is a function that returns the struct type.
// A constructor from the inside the body is treated as a function that modifies `this` and not return.

// In the following test-cases where the constructor either returns an expr or no value,
// within a condition for early exit or not, it automatically gets resolved as
// return *this; or return;

[__NonCopyableType] struct S
{
    float v;    
}

[__NonCopyableType] struct S1 : S
{
    __init()
    {
        this.v = 5;
        // CHECK-NOT: ([[# @LINE+1]]): error
        return this;
    }
}

[__NonCopyableType] struct S4 : S
{
    __init(float u)
    {
        if (u != 0)
        {
            this.v = u;
            // CHECK-NOT: ([[# @LINE+1]]): error
            return this;
        }
        // CHECK-NOT: ([[# @LINE+1]]): error
        return;
    }
}

[__NonCopyableType] struct S2 : S
{
    __init()
    {
        S2 t;
        t.v = 5;
        // CHECK-NOT: ([[# @LINE+1]]): error
        return;
    }
}

[__NonCopyableType] struct S5 : S
{
    __init(float u)
    {
        if (u != 0)
        {
            S5 t;
            t.v = u;
            // CHECK-NOT: ([[# @LINE+1]]): error
            return;
        }
        // CHECK-NOT: ([[# @LINE+1]]): error
        return;
    }
}

[__NonCopyableType] struct S3 : S
{
    __init()
    {
        S3 t;
        t.v = 5;
        // CHECK-NOT: ([[# @LINE+1]]): error
        return t;
    }
}

[__NonCopyableType] struct S6 : S
{
    __init(float u)
    {
        if (u != 0)
        {
            S6 t;
            t.v = u;
            // CHECK-NOT: ([[# @LINE+1]]): error
            return t;
        }
        // CHECK-NOT: ([[# @LINE+1]]): error
        return;
    }
}

void main()
{
    S1 s1;
    S2 s2;
    S3 s3;
    S4 s4 = S4(0);
    S5 s5 = S5(0);
    S6 s6 = S6(0);
}
