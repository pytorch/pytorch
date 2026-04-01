//TEST:SIMPLE(filecheck=CHECK):

// Test to catch compilation time errors with return values in constructors within an ordinary struct.
// A constructor from the callsite's point of view is a function that returns the struct type.
// A constructor from the inside the body is treated as a function that modifies `this` and not return.

// In the following test-cases where the constructor either returns an expr or no value,
// within a condition for early exit or not, it automatically gets resolved as
// this = t;
// return *this; or return;

struct S
{
    float v;
    __init() { v = 0; }
}

struct S1a : S
{
    __init()
    {
        this.v = 5;
        // CHECK-NOT: ([[# @LINE+1]]): error
        return this;
    }
}

struct S1b : S
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

struct S2a : S
{
    __init()
    {
        S2a t;
        t.v = 5;
        // CHECK-NOT: ([[# @LINE+1]]): error
        return;
    }
}

struct S2b : S
{
    __init(float u)
    {
        if (u != 0)
        {
            S2b t;
            t.v = u;
            // CHECK-NOT: ([[# @LINE+1]]): error
            return;
        }
        // CHECK-NOT: ([[# @LINE+1]]): error
        return;
    }
}

struct S3a : S
{
    __init()
    {
        S3a t;
        t.v = 5;
        // CHECK-NOT: ([[# @LINE+1]]): error
        return t;
    }
}

struct S3b : S
{
    __init(float u)
    {
        if (u != 0)
        {
            S3b t;
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
    S1a S1a;
    S2a S2a;
    S3a S3a;
    S1b S1b = S1b(0);
    S2b S2b = S2b(0);
    S3b S3b = S3b(0);
}
