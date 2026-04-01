//TEST:SIMPLE(filecheck=CHK): -target spirv -entry computeMain

// Both out and inout parameters
// should have this treated as writes
void out_test(int tmp, out int x)
{
    x = tmp;
}

// Permuting arguments to ensure that
// the correct argument is checked
void inout_test(inout int x, int tmp)
{
    x = tmp;
}

// __ref parameters should also be fine
void ref_test(__ref int x, int tmp)
{
    x = tmp;
}

void undefined_function_use()
{
    int x;
    int tmp = 1;

    //CHK-DAG: warning 41016: use of uninitialized variable 'x'
    out_test(x, tmp);
    
    //CHK-DAG: warning 41016: use of uninitialized variable 'x'
    inout_test(tmp, x);
}

void out_function_use()
{
    int x;
    int tmp = 1;

    // Acts as a write
    out_test(tmp, x);

    // Rest are fine now
    out_test(x, tmp);
    inout_test(tmp, x);
}

void inout_function_use()
{
    int x;
    int tmp = 1;

    // Acts as a write
    inout_test(x, tmp);

    // Rest are fine now
    out_test(x, tmp);
    inout_test(tmp, x);
}

void ref_function_use()
{
    int x;
    int tmp = 1;

    // Acts as a write
    ref_test(x, tmp);

    // Rest are fine now
    out_test(x, tmp);
    inout_test(tmp, x);
}

// Likewise for generic functions
static int ord_gen_result;

__generic<T>
void ordinary_generic(T x)
{
    ord_gen_result = __slang_noop_cast<int, T>(x);
}

__generic<T>
void unordinary_generic(out T x)
{
    x = __slang_noop_cast<T, int>(1);
}

void undefined_generic_use()
{
    int x;

    //CHK-DAG: warning 41016: use of uninitialized variable 'x'
    ordinary_generic(x);
}

void ok_generic_use()
{
    int x;
    unordinary_generic(x);
    ordinary_generic(x);
}

// Check proper handling of aliases passed
void f()
{
    int3 dim;

    // Should have no warnings
    out_test(1, dim.x);
}

//CHK-NOT: warning 41016

[Shader("compute")]
[NumThreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
}

