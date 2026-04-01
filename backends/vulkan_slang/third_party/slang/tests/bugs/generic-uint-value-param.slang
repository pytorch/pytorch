// generic-uint-value-param.slang

//DIAGNOSTIC_TEST:SIMPLE:

// Regression test to confirm that type checker
// doesn't report warnings for these constant coercions.

struct BoolG<let v : bool>
{ }

struct Test<let v : uint>
{
    int arr[v];
}

static const uint uv = 5;

void t()
{
    BoolG<true> gt;
    BoolG<bool(1)> gt2;
    BoolG<1 != 2> gt3;
    Test<uv> v;
}