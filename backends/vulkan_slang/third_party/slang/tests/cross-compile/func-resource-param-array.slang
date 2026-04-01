// func-resource-param-array.slang

//TEST:CROSS_COMPILE:-target spirv-assembly -entry main -stage compute

// Test that we gernerate expected code for scenarios involving
// resource-type function parameters, even when working with
// arrays of resources.

int f(RWStructuredBuffer<int> fx,   uint fi)         { return fx[fi]    ; }

// TODO: Note that we are declaring the function
// parameter here with an explicitly-sized array
// because Slang currently doesn't support converison
// from a sized to an unsized array type.
//
int g(RWStructuredBuffer<int> gx[3], uint gi, uint gj) { return gx[gi][gj]; }

RWStructuredBuffer<int> a;
RWStructuredBuffer<int> b[3];

// Note: Slang currently genreates an array-of-arrays in the output
// for this declaration, which glslang complains leads to invalid
// SPIR-V. This means that there is yet another legalization step
// that Slang should perform on this declaration.
//
// For now we are fine with generating invalid SPIR-V, because
// we are not going to execute the output of this test case.
//
RWStructuredBuffer<int> c[4][3];

void main(uint3 tid : SV_DispatchThreadID)
{
    uint ii = tid.x;
    uint jj = tid.y;

    // Can we specialize `f`?
    //
    int tmp = f(a, ii);

    // If we ask for the same specialization again, do
    // we avoid code duplication?
    //
    tmp += f(a, jj);

    // If we pass in a reference to an array element,
    // can we still specialize?
    //
    tmp += f(b[ii], jj);

    // If we have a function that takes an *array* can
    // we specialize?
    //
    tmp += g(b, ii, jj);

    // What if the function takes an array, and we pass
    // in an element of an array-of-arrays?
    //
    tmp += g(c[ii], jj, tid.z);

    a[ii] = tmp;
}
