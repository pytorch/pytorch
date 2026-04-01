// array-size-static-const.hlsl
//TEST:COMPARE_HLSL: -profile cs_5_0

// The bug in this case is that were have a (hidden)
// cast from the `uint` constant to `int` to get
// the size of the array, and this cast was tripping
// up the constant-folding logic.

static const uint n = 16;
groupshared float b[n];

[numthreads(1,1,1)]
void main()
{}
