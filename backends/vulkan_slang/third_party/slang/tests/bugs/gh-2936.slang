//TEST(smoke,compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu -shaderobj -output-using-type

// CHECK:        4
// CHECK-NEXT:  99
// CHECK-NEXT: 111
// CHECK-NEXT:  40

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain()
{
    outputBuffer[0] = f();
    outputBuffer[1] = g();
    outputBuffer[2] = h();
    outputBuffer[3] = k(30, 40);
}

static let x = 1;

func f() -> int
{
    let y = x; // should refer to the global x
    let x = 3;
    return x + y; // should refer to the local x
}
func g() -> int
{
    // Can we shadow this keyword
    int out;
    out = 99;
    return out;
}

static var input = 999;
func h() -> int
{
    // This should still parse as an identifier (it's a keyword, being shadowed
    // by something in a parent scope)
    input = 10;
    int input = input;
    input += 101;
    return input;
}

int k(int a, int b)
{
    int umax = max(a, b);

    int max = umax;
    return max;
}
