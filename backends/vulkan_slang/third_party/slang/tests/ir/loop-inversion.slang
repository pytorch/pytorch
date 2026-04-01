//TEST():SIMPLE(filecheck=CHECK):-entry computeMain -stage compute -line-directive-mode none -target hlsl -loop-inversion
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=OUT):-shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=OUT):-dx12 -use-dxil -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=OUT):-cpu -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=OUT):-vk -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=OUT):-cpu -shaderobj -output-using-type

// Check that all the backends cope with the slightly unusual IR the loop inversion generated

// OUT: 180

// For all the below functions, verify that the body (adding to j and
// incrementing i) comes before any break. This verifies that the `break` has
// been moved to the end of the loop.

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

// A standard loop
// CHECK-LABEL: int a_{{.*}}()
// CHECK-NOT: break;
// CHECK: int {{.*}} = j_{{.*}} + [[i:i_[0-9]+]]
// CHECK: [[i]] + int(1);
// CHECK: if(
// CHECK: break;
// CHECK: return
int a()
{
    int j = 0;
    for(int i = 0; i < 10; ++i)
        j += i;
    return j;
}

// A vanilla while loop
// CHECK-LABEL: int b_{{.*}}()
// CHECK-NOT: break;
// CHECK: int {{.*}} = j_{{.*}} + [[i:i_[0-9]+]]
// CHECK: [[i]] + int(1);
// CHECK: if(
// CHECK: break;
// CHECK: return
int b()
{
    int j = 0;
    int i = 0;
    while(i < 10)
    {
        j += i;
        i++;
    }
    return j;
}

// A while loop with a break on the false branch
// CHECK-LABEL: int c_{{.*}}()
// CHECK-NOT: break;
// CHECK: int {{.*}} = j_{{.*}} + [[i:i_[0-9]+]]
// CHECK: [[i]] + int(1);
// CHECK: if(
// CHECK: break;
// CHECK: return
int c()
{
    int j = 0;
    int i = 0;
    do
    {
        if(i < 10)
            {}
        else
            break;
        j += i;
        i++;
    } while(true);
    return j;
}

// A while loop with a break on the true branch
// CHECK-LABEL: int d_{{.*}}()
// CHECK-NOT: break;
// CHECK: int {{.*}} = j_{{.*}} + [[i:i_[0-9]+]]
// CHECK: [[i]] + int(1);
// CHECK: if(
// CHECK: break;
// CHECK: return
int d()
{
    int j = 0;
    int i = 0;
    do
    {
        if(i >= 10)
            break;
        else
            {}
        j += i;
        i++;
    } while(true);
    return j;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    outputBuffer[dispatchThreadID.x] = a() + b() + c() + d();
}
