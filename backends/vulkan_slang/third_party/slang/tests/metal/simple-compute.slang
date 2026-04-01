//TEST:SIMPLE(filecheck=CHECK): -target metal
//TEST:SIMPLE(filecheck=CHECK-ASM): -target metallib
//TEST:REFLECTION(filecheck=REFLECT):-target metal -entry main_kernel -stage compute

uniform RWStructuredBuffer<float> outputBuffer;

struct MyBlock
{
    StructuredBuffer<float> b1;
    StructuredBuffer<float> b2;
}
ParameterBlock<MyBlock> block;
ParameterBlock<MyBlock> block2;

// CHECK: {{\[\[}}kernel{{\]\]}} void main_kernel(float device* {{.*}} {{\[\[}}buffer(0){{\]\]}}, MyBlock{{.*}} constant* block{{.*}} {{\[\[}}buffer(1){{\]\]}}, MyBlock{{.*}} constant* block2{{.*}} {{\[\[}}buffer(2){{\]\]}})

// CHECK-ASM: define void @main_kernel

// REFLECT: "elementVarLayout": {
// REFLECT: "name": "b1",
// REFLECT: "binding": {"kind": "metalArgumentBufferElement", "index": 0}

// REFLECT: "name": "b2",
// REFLECT: "binding": {"kind": "metalArgumentBufferElement", "index": 1}

// REFLECT: "binding": {"kind": "metalArgumentBufferElement", "index": 0, "count": 2}

// REFLECT: "name": "outputBuffer",
// REFLECT-NEXT: "binding": {"kind": "constantBuffer", "index": 0{{.*}}}

// REFLECT: "name": "block",
// REFLECT-NEXT: "binding": {"kind": "constantBuffer", "index": 1{{.*}}}

void func(float v)
{
    outputBuffer[0] = v;
    outputBuffer[1] = outputBuffer.Load(0);
    outputBuffer[2] = block.b1[0] + block2.b2[0];
}

[numthreads(1,1,1)]
void main_kernel()
{
    func(3.0f);
}