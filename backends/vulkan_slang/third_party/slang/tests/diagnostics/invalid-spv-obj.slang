//TEST:SIMPLE(filecheck=CHECK): -target spirv

[shader("compute")]
[numthreads(1,1,1)]
void main() {
    int var2 = spirv_asm {
        // CHECK: ([[# @LINE+1]]): error 29114
        result: $$int = OpIMul $(15) %invalidId
    };
}