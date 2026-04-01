struct input_t {
    uint32_t i : 8;
    uint32_t j : 24;
};

struct output_t {
    uint32_t i;
    uint32_t j;
};

uniform void *parameters[1000];

//TEST:SIMPLE(filecheck=CHECK): -target spirv

//CHECK: OpEntryPoint

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID, uint groupIndex: SV_GroupIndex) {
    // buffer 0 is input
    // buffer 1 is output
    input_t *input = (input_t *)(parameters[0]);
    output_t *output = (output_t *)(parameters[1]);

    output->i = input->i;
    output->j = input->j;
}