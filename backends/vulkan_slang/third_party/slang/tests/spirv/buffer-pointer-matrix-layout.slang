//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly -stage compute -entry main -matrix-layout-column-major

// CHECK: OpLoad {{.*}} Aligned 4

struct Push
{
    float3x4* ptr;
};

[[vk::push_constant]] Push push;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{    
    // This matrix is in memry column major. Slang respects this here and load it properly!
    float3x4 correctly_read_matrix = *push.ptr;
    printf("(%f,%f,%f,%f)\n(%f,%f,%f,%f)\n",
        correctly_read_matrix[0][0], correctly_read_matrix[0][1], correctly_read_matrix[0][2], correctly_read_matrix[0][3],
        correctly_read_matrix[1][0], correctly_read_matrix[1][1], correctly_read_matrix[1][2], correctly_read_matrix[1][3]
    );
    printf("(%f,%f,%f,%f)\n\n",
        correctly_read_matrix[2][0], correctly_read_matrix[2][1], correctly_read_matrix[2][2], correctly_read_matrix[2][3]
    );
    // With this syntax however, Slang ignores the column major setting and loads it as it it was row major!
    float3x4 broken_matrix = push.ptr[0];
    printf("(%f,%f,%f,%f)\n(%f,%f,%f,%f)\n",
        broken_matrix[0][0], broken_matrix[0][1], broken_matrix[0][2], broken_matrix[0][3],
        broken_matrix[1][0], broken_matrix[1][1], broken_matrix[1][2], broken_matrix[1][3]
    );
    printf("(%f,%f,%f,%f)\n\n",
        broken_matrix[2][0], broken_matrix[2][1], broken_matrix[2][2], broken_matrix[2][3]
    );
}