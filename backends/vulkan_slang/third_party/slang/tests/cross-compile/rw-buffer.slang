// rw-buffer.slang

// Confirm that writing into a `RWBuffer` generates appropriate GLSL/SPIR-V.

//TEST:SIMPLE(filecheck=CHECK): -profile ps_5_0 -entry main -target spirv-assembly
//TEST:SIMPLE(filecheck=CHECK): -profile ps_5_0 -entry main -target spirv-assembly -emit-spirv-directly

RWBuffer<float> buffer;
Buffer<float> inBuffer;

float4 main(float u : U, int idx : IDX) : SV_Target
{
    // CHECK-DAG: OpImageQuerySize

    // CHECK-DAG: OpImageRead

    // CHECK-DAG: OpImageFetch

    // CHECK-DAG: OpImageWrite

    uint s1, s2;
    buffer.GetDimensions(s1);
    buffer.GetDimensions(s2);

	buffer[idx] = buffer[idx] + inBuffer[idx] + s1 + s2;
	return u;
}
