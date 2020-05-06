#version 450 core
layout(std430) buffer;
layout(std430) uniform;
layout(set=0, rgba16f, binding=0) writeonly mediump uniform image3D uOutput;
layout(set=0, binding=1) uniform mediump sampler3D uM1;
layout(set=0, binding=2) uniform mediump sampler3D uM2;
layout(set=0, binding=3) uniform mediump sampler3D uT;
layout(set=0, binding=4) uniform constBlock{
    ivec4 outputSize;
		float beta;
		float alpha;
		int K;
} uConstBlock;

layout (local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

void main() {
	ivec3 pos = ivec3(gl_GlobalInvocationID);
	if (all(lessThan(pos, uConstBlock.outputSize.xyz))) {
		int K = uConstBlock.K;
		vec4 mmv = vec4(0);
		int ki = 0;
		for (; ki<K; ++ki) {
			vec4 m1ki = texelFetch(uM1, ivec3(ki, pos.y, pos.z), 0);
			vec4 m2ki = texelFetch(uM2, ivec3(pos.x, ki, pos.z), 0);
			mmv += m1ki * m2ki;
		}
		vec4 tv = texelFetch(uT, pos, 0);
		imageStore(uOutput, pos, uConstBlock.beta * tv + uConstBlock.alpha * mmv);
	}
}
