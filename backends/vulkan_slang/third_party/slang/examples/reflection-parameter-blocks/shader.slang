// shader.slang

// This shader is part of the `reflection-parameter-blocks`
// example program.
//
// This file is an example program that *only* declares
// shader parameters in the global scope, and *only* uses
// explicit parameter blocks.

import common;

ParameterBlock<Environment> environment;
ParameterBlock<View>        view;
ParameterBlock<Material>    material;
ParameterBlock<Mesh>        mesh;

[shader("compute")]
void main()
{
    float4 r = 0;

    use(r, mesh);
    use(r, material);
    use(r, view);
    use(r, environment);
}
