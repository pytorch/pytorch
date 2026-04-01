//TEST:SIMPLE(filecheck=METAL): -target metal -stage fragment -entry fragmentMain
//TEST:SIMPLE(filecheck=WGSL): -target wgsl -stage fragment -entry fragmentMain
//TEST:SIMPLE(filecheck=METALLIB): -target metallib -stage fragment -entry fragmentMain
//TEST:SIMPLE(filecheck=WGSLSPIRV): -target wgsl-spirv-asm -stage fragment -entry fragmentMain

// METAL: COARSEVERTEX_7
// METAL: COARSEVERTEX_6

// Ensure each attribute which may vary only appears once.
// Ensure 1, 2, 3, 4 all appear

// METAL-DAG: [[ATTR1:COARSEVERTEX_(1|2|3|4)]]

// METAL-NOT: [[ATTR1]]
// METAL-DAG: [[ATTR2:COARSEVERTEX_(1|2|3|4)]]

// METAL: COARSEVERTEX{{(_0|())}}

// METAL-NOT: [[ATTR2]]
// METAL-DAG: [[ATTR3:COARSEVERTEX_(1|2|3|4)]]

// METAL-NOT: [[ATTR3]]
// METAL-DAG: [[ATTR4:COARSEVERTEX_(1|2|3|4)]]

// WGSL:struct pixelInput
// WGSL-DAG:@location(0) [[VAR0:[A-Za-z_0-9]+]]
// WGSL-DAG:@location(1) [[VAR1:[A-Za-z_0-9]+]]
// WGSL-DAG:@location(2) [[VAR2:[A-Za-z_0-9]+]]
// WGSL-DAG:@location(3) [[VAR3:[A-Za-z_0-9]+]]
// WGSL-DAG:@location(4) [[VAR4:[A-Za-z_0-9]+]]
// WGSL-DAG:@location(6) [[VAR6:[A-Za-z_0-9]+]]
// WGSL-DAG:@location(7) [[VAR7:[A-Za-z_0-9]+]]
// WGSL-NOT:@location(8)
// WGSL:}

// METALLIB: @fragmentMain
// WGSLSPIRV: %fragmentMain = OpFunction %void None

RWStructuredBuffer<float> outputBuffer;

struct BottomFragment1
{
    float p1;
};
struct BottomFragment2
{
    float p1;
};

struct MiddleFragment1
{
    float p1;
    BottomFragment1 p2;
    BottomFragment2 p3;
};
struct TopFragment
{
    float p1 : CoarseVertex7;
    MiddleFragment1 p2 : CoarseVertex6;
    MiddleFragment1 p3 : CoarseVertex0;
};

struct FragmentStageInput
{
	TopFragment	coarseVertex : CoarseVertex;
};

// WGSL: fn fragmentMain{{[( ]*}}[[InputVar:[A-Za-z_0-9]+]]
float4 fragmentMain(FragmentStageInput input)
{
    // METAL-DAG: {{.*}}->p1{{.*}}=

    // METAL-DAG: {{.*}}->p2{{.*}}->p1{{.*}}=
    // METAL-DAG: {{.*}}->p2{{.*}}->p2{{.*}}->p1{{.*}}=
    // METAL-DAG: {{.*}}->p2{{.*}}->p3{{.*}}->p1{{.*}}=

    // METAL-DAG: {{.*}}->p3{{.*}}->p1{{.*}}=
    // METAL-DAG: {{.*}}->p3{{.*}}->p2{{.*}}->p1{{.*}}=
    // METAL-DAG: {{.*}}->p3{{.*}}->p3{{.*}}->p1{{.*}}=

    // WGSL: var [[UnpackedInput:[A-Za-z_0-9]+]] : pixelInput
    // WGSL-DAG: [[UnpackedInput]].{{[A-Za-z_0-9]+}}.{{[A-Za-z_0-9]+}} = [[InputVar]].[[VAR7]];

    // WGSL-DAG: [[UnpackedInput]].{{[A-Za-z_0-9]+}}.{{[A-Za-z_0-9]+}}.{{[A-Za-z_0-9]+}} = [[InputVar]].[[VAR6]];
    // WGSL-DAG: [[UnpackedInput]].{{[A-Za-z_0-9]+}}.{{[A-Za-z_0-9]+}}.{{[A-Za-z_0-9]+}}.{{[A-Za-z_0-9]+}} = [[InputVar]].[[VAR1]];
    // WGSL-DAG: [[UnpackedInput]].{{[A-Za-z_0-9]+}}.{{[A-Za-z_0-9]+}}.{{[A-Za-z_0-9]+}}.{{[A-Za-z_0-9]+}} = [[InputVar]].[[VAR2]];

    // WGSL-DAG: [[UnpackedInput]].{{[A-Za-z_0-9]+}}.{{[A-Za-z_0-9]+}}.{{[A-Za-z_0-9]+}} = [[InputVar]].[[VAR0]];
    // WGSL-DAG: [[UnpackedInput]].{{[A-Za-z_0-9]+}}.{{[A-Za-z_0-9]+}}.{{[A-Za-z_0-9]+}}.{{[A-Za-z_0-9]+}} = [[InputVar]].[[VAR3]];
    // WGSL-DAG: [[UnpackedInput]].{{[A-Za-z_0-9]+}}.{{[A-Za-z_0-9]+}}.{{[A-Za-z_0-9]+}}.{{[A-Za-z_0-9]+}} = [[InputVar]].[[VAR4]];

    outputBuffer[0] = input.coarseVertex.p1 + input.coarseVertex.p2.p1 + +input.coarseVertex.p3.p1;
    return float4(0, 0, 0, 0);
}
