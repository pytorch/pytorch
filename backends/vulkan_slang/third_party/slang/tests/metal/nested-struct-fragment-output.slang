//TEST:SIMPLE(filecheck=METAL): -target metal -stage fragment -entry fragmentMain
//TEST:SIMPLE(filecheck=WGSL): -target wgsl -stage fragment -entry fragmentMain
//TEST:SIMPLE(filecheck=METALLIB): -target metallib -stage fragment -entry fragmentMain
//TEST:SIMPLE(filecheck=WGSLSPIRV): -target wgsl-spirv-asm -stage fragment -entry fragmentMain

//METAL-DAG: color(0)
//METAL-DAG: color(1)
//METAL-DAG: color(2)
//METAL-DAG: color(3)
//METAL-DAG: color(4)
//METAL-DAG: color(5)
//METAL-DAG: color(6)
//METAL-NOT: color(7)

//WGSL-DAG:@location(0) [[VAR0:[A-Za-z_0-9]+]]
//WGSL-DAG:@location(1) [[VAR1:[A-Za-z_0-9]+]]
//WGSL-DAG:@location(2) [[VAR2:[A-Za-z_0-9]+]]
//WGSL-DAG:@location(3) [[VAR3:[A-Za-z_0-9]+]]
//WGSL-DAG:@location(4) [[VAR4:[A-Za-z_0-9]+]]
//WGSL-DAG:@location(5) [[VAR5:[A-Za-z_0-9]+]]
//WGSL-DAG:@location(6) [[VAR6:[A-Za-z_0-9]+]]
//WGSL-NOT:@location(7)

//METALLIB: @fragmentMain
//WGSLSPIRV: %fragmentMain = OpFunction %void None

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
    float p1;
    MiddleFragment1 p2;
    MiddleFragment1 p3;
};

struct FragmentStageInput
{
	float4	coarseVertex	: CoarseVertex;
};

struct FragmentStageOutput
{
	TopFragment fragment	: SV_Target;
};

//WGSL: fn fragmentMain{{.*}}-> [[ReturnType:FragmentStageOutput[_0-9]+]]
FragmentStageOutput fragmentMain(FragmentStageInput input)
{
    FragmentStageOutput output;
    output.fragment.p1 = 1;

    output.fragment.p2.p1 = 3;
    output.fragment.p2.p2.p1 = 4;
    output.fragment.p2.p3.p1 = 5;

    output.fragment.p3.p1 = 8;
    output.fragment.p3.p2.p1 = 9;
    output.fragment.p3.p3.p1 = 10;

    // METAL-DAG: ={{.*}}.p1

    // METAL-DAG: ={{.*}}.p2{{.*}}.p1
    // METAL-DAG: ={{.*}}.p2{{.*}}.p2{{.*}}.p1
    // METAL-DAG: ={{.*}}.p2{{.*}}.p3{{.*}}.p1

    // METAL-DAG: ={{.*}}.p3{{.*}}.p1
    // METAL-DAG: ={{.*}}.p3{{.*}}.p2{{.*}}.p1
    // METAL-DAG: ={{.*}}.p3{{.*}}.p3{{.*}}.p1

    // WGSL: var [[ReturnVar:[A-Za-z_0-9]+]] : [[ReturnType]]
    // WGSL-DAG: [[ReturnVar]].[[VAR0]]{{.*}} = {{.*}};
    // WGSL-DAG: [[ReturnVar]].[[VAR1]]{{.*}} = {{.*}};
    // WGSL-DAG: [[ReturnVar]].[[VAR2]]{{.*}} = {{.*}};
    // WGSL-DAG: [[ReturnVar]].[[VAR3]]{{.*}} = {{.*}};
    // WGSL-DAG: [[ReturnVar]].[[VAR4]]{{.*}} = {{.*}};
    // WGSL-DAG: [[ReturnVar]].[[VAR5]]{{.*}} = {{.*}};
    // WGSL-DAG: [[ReturnVar]].[[VAR6]]{{.*}} = {{.*}};

    outputBuffer[0] = 1;
    return output;
}
