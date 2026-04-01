//TEST:SIMPLE(filecheck=CHK):-target hlsl -stage compute -entry computeMain
//TEST:SIMPLE(filecheck=HLSL):-target hlsl -stage compute -entry computeMain
//TEST:SIMPLE(filecheck=GLSL):-target glsl -stage compute -entry computeMain
//TEST:SIMPLE(filecheck=SPV):-target spirv -stage compute -entry computeMain

// TODO: requires changes in parser
//static_assert(true, "PASS_global");
//static_assert(false, "FAIL_global");

__generic<T:IArithmetic>
void TEST_specialize()
{
    static_assert(true,   "TEST_specialize true");
    static_assert(T is float, "TEST_specialize T_is_float");
    static_assert(T is int,   "TEST_specialize T_is_int");
}

void TEST_target_switch()
{
    static_assert(false, "TEST_target_switch all");
    __target_switch
    {
    case hlsl:
        static_assert(false, "TEST_target_switch hlsl");
        return;
    case glsl:
        static_assert(false, "TEST_target_switch glsl");
        return;
    default:
        static_assert(false, "TEST_target_switch default");
        return;
    }
}

__generic<T:IArithmetic>
struct MyType
{
    // TODO: requires changes in parser
    //static_assert(true, "PASS_struct");
    //static_assert(false, "FAIL_struct");

    __init()
    {
        static_assert(true, "MyType init true");
        static_assert(false, "MyType init false");
    }
};

__generic<T:IFloat>
extension MyType<T>
{
};

[numthreads(1,1,1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    // CHK-DAG: error {{.*}} TEST_specialize T_is_int
    // CHK-DAG:{{.*}} TEST_specialize<float>
    TEST_specialize<float>();

    // CHK-DAG: error {{.*}} TEST_specialize T_is_float
    // CHK-DAG:{{.*}} TEST_specialize<int>
    TEST_specialize<int>();

    //HLSL: error {{.*}} TEST_target_switch all
    //HLSL: error {{.*}} TEST_target_switch hlsl
    //HLSL-NOT: error {{.*}} TEST_target_switch glsl
    //HLSL-NOT: error {{.*}} TEST_target_switch default
    //
    //GLSL: error {{.*}} TEST_target_switch all
    //GLSL-NOT: error {{.*}} TEST_target_switch hlsl
    //GLSL: error {{.*}} TEST_target_switch glsl
    //GLSL-NOT: error {{.*}} TEST_target_switch default
    //
    //SPV: error {{.*}} TEST_target_switch all
    //SPV-NOT: error {{.*}} TEST_target_switch hlsl
    //SPV-NOT: error {{.*}} TEST_target_switch glsl
    //SPV: error {{.*}} TEST_target_switch default
    //
    TEST_target_switch();

    //CHK-NOT: error {{.*}} MyType init true
    //CHK: error {{.*}} MyType init false
    MyType<float> obj;

    //CHK: ([[#@LINE+1]]): error
    static_assert(dispatchThreadID.x == 0, "not constant");
}

