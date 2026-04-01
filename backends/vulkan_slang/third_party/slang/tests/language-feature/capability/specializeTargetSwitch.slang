//TEST:SIMPLE(filecheck=CHECK_HLSL): -target hlsl -entry main -stage compute -capability _sm_5_1
//TEST:SIMPLE(filecheck=CHECK_GLSL1): -target glsl -entry main -stage compute -capability _GLSL_420
//TEST:SIMPLE(filecheck=CHECK_GLSL1): -target glsl -entry main -stage compute -capability _GLSL_420
//TEST:SIMPLE(filecheck=CHECK_GLSL2): -target glsl -entry main -stage compute -capability _GLSL_330
//TEST:SIMPLE(filecheck=CHECK_GLSL2_NO_UPGRADE): -target glsl -entry main -stage compute -capability _GLSL_330 -DTURN_OFF_LARGER_GLSL_TARGETS 
//TEST:SIMPLE(filecheck=CHECK_METAL): -target cpp -entry main -stage compute -capability image_loadstore
//TEST:SIMPLE(filecheck=CHECK_WILL_ERROR1): -target glsl -entry main -stage compute -capability image_loadstore -DWILL_ERROR1
//TEST:SIMPLE(filecheck=CHECK_WILL_ERROR2): -target glsl -entry main -stage compute -capability _GLSL_130 -DWILL_ERROR2
//TEST:SIMPLE(filecheck=CHECK_GLSL3): -target glsl -entry main -stage compute -capability _GLSL_130

RWTexture1D<int> tex;

//CHECK_HLSL: {{.*}}21{{.*}};

//CHECK_GLSL1: {{.*}}13{{.*}}

//CHECK_GLSL2: {{.*}}11{{.*}}

//CHECK_GLSL2_NO_UPGRADE-NOT: warning 41012
//CHECK_GLSL2_NO_UPGRADE-NOT: error 41012

//CHECK_METAL: {{.*}}30{{.*}}

//CHECK_WILL_ERROR1: error 36109
//CHECK_WILL_ERROR2: error 41011

//CHECK_GLSL3: {{.*}}30{{.*}}

int specialize()
{
    __target_switch
    {
    case spirv_1_0:
        return 1; 
    case spirv_1_1:
        return 2;
    case spirv_1_2:
        return 3;

    case _GLSL_150:
        return 10;
    case _GLSL_330:
        return 11;
#ifndef TURN_OFF_LARGER_GLSL_TARGETS
    case _GLSL_400:
        return 12;
    case _GLSL_410:
        return 13;
#endif
#ifdef WILL_ERROR1
    case image_loadstore:
        return 14;
#endif
    case _sm_5_0:
        return 20;
    case _sm_5_1:
        return 21;
    case _sm_6_0:
        return 21;
#ifndef WILL_ERROR2
    default:
        return 30;
#endif
    }
}

[numthreads(1,1,1)] 
void main()
{
    tex[0] = specialize();
}
