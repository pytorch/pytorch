//TEST:SIMPLE(filecheck=CHECK_GLSL): -stage compute -entry computeMain -allow-glsl -target glsl
//TEST:SIMPLE(filecheck=CHECK_SPV): -stage compute -entry computeMain -allow-glsl -target spirv -emit-spirv-directly

// due to implementation limitations (slang-check-expr.cpp, `void visitMemberExpr()`) we are unable to 
// implement and use .length with glsl vectors to test `readonly writeonly` 
//#define TEST_whenVectorLengthIsImplemented

// CHECK_SPV: OpEntryPoint

//CHECK_GLSL: restrict
//CHECK_GLSL: readonly
//CHECK_GLSL: {{(writeonly|readonly)}} {{(writeonly|readonly)}}
//CHECK_GLSL: writeonly
//CHECK_GLSL: coherent
//CHECK_GLSL: volatile
//CHECK_SPV: OpMemberDecorate %{{.*}} 0 Restrict
//CHECK_SPV: OpMemberDecorate %{{.*}} 1 NonWritable
//CHECK_SPV: OpMemberDecorate %{{.*}} 2 {{(NonWritable|NonReadable)}}
//CHECK_SPV: OpMemberDecorate %{{.*}} 2 {{(NonWritable|NonReadable)}}
//CHECK_SPV: OpMemberDecorate %{{.*}} 3 NonReadable
//CHECK_SPV: OpMemberDecorate %{{.*}} 4 Coherent
//CHECK_SPV: OpMemberDecorate %{{.*}} 5 Volatile
//CHECK_SPV: OpDecorate %{{.*}} {{(Restrict|Volatile)}}
restrict volatile buffer Block1
{
    restrict uint data1;
    readonly uint data2;
    readonly writeonly uint data3;
    writeonly uint data4;
    coherent uint data5;
    volatile uint data6;
} inputBuffer1;

//CHECK_GLSL: coherent
//CHECK_GLSL: readonly
//CHECK_SPV: OpMemberDecorate %{{.*}} 0 NonWritable
//CHECK_SPV: OpDecorate %{{.*}} Coherent
coherent buffer Block2 {
    readonly vec4 member1;
    vec4 member2;
}inputBuffer2;

//CHECK_GLSL: {{(coherent|readonly)}} {{(coherent|readonly)}}
//CHECK_GLSL: coherent
//CHECK_SPV: OpMemberDecorate %{{.*}} 0 {{(Coherent|NonWritable)}}
//CHECK_SPV: OpMemberDecorate %{{.*}} 0 {{(Coherent|NonWritable)}}
//CHECK_SPV: OpMemberDecorate %{{.*}} 1 Coherent
//CHECK_SPV: OpDecorate %{{.*}}
buffer Block3 {
    coherent readonly vec4 member1;
    coherent vec4 member2;
}inputBuffer3;

//CHECK_GLSL: {{(coherent|readonly)}} {{(coherent|readonly)}}
//CHECK_GLSL: coherent
//CHECK_GLSL: {{(volatile|writeonly)}} {{(volatile|writeonly)}}
//CHECK_GLSL: readonly
//CHECK_SPV: OpMemberDecorate %{{.*}} 0 {{(Coherent|NonWritable)}}
//CHECK_SPV: OpMemberDecorate %{{.*}} 0 {{(Coherent|NonWritable)}}
//CHECK_SPV: OpMemberDecorate %{{.*}} 1 Coherent
//CHECK_SPV: OpMemberDecorate %{{.*}} 2 {{(NonReadable|Volatile)}}
//CHECK_SPV: OpMemberDecorate %{{.*}} 2 {{(NonReadable|Volatile)}}
//CHECK_SPV: OpMemberDecorate %{{.*}} 3 NonWritable
buffer Block4 {
    struct structTmp
    {
        int val;
    };
    coherent readonly int member1;
    coherent int member2;
    volatile writeonly int member3;
    readonly structTmp member4;
}inputBuffer4;

// CHECK_GLSL: main
layout(local_size_x = 1) in;
void computeMain()
{
    int v = inputBuffer1.data1;
    v += inputBuffer1.data2;
    inputBuffer1.data4 = 1;
    inputBuffer1.data5 = 1;
    inputBuffer1.data6 = 1;
    inputBuffer2.member2 = inputBuffer2.member1;
    inputBuffer3.member2 = inputBuffer3.member1;
    v += inputBuffer4.member1;
    v += inputBuffer4.member2;
    inputBuffer4.member3 = v;
    v += inputBuffer4.member4.val;
#ifdef TEST_whenVectorLengthIsImplemented
    v += inputBuffer4.member3.length();
#endif
    // ensure code is not optimized out
    inputBuffer1.data4 = v;
}