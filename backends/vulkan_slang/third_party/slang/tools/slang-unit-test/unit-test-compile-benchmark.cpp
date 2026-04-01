// unit-test-glsl-compile.cpp

#include "../../source/core/slang-io.h"
#include "../../source/core/slang-process.h"
#include "../../tools/platform/performance-counter.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

#include <stdio.h>
#include <stdlib.h>

using namespace Slang;

// Test the compilation API for cross-compiling glsl source to SPIRV.

SLANG_UNIT_TEST(compileBenchmark)
{
    const char* userSourceBody = R"(
// shader.slang

struct PushConstantCompute
{
  uint64_t bufferAddress;
  uint     numVertices;
};

struct Vertex
{
  float3 position;
};


[[vk::push_constant]]
ConstantBuffer<PushConstantCompute> pushConst;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
  uint index = threadIdx.x;

  if(index >= pushConst.numVertices)
   return;

  Vertex* vertices = (Vertex*)pushConst.bufferAddress;
 
  float angle = (index + 1) * 2.3f;

  float3 vertex = vertices[index].position;

  float cosAngle = cos(angle);
  float sinAngle = sin(angle);
  float3x3 rotationMatrix = float3x3(
    cosAngle, -sinAngle, 0.0,
    sinAngle,  cosAngle, 0.0,
         0.0,       0.0, 1.0
  );

  float3 rotatedVertex = mul(rotationMatrix, vertex);

  vertices[index].position = rotatedVertex;
}
        )";
    ComPtr<slang::IGlobalSession> globalSession;
    SlangGlobalSessionDesc globalDesc = {};
    globalDesc.enableGLSL = false;
    SLANG_CHECK(slang_createGlobalSession2(&globalDesc, globalSession.writeRef()) == SLANG_OK);
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_SPIRV;
    targetDesc.profile = globalSession->findProfile("spirv_1_5");
    slang::SessionDesc sessionDesc = {};
    sessionDesc.targetCount = 1;
    sessionDesc.targets = &targetDesc;

    auto start = platform::PerformanceCounter::now();
    for (int pass = 0; pass < 100; pass++)
    {
        ComPtr<slang::ISession> session;
        SLANG_CHECK(globalSession->createSession(sessionDesc, session.writeRef()) == SLANG_OK);

        ComPtr<slang::IBlob> diagnosticBlob;
        auto module = session->loadModuleFromSourceString(
            "m",
            "m.slang",
            userSourceBody,
            diagnosticBlob.writeRef());
        SLANG_CHECK(module != nullptr);

        ComPtr<slang::IEntryPoint> entryPoint;
        module->findAndCheckEntryPoint(
            "main",
            SLANG_STAGE_VERTEX,
            entryPoint.writeRef(),
            diagnosticBlob.writeRef());

        slang::IComponentType* componentTypes[2] = {module, entryPoint.get()};
        ComPtr<slang::IComponentType> composedProgram;
        session->createCompositeComponentType(
            componentTypes,
            2,
            composedProgram.writeRef(),
            diagnosticBlob.writeRef());

        ComPtr<slang::IComponentType> linkedProgram;
        composedProgram->link(linkedProgram.writeRef(), diagnosticBlob.writeRef());

        ComPtr<slang::IBlob> code;
        linkedProgram->getEntryPointCode(0, 0, code.writeRef(), diagnosticBlob.writeRef());
    }
    auto time = platform::PerformanceCounter::getElapsedTimeInSeconds(start);
    getTestReporter()->addExecutionTime(time);
}
