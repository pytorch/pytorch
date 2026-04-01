// unit-test-parameter-usage-reflection.cpp

#include "../../source/core/slang-io.h"
#include "../../source/core/slang-process.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

#include <stdio.h>
#include <stdlib.h>

using namespace Slang;

// Test that the isParameterLocationUsed API works.

SLANG_UNIT_TEST(isParameterLocationUsedReflection)
{
    // Source for a module that contains an undecorated entrypoint.
    const char* userSourceBody = R"(
        Texture2D g_tex : register(t0);
        struct Params {
            Texture2D tex2;
            Texture2D tex3;
        };
        struct Material
        {
            float2 uvScale;
            float2 uvBias;
        }
        ParameterBlock<Params> gParams;
        ConstantBuffer<Material> gcMaterial;
        ParameterBlock<Material> gMaterial;
        [shader("fragment")]
        float4 fragMain(float4 pos:SV_Position, float unused:COLOR0, float4 used:COLOR1) : SV_Target
        {
            return g_tex.Load(int3(0, 0, 0)) + gParams.tex3.Load(int3(0)) + used + gMaterial.uvScale.x + gcMaterial.uvBias.x;
        }
        )";

    auto moduleName = "moduleG" + String(Process::getId());
    String userSource = "import " + moduleName + ";\n" + userSourceBody;
    ComPtr<slang::IGlobalSession> globalSession;
    SLANG_CHECK(slang_createGlobalSession(SLANG_API_VERSION, globalSession.writeRef()) == SLANG_OK);
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_SPIRV;
    targetDesc.profile = globalSession->findProfile("spirv_1_5");
    slang::SessionDesc sessionDesc = {};
    sessionDesc.targetCount = 1;
    sessionDesc.targets = &targetDesc;
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
        "fragMain",
        SLANG_STAGE_FRAGMENT,
        entryPoint.writeRef(),
        diagnosticBlob.writeRef());
    SLANG_CHECK(entryPoint != nullptr);

    ComPtr<slang::IComponentType> compositeProgram;
    slang::IComponentType* components[] = {module, entryPoint.get()};
    session->createCompositeComponentType(
        components,
        2,
        compositeProgram.writeRef(),
        diagnosticBlob.writeRef());
    SLANG_CHECK(compositeProgram != nullptr);

    ComPtr<slang::IComponentType> linkedProgram;
    compositeProgram->link(linkedProgram.writeRef(), nullptr);

    ComPtr<slang::IMetadata> metadata;
    linkedProgram->getTargetMetadata(0, metadata.writeRef(), nullptr);

    bool isUsed = false;
    metadata->isParameterLocationUsed(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT, 0, 0, isUsed);
    SLANG_CHECK(isUsed);

    metadata->isParameterLocationUsed(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT, 0, 1, isUsed);
    SLANG_CHECK(isUsed);

    metadata->isParameterLocationUsed(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT, 0, 2, isUsed);
    SLANG_CHECK(!isUsed);

    metadata->isParameterLocationUsed(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT, 1, 0, isUsed);
    SLANG_CHECK(!isUsed);

    metadata->isParameterLocationUsed(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT, 1, 1, isUsed);
    SLANG_CHECK(isUsed);

    metadata->isParameterLocationUsed(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT, 2, 0, isUsed);
    SLANG_CHECK(isUsed);

    metadata->isParameterLocationUsed(SLANG_PARAMETER_CATEGORY_VARYING_INPUT, 0, 0, isUsed);
    SLANG_CHECK(!isUsed);

    metadata->isParameterLocationUsed(SLANG_PARAMETER_CATEGORY_VARYING_INPUT, 0, 1, isUsed);
    SLANG_CHECK(isUsed);
}
