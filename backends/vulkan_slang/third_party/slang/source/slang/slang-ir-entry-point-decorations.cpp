#include "slang-ir-entry-point-decorations.h"

#include "compiler-core/slang-diagnostic-sink.h"
#include "core/slang-signal.h"
#include "core/slang-string.h"
#include "core/slang-type-text-util.h"
#include "slang-compiler.h"
#include "slang-ir-insts.h"
#include "slang-ir.h"
#include "slang-options.h"

namespace Slang
{

class CheckEntryPointDecorationsContext
{
public:
    CheckEntryPointDecorationsContext(IRModule* module, CodeGenTarget target, DiagnosticSink* sink)
        : m_module(module), m_target(target), m_sink(sink)
    {
    }

    void check()
    {
        for (auto inst : m_module->getGlobalInsts())
        {
            const auto func = as<IRFunc>(inst);
            if (!func)
                continue;
            const auto entryPointDecoration = func->findDecoration<IREntryPointDecoration>();
            if (!entryPointDecoration)
                continue;

            checkEntryPoint(func, entryPointDecoration->getProfile().getStage());
        }
    }

private:
    void checkEntryPoint(IRFunc* entryPoint, Stage stage)
    {
        for (auto decoration : entryPoint->getDecorations())
        {
            if (auto outputTopologyDecoration = as<IROutputTopologyDecoration>(decoration))
            {
                checkOutputTopologyDecoration(outputTopologyDecoration, stage);
            }
        }
    }

    void checkOutputTopologyDecoration(IROutputTopologyDecoration* decoration, Stage stage)
    {
        if (stage == Stage::Mesh)
        {
            const auto outputTopologyType = OutputTopologyType(decoration->getTopologyType());
            if (isTargetGLSL() || isTargetSPIRV() || isTargetMetal())
            {
                if (outputTopologyType != OutputTopologyType::Point &&
                    outputTopologyType != OutputTopologyType::Line &&
                    outputTopologyType != OutputTopologyType::Triangle)
                {
                    diagnoseInvalidMeshStageOutputTopology(
                        decoration,
                        "'point', 'line', 'triangle'");
                }
            }
            else if (isTargetHLSL())
            {
                if (outputTopologyType != OutputTopologyType::Line &&
                    outputTopologyType != OutputTopologyType::Triangle)
                {
                    diagnoseInvalidMeshStageOutputTopology(decoration, "'line', 'triangle'");
                }
            }
            else
            {
                SLANG_UNEXPECTED("Invalid compilation target for mesh stage");
            }
        }
    }

    void diagnoseInvalidMeshStageOutputTopology(
        IROutputTopologyDecoration* decoration,
        String validTopologies)
    {
        m_sink->diagnose(
            decoration,
            Diagnostics::invalidMeshStageOutputTopology,
            decoration->getTopology()->getStringSlice(),
            TypeTextUtil::getCompileTargetName(SlangCompileTarget(m_target)),
            validTopologies);
    }

    bool isTargetHLSL() const { return m_target == CodeGenTarget::HLSL; }

    bool isTargetGLSL() const { return m_target == CodeGenTarget::GLSL; }

    bool isTargetSPIRV() const
    {
        return m_target == CodeGenTarget::SPIRV || m_target == CodeGenTarget::SPIRVAssembly;
    }

    bool isTargetMetal() const
    {
        return m_target == CodeGenTarget::Metal || m_target == CodeGenTarget::MetalLib ||
               m_target == CodeGenTarget::MetalLibAssembly;
    }

    IRModule* m_module;
    const CodeGenTarget m_target;
    DiagnosticSink* m_sink;
};

void checkEntryPointDecorations(IRModule* module, CodeGenTarget target, DiagnosticSink* sink)
{
    CheckEntryPointDecorationsContext(module, target, sink).check();
}

OutputTopologyType convertOutputTopologyStringToEnum(String rawOutputTopology)
{
    auto name = rawOutputTopology.toLower();

    OutputTopologyType outputTopologyType = OutputTopologyType::Unknown;

#define CASE(ID, NAME)                               \
    if (name == String(#NAME).toLower())             \
    {                                                \
        outputTopologyType = OutputTopologyType::ID; \
    }                                                \
    else

    OUTPUT_TOPOLOGY_TYPES(CASE)
#undef CASE
    {
        outputTopologyType = OutputTopologyType::Unknown;
        // no match
    }
    return outputTopologyType;
}

} // namespace Slang
