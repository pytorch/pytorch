#ifndef SLANG_COMPILER_OPTIONS_H
#define SLANG_COMPILER_OPTIONS_H

#include "../core/slang-basic.h"
#include "../core/slang-crypto.h"
#include "slang-generated-capability-defs.h"
#include "slang-profile.h"
#include "slang.h"

namespace Slang
{
using slang::CompilerOptionName;
using slang::CompilerOptionValueKind;
enum MatrixLayoutMode : SlangMatrixLayoutModeIntegral;
enum class LineDirectiveMode : SlangLineDirectiveModeIntegral;
enum class FloatingPointMode : SlangFloatingPointModeIntegral;
enum class OptimizationLevel : SlangOptimizationLevelIntegral;
enum class DebugInfoLevel : SlangDebugInfoLevelIntegral;
enum class CodeGenTarget : SlangCompileTargetIntegral;

struct CompilerOptionValue
{
    CompilerOptionValueKind kind = CompilerOptionValueKind::Int;
    int intValue = 0;
    int intValue2 = 0;
    String stringValue;
    String stringValue2;

    template<typename T>
    static CompilerOptionValue fromEnum(T val)
    {
        static_assert(std::is_enum<T>::value);
        CompilerOptionValue value;
        value.intValue = (int)val;
        value.kind = CompilerOptionValueKind::Int;
        return value;
    }

    static CompilerOptionValue fromInt(int val)
    {
        CompilerOptionValue value;
        value.intValue = val;
        value.kind = CompilerOptionValueKind::Int;
        return value;
    }

    static CompilerOptionValue fromInt2(int val, int val2)
    {
        CompilerOptionValue value;
        value.intValue = val;
        value.intValue2 = val2;
        value.kind = CompilerOptionValueKind::Int;
        return value;
    }

    void unpackInt3(uint8_t& v0, int& v1, int& v2)
    {
        v0 = intValue >> 24;
        v1 = intValue & 0xFFFFFF;
        v2 = intValue2;
    }

    static CompilerOptionValue fromInt3(uint8_t v0, int v1, int v2)
    {
        CompilerOptionValue value;
        value.intValue = (v0 << 24) + (v1 & 0xFFFFFF);
        value.intValue2 = v2;
        value.kind = CompilerOptionValueKind::Int;
        return value;
    }

    static CompilerOptionValue fromString(String val)
    {
        CompilerOptionValue value;
        value.stringValue = val;
        value.kind = CompilerOptionValueKind::String;
        return value;
    }
};

struct SerializedOptionsData
{
    List<slang::CompilerOptionEntry> entries;
    List<String> stringPool;
};

class Session;

struct CompilerOptionSet
{
    void load(uint32_t count, slang::CompilerOptionEntry* entries);

    void buildHash(DigestBuilder<SHA1>& builder);

    static bool allowDuplicate(CompilerOptionName name);

    void writeCommandLineArgs(Session* globalSession, StringBuilder& sb);

    OrderedDictionary<CompilerOptionName, List<CompilerOptionValue>> options;

    bool hasOption(CompilerOptionName name) { return options.containsKey(name); }

    void set(CompilerOptionName name, CompilerOptionValue value)
    {
        if (auto v = options.tryGetValue(name))
        {
            v->clear();
            v->add(value);
            return;
        }
        options[name] = List<CompilerOptionValue>{value};
    }

    void set(CompilerOptionName name, const List<CompilerOptionValue>& value)
    {
        if (auto v = options.tryGetValue(name))
        {
            v->clear();
            v->addRange(value);
            return;
        }
        options[name] = List<CompilerOptionValue>{value};
    }

    void add(CompilerOptionName name, CompilerOptionValue value)
    {
        if (auto v = options.tryGetValue(name))
        {
            v->add(value);
            return;
        }
        options[name] = List<CompilerOptionValue>{value};
    }

    void add(
        CompilerOptionName name,
        const List<CompilerOptionValue>& value,
        bool replaceDuplicate = true)
    {
        if (auto v = options.tryGetValue(name))
        {
            for (auto element : value)
            {
                Index index = v->findFirstIndex(
                    [&](const CompilerOptionValue& existingVal)
                    {
                        if (existingVal.kind == CompilerOptionValueKind::Int)
                            return existingVal.intValue == element.intValue;
                        else
                            return existingVal.stringValue == element.stringValue;
                    });
                if (index != -1)
                {
                    if (replaceDuplicate)
                    {
                        (*v)[index].intValue2 = element.intValue;
                        (*v)[index].stringValue2 = element.stringValue2;
                    }
                }
                else
                {
                    v->add(element);
                }
            }
            return;
        }
        options[name] = List<CompilerOptionValue>{value};
    }

    // Copy settings from other, and replace the current setting.
    void overrideWith(const CompilerOptionSet& other)
    {
        for (auto& kv : other.options)
        {
            if (allowDuplicate(kv.key))
                add(kv.key, kv.value, true);
            else
                set(kv.key, kv.value);
        }
    }

    // Copy settings from other, but do not replace the current setting
    void inheritFrom(const CompilerOptionSet& other)
    {
        for (auto& kv : other.options)
        {
            if (allowDuplicate(kv.key))
                add(kv.key, kv.value, false);
            else
            {
                if (options.containsKey(kv.key))
                    continue;
                set(kv.key, kv.value);
            }
        }
    }

    void add(CompilerOptionName name, int intVal)
    {
        add(name, CompilerOptionValue::fromInt(intVal));
    }
    void add(CompilerOptionName name, int intVal, int intVal2)
    {
        add(name, CompilerOptionValue::fromInt2(intVal, intVal2));
    }
    void add(CompilerOptionName name, uint8_t intVal, int intVal2, int intVal3)
    {
        add(name, CompilerOptionValue::fromInt3(intVal, intVal2, intVal3));
    }
    void add(CompilerOptionName name, String stringVal)
    {
        add(name, CompilerOptionValue::fromString(stringVal));
    }
    void add(CompilerOptionName name, UnownedStringSlice stringVal)
    {
        add(name, CompilerOptionValue::fromString(stringVal));
    }
    void add(CompilerOptionName name, bool boolVal)
    {
        add(name, CompilerOptionValue::fromInt(boolVal ? 1 : 0));
    }

    template<typename EnumType>
    void add(CompilerOptionName name, EnumType enumVal)
    {
        static_assert(std::is_enum<EnumType>::value);
        add(name, (int)enumVal);
    }

    void set(CompilerOptionName name, int intVal)
    {
        set(name, CompilerOptionValue::fromInt(intVal));
    }
    void set(CompilerOptionName name, int intVal1, int intVal2)
    {
        set(name, CompilerOptionValue::fromInt2(intVal1, intVal2));
    }
    void set(CompilerOptionName name, uint8_t intVal1, int intVal2, int intVal3)
    {
        set(name, CompilerOptionValue::fromInt3(intVal1, intVal2, intVal3));
    }
    void set(CompilerOptionName name, String stringVal)
    {
        set(name, CompilerOptionValue::fromString(stringVal));
    }
    void set(CompilerOptionName name, bool boolVal)
    {
        set(name, CompilerOptionValue::fromInt(boolVal ? 1 : 0));
    }

    template<typename EnumType>
    void set(CompilerOptionName name, EnumType enumVal)
    {
        static_assert(std::is_enum<EnumType>::value);
        set(name, (int)enumVal);
    }

    static CompilerOptionValue getDefault(CompilerOptionName name);
    bool getBoolOption(CompilerOptionName name)
    {
        if (auto result = options.tryGetValue(name))
        {
            SLANG_ASSERT(
                result->getCount() != 0 && (*result)[0].kind == CompilerOptionValueKind::Int);
            return result->getCount() != 0 && (*result)[0].intValue != 0;
        }
        return getDefault(name).intValue != 0;
    }
    int getIntOption(CompilerOptionName name)
    {
        if (auto result = options.tryGetValue(name))
        {
            SLANG_ASSERT(
                result->getCount() != 0 && (*result)[0].kind == CompilerOptionValueKind::Int);
            return (*result)[0].intValue;
        }
        return getDefault(name).intValue;
    }
    String getStringOption(CompilerOptionName name)
    {
        if (auto result = options.tryGetValue(name))
        {
            SLANG_ASSERT(
                result->getCount() != 0 && (*result)[0].kind == CompilerOptionValueKind::String);
            return (*result)[0].stringValue;
        }
        return getDefault(name).stringValue;
    }

    template<typename EnumType>
    EnumType getEnumOption(CompilerOptionName name)
    {
        static_assert(std::is_enum<EnumType>::value);
        return (EnumType)getIntOption(name);
    }
    ArrayView<CompilerOptionValue> getArray(CompilerOptionName name)
    {
        if (auto result = options.tryGetValue(name))
        {
            return result->getArrayView();
        }
        return ArrayView<CompilerOptionValue>();
    }

    CodeGenTarget getTarget() { return getEnumOption<CodeGenTarget>(CompilerOptionName::Target); }

    SlangTargetFlags getTargetFlags();
    void setTargetFlags(SlangTargetFlags flags);
    void addTargetFlags(SlangTargetFlags flags);

    MatrixLayoutMode getMatrixLayoutMode();

    void setMatrixLayoutMode(MatrixLayoutMode mode);

    ProfileVersion getProfileVersion();

    Profile getProfile();
    void setProfile(Profile profile);

    void setProfileVersion(ProfileVersion version);

    void addCapabilityAtom(CapabilityName cap);

    void addPreprocessorDefine(String name, String value)
    {
        CompilerOptionValue v;
        v.stringValue = name;
        v.stringValue2 = value;
        v.kind = CompilerOptionValueKind::String;
        add(CompilerOptionName::MacroDefine, v);
    }

    void addSearchPath(String path) { add(CompilerOptionName::Include, String(path)); }

    bool shouldEmitSPIRVDirectly()
    {
        SlangEmitSpirvMethod emitSpvMethod =
            getEnumOption<SlangEmitSpirvMethod>(CompilerOptionName::EmitSpirvMethod);

        return (emitSpvMethod != SlangEmitSpirvMethod::SLANG_EMIT_SPIRV_VIA_GLSL);
    }

    bool shouldUseScalarLayout()
    {
        return getBoolOption(CompilerOptionName::GLSLForceScalarLayout);
    }

    bool shouldUseDXLayout() { return getBoolOption(CompilerOptionName::ForceDXLayout); }

    bool shouldDumpIntermediates() { return getBoolOption(CompilerOptionName::DumpIntermediates); }

    bool shouldDumpIR() { return getBoolOption(CompilerOptionName::DumpIr); }

    bool shouldObfuscateCode() { return getBoolOption(CompilerOptionName::Obfuscate); }

    bool shouldPerformMinimumOptimizations()
    {
        return getBoolOption(CompilerOptionName::MinimumSlangOptimization);
    }

    bool shouldRunNonEssentialValidation()
    {
        return !getBoolOption(CompilerOptionName::DisableNonEssentialValidations);
    }

    bool shouldHaveSourceMap() { return !getBoolOption(CompilerOptionName::DisableSourceMap); }

    FloatingPointMode getFloatingPointMode()
    {
        return getEnumOption<FloatingPointMode>(CompilerOptionName::FloatingPointMode);
    }

    LineDirectiveMode getLineDirectiveMode()
    {
        return getEnumOption<LineDirectiveMode>(CompilerOptionName::LineDirectiveMode);
    }

    OptimizationLevel getOptimizationLevel()
    {
        return getEnumOption<OptimizationLevel>(CompilerOptionName::Optimization);
    }

    DebugInfoLevel getDebugInfoLevel()
    {
        return getEnumOption<DebugInfoLevel>(CompilerOptionName::DebugInformation);
    }

    List<String> getDownstreamArgs(String downstreamToolName);

    void serialize(SerializedOptionsData* outData);
};

class DiagnosticSink;
void applySettingsToDiagnosticSink(
    DiagnosticSink* targetSink,
    DiagnosticSink* outputSink,
    CompilerOptionSet& options);

} // namespace Slang

#endif
