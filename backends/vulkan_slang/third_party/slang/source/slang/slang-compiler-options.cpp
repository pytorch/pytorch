#include "slang-compiler-options.h"

#include "slang-compiler.h"

namespace Slang
{
void CompilerOptionSet::load(uint32_t count, slang::CompilerOptionEntry* entries)
{
    for (uint32_t i = 0; i < count; i++)
    {
        CompilerOptionValue value;
        value.kind = entries[i].value.kind;
        value.intValue = entries[i].value.intValue0;
        value.intValue2 = entries[i].value.intValue1;
        if (value.kind == CompilerOptionValueKind::String)
        {
            value.stringValue = entries[i].value.stringValue0;
            value.stringValue2 = entries[i].value.stringValue1;
        }
        add(entries[i].name, value);

        // When we see option EmitSpirvDirectly or EmitSpirvViaGLSL, we will need to
        // translate them to EmitSpirvMethod.
        if (entries[i].name == slang::CompilerOptionName::EmitSpirvDirectly)
        {
            set(slang::CompilerOptionName::EmitSpirvMethod,
                value.intValue != 0 ? SLANG_EMIT_SPIRV_DIRECTLY : SLANG_EMIT_SPIRV_VIA_GLSL);
        }
        else if (entries[i].name == slang::CompilerOptionName::EmitSpirvViaGLSL)
        {
            SlangEmitSpirvMethod current =
                getEnumOption<SlangEmitSpirvMethod>(slang::CompilerOptionName::EmitSpirvMethod);
            if (current == SLANG_EMIT_SPIRV_DEFAULT && value.intValue)
            {
                set(CompilerOptionName::EmitSpirvMethod, SLANG_EMIT_SPIRV_VIA_GLSL);
            }
        }
    }
}

void CompilerOptionSet::writeCommandLineArgs(Session* globalSession, StringBuilder& sb)
{
    for (auto& option : options)
    {
        auto optionInfoIndex = globalSession->m_commandOptions.findOptionByUserValue(
            CommandOptions::UserValue(option.key));
        if (optionInfoIndex == -1)
            continue;
        auto optionInfo = globalSession->m_commandOptions.getOptionAt(optionInfoIndex);
        auto nameCommaIndex = optionInfo.names.indexOf(',');
        if (nameCommaIndex == -1)
            nameCommaIndex = optionInfo.names.getLength();
        auto name = optionInfo.names.head(nameCommaIndex);
        switch (option.key)
        {
        case CompilerOptionName::Capability:
            {
                StringBuilder subBuilder;
                for (auto v : option.value)
                {
                    if (subBuilder.getLength() != 0)
                        subBuilder << "+";
                    if (v.kind == CompilerOptionValueKind::Int)
                        subBuilder << capabilityNameToString((CapabilityName)v.intValue);
                    else
                        subBuilder << v.stringValue;
                }
                if (subBuilder.getLength())
                    sb << " " << optionInfo.names << " " << subBuilder.produceString();
                break;
            }
        case CompilerOptionName::Include:
            for (auto v : option.value)
            {
                sb << " -I \"" << v.stringValue << "\"";
            }
            break;
        case CompilerOptionName::MacroDefine:
            for (auto v : option.value)
            {
                sb << " -D" << v.stringValue;
                if (v.stringValue2.getLength())
                    sb << "=" << v.stringValue2;
            }
            break;
        case CompilerOptionName::VulkanBindShift: // intValue0 (higher 8 bits): kind;
                                                  // intValue0(higher bits): set; intValue1:
                                                  // shift
            for (auto v : option.value)
            {
                uint8_t kind;
                int set, shift;
                v.unpackInt3(kind, set, shift);
                switch ((HLSLToVulkanLayoutOptions::Kind)(kind))
                {
                case HLSLToVulkanLayoutOptions::Kind::UnorderedAccess:
                    sb << " -fvk-u-shift";
                    break;
                case HLSLToVulkanLayoutOptions::Kind::Sampler:
                    sb << " -fvk-s-shift";
                    break;
                case HLSLToVulkanLayoutOptions::Kind::ShaderResource:
                    sb << " -fvk-t-shift";
                    break;
                case HLSLToVulkanLayoutOptions::Kind::ConstantBuffer:
                    sb << " -fvk-b-shift";
                    break;
                default:
                    continue;
                }
                sb << " " << shift << " " << set;
            }
            break;
        case CompilerOptionName::VulkanBindShiftAll: // intValue0: set; intValue1: shift
            for (auto v : option.value)
            {
                sb << " -fvk-all-shift " << v.intValue2 << " " << v.intValue;
            }
            break;
        case CompilerOptionName::VulkanBindGlobals: // intValue0: index; intValue1: set
            for (auto v : option.value)
            {
                sb << " " << name << v.intValue << " " << v.intValue2;
            }
            break;
        case CompilerOptionName::Optimization:
            for (auto v : option.value)
            {
                sb << " -O" << v.intValue;
            }
            break;
        case CompilerOptionName::DownstreamArgs:
            for (auto v : option.value)
            {
                List<UnownedStringSlice> lines;
                StringUtil::split(v.stringValue2.getUnownedSlice(), '\n', lines);
                for (auto l : lines)
                {
                    sb << " -x" << v.stringValue << " " << l.trim();
                }
            }
            break;
        case CompilerOptionName::EmitSpirvDirectly:
        case CompilerOptionName::GLSLForceScalarLayout:
        case CompilerOptionName::ForceDXLayout:
        case CompilerOptionName::MatrixLayoutRow:
        case CompilerOptionName::MatrixLayoutColumn:
        case CompilerOptionName::VulkanInvertY:
        case CompilerOptionName::VulkanUseDxPositionW:
        case CompilerOptionName::VulkanUseEntryPointName:
        case CompilerOptionName::VulkanUseGLLayout:
        case CompilerOptionName::VulkanEmitReflection:
        case CompilerOptionName::EnableEffectAnnotations:
        case CompilerOptionName::DefaultImageFormatUnknown:
        case CompilerOptionName::DisableDynamicDispatch:
        case CompilerOptionName::DisableSpecialization:
        case CompilerOptionName::DumpIntermediates:
        case CompilerOptionName::SkipSPIRVValidation:
            if (option.value.getCount() && option.value[0].intValue != 0)
                sb << " " << name;
            break;
        }
    }
}

void CompilerOptionSet::buildHash(DigestBuilder<SHA1>& builder)
{
    for (auto& kv : options)
    {
        builder.append(kv.key);
        builder.append(kv.value.getCount());
        for (auto& v : kv.value)
        {
            if (v.kind == CompilerOptionValueKind::Int)
            {
                builder.append(v.intValue);
            }
            else
            {
                builder.append(v.stringValue);
                builder.append(v.stringValue2);
            }
        }
    }
}

bool CompilerOptionSet::allowDuplicate(CompilerOptionName name)
{
    switch (name)
    {
    case CompilerOptionName::Include:
    case CompilerOptionName::MacroDefine:
    case CompilerOptionName::WarningsAsErrors:
    case CompilerOptionName::DisableWarning:
    case CompilerOptionName::DisableWarnings:
    case CompilerOptionName::EnableWarning:
    case CompilerOptionName::Capability:
    case CompilerOptionName::DownstreamArgs:
    case CompilerOptionName::VulkanBindShift:
    case CompilerOptionName::VulkanBindShiftAll:
        return true;
    }
    return false;
}
CompilerOptionValue Slang::CompilerOptionSet::getDefault(CompilerOptionName name)
{
    switch (name)
    {
    case CompilerOptionName::Optimization:
        return CompilerOptionValue::fromEnum(OptimizationLevel::Default);
    default:
        return CompilerOptionValue();
    }
}

SlangTargetFlags CompilerOptionSet::getTargetFlags()
{
    SlangTargetFlags result = 0;
    if (getBoolOption(CompilerOptionName::DumpIr))
        result |= SLANG_TARGET_FLAG_DUMP_IR;
    if (getBoolOption(CompilerOptionName::GenerateWholeProgram))
        result |= SLANG_TARGET_FLAG_GENERATE_WHOLE_PROGRAM;
    if (getBoolOption(CompilerOptionName::ParameterBlocksUseRegisterSpaces))
        result |= SLANG_TARGET_FLAG_PARAMETER_BLOCKS_USE_REGISTER_SPACES;
    if (shouldEmitSPIRVDirectly())
        result |= SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY;
    return result;
}

void CompilerOptionSet::setTargetFlags(SlangTargetFlags flags)
{
    set(CompilerOptionName::DumpIr, (flags & SLANG_TARGET_FLAG_DUMP_IR) != 0);
    set(CompilerOptionName::GenerateWholeProgram,
        (flags & SLANG_TARGET_FLAG_GENERATE_WHOLE_PROGRAM) != 0);

    if ((flags & SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY) != 0)
        set(CompilerOptionName::EmitSpirvMethod, SLANG_EMIT_SPIRV_DIRECTLY);
    else
    {
        // We allow to set this flag only when users are not setting the
        // the spirv emit method via CompilerOptionName.
        SlangEmitSpirvMethod current =
            getEnumOption<SlangEmitSpirvMethod>(CompilerOptionName::EmitSpirvMethod);
        if (current != SLANG_EMIT_SPIRV_DIRECTLY)
            set(CompilerOptionName::EmitSpirvMethod, SLANG_EMIT_SPIRV_VIA_GLSL);
    }

    set(CompilerOptionName::ParameterBlocksUseRegisterSpaces,
        (flags & SLANG_TARGET_FLAG_PARAMETER_BLOCKS_USE_REGISTER_SPACES) != 0);
}

void CompilerOptionSet::addTargetFlags(SlangTargetFlags flags)
{
    if ((flags & SLANG_TARGET_FLAG_DUMP_IR))
        set(CompilerOptionName::DumpIr, true);

    if ((flags & SLANG_TARGET_FLAG_GENERATE_WHOLE_PROGRAM) != 0)
        set(CompilerOptionName::GenerateWholeProgram, true);

    if ((flags & SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY) != 0)
        set(CompilerOptionName::EmitSpirvMethod, SLANG_EMIT_SPIRV_DIRECTLY);

    if ((flags & SLANG_TARGET_FLAG_PARAMETER_BLOCKS_USE_REGISTER_SPACES) != 0)
        set(CompilerOptionName::ParameterBlocksUseRegisterSpaces, true);
}
MatrixLayoutMode CompilerOptionSet::getMatrixLayoutMode()
{
    if (getBoolOption(CompilerOptionName::MatrixLayoutRow))
        return kMatrixLayoutMode_RowMajor;
    if (getBoolOption(CompilerOptionName::MatrixLayoutColumn))
        return kMatrixLayoutMode_ColumnMajor;

    return (MatrixLayoutMode)kMatrixLayoutMode_RowMajor;
}

void CompilerOptionSet::setMatrixLayoutMode(MatrixLayoutMode mode)
{
    options.remove(CompilerOptionName::MatrixLayoutColumn);
    options.remove(CompilerOptionName::MatrixLayoutRow);
    if (mode == kMatrixLayoutMode_ColumnMajor)
        set(CompilerOptionName::MatrixLayoutColumn, true);
    if (mode == kMatrixLayoutMode_RowMajor)
        set(CompilerOptionName::MatrixLayoutRow, true);
}

Profile CompilerOptionSet::getProfile()
{
    if (auto profileRaw = getEnumOption<Profile::RawEnum>(CompilerOptionName::Profile))
        return Profile(profileRaw);
    return Profile();
}

void CompilerOptionSet::setProfile(Profile profile)
{
    set(CompilerOptionName::Profile, (int)profile.raw);
}

ProfileVersion CompilerOptionSet::getProfileVersion()
{
    if (auto profileRaw = getEnumOption<Profile::RawEnum>(CompilerOptionName::Profile))
        return Profile(profileRaw).getVersion();
    return ProfileVersion::Unknown;
}

void CompilerOptionSet::setProfileVersion(ProfileVersion version)
{
    Profile profile;
    if (auto profileRaw = getEnumOption<Profile::RawEnum>(CompilerOptionName::Profile))
        profile = Profile(profileRaw);
    profile.setVersion(version);
    set(CompilerOptionName::Profile, (int)profile.raw);
}

void CompilerOptionSet::addCapabilityAtom(CapabilityName cap)
{
    add(CompilerOptionName::Capability, cap);
}

List<String> CompilerOptionSet::getDownstreamArgs(String downstreamToolName)
{
    List<String> result;
    auto downstreamArgsArray = getArray(CompilerOptionName::DownstreamArgs);
    for (auto& argSet : downstreamArgsArray)
    {
        if (argSet.stringValue == downstreamToolName)
        {
            CommandLineArgs args;
            args.deserialize(argSet.stringValue2);
            for (auto arg : args.m_args)
                result.add(arg.value);
            break;
        }
    }
    return result;
}

void CompilerOptionSet::serialize(SerializedOptionsData* outData)
{
    for (auto& option : options)
    {
        for (auto val : option.value)
        {
            slang::CompilerOptionEntry entry = {};
            entry.name = option.key;
            entry.value.kind = val.kind;
            entry.value.intValue0 = val.intValue;
            entry.value.intValue1 = val.intValue2;
            outData->stringPool.add(val.stringValue);
            entry.value.stringValue0 = val.stringValue.getBuffer();
            outData->stringPool.add(val.stringValue2);
            entry.value.stringValue1 = val.stringValue2.getBuffer();
            outData->entries.add(entry);
        }
    }
}

void applySettingsToDiagnosticSink(
    DiagnosticSink* targetSink,
    DiagnosticSink* outputSink,
    CompilerOptionSet& options)
{
    auto disableArray = options.getArray(CompilerOptionName::DisableWarning);
    for (auto& element : disableArray)
    {
        overrideDiagnostic(
            targetSink,
            outputSink,
            element.stringValue.getUnownedSlice(),
            Severity::Warning,
            Severity::Disable);
    }
    disableArray = options.getArray(CompilerOptionName::DisableWarnings);
    for (auto& element : disableArray)
    {
        overrideDiagnostics(
            targetSink,
            outputSink,
            element.stringValue.getUnownedSlice(),
            Severity::Warning,
            Severity::Disable);
    }
    auto enableArray = options.getArray(CompilerOptionName::EnableWarning);
    for (auto& element : enableArray)
    {
        overrideDiagnostics(
            targetSink,
            outputSink,
            element.stringValue.getUnownedSlice(),
            Severity::Warning,
            Severity::Warning);
    }
    auto warningsAsErrorsArray = options.getArray(CompilerOptionName::WarningsAsErrors);
    for (auto& element : warningsAsErrorsArray)
    {
        if (element.stringValue == "all")
            targetSink->setFlag(DiagnosticSink::Flag::TreatWarningsAsErrors);
        else
            overrideDiagnostics(
                targetSink,
                outputSink,
                element.stringValue.getUnownedSlice(),
                Severity::Warning,
                Severity::Error);
    }
}
} // namespace Slang
