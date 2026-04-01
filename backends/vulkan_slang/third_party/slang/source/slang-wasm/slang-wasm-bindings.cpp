#include "slang-wasm.h"

#include <emscripten/bind.h>
#include <slang-com-ptr.h>

using namespace emscripten;

EMSCRIPTEN_BINDINGS(slang)
{
    constant("SLANG_OK", SLANG_OK);

    function("getLastError", &slang::wgsl::getLastError);

    function("getCompileTargets", &slang::wgsl::getCompileTargets);

    class_<slang::wgsl::GlobalSession>("GlobalSession")
        .function(
            "createSession",
            &slang::wgsl::GlobalSession::createSession,
            allow_raw_pointers());

    function("createGlobalSession", &slang::wgsl::createGlobalSession, allow_raw_pointers());

    class_<slang::wgsl::Session>("Session")
        .function(
            "loadModuleFromSource",
            &slang::wgsl::Session::loadModuleFromSource,
            allow_raw_pointers())
        .function(
            "createCompositeComponentType",
            &slang::wgsl::Session::createCompositeComponentType,
            allow_raw_pointers());

    class_<slang::wgsl::ComponentType>("ComponentType")
        .function("link", &slang::wgsl::ComponentType::link, allow_raw_pointers())
        .function("getEntryPointCode", &slang::wgsl::ComponentType::getEntryPointCode)
        .function("getEntryPointCodeBlob", &slang::wgsl::ComponentType::getEntryPointCodeBlob)
        .function("getTargetCodeBlob", &slang::wgsl::ComponentType::getTargetCodeBlob)
        .function("getTargetCode", &slang::wgsl::ComponentType::getTargetCode)
        .function("getLayout", &slang::wgsl::ComponentType::getLayout, allow_raw_pointers())
        .function("loadStrings", &slang::wgsl::ComponentType::loadStrings, allow_raw_pointers());

    class_<slang::wgsl::TypeLayoutReflection>("TypeLayoutReflection")
        .function(
            "getDescriptorSetDescriptorRangeType",
            &slang::wgsl::TypeLayoutReflection::getDescriptorSetDescriptorRangeType);

    enum_<slang::Modifier::ID>("ModifierID")
        .value("Shared", slang::Modifier::ID::Shared)
        .value("NoDiff", slang::Modifier::ID::NoDiff)
        .value("Static", slang::Modifier::ID::Static)
        .value("Const", slang::Modifier::ID::Const)
        .value("Export", slang::Modifier::ID::Export)
        .value("Extern", slang::Modifier::ID::Extern)
        .value("Differentiable", slang::Modifier::ID::Differentiable)
        .value("Mutating", slang::Modifier::ID::Mutating)
        .value("In", slang::Modifier::ID::In)
        .value("Out", slang::Modifier::ID::Out)
        .value("InOut", slang::Modifier::ID::InOut);

    class_<slang::Modifier>("Modifier");

    class_<slang::wgsl::VariableReflection>("VariableReflection")
        .function("getName", &slang::wgsl::VariableReflection::getName)
        .function(
            "findModifier",
            &slang::wgsl::VariableReflection::findModifier,
            allow_raw_pointers())
        .function("getType", &slang::wgsl::VariableReflection::getType, allow_raw_pointers())
        .function("getUserAttributeCount", &slang::wgsl::VariableReflection::getUserAttributeCount)
        .function(
            "getUserAttributeByIndex",
            &slang::wgsl::VariableReflection::getUserAttributeByIndex,
            allow_raw_pointers())
        .function("hasDefaultValue", &slang::wgsl::VariableReflection::hasDefaultValue);


    class_<slang::wgsl::VariableLayoutReflection>("VariableLayoutReflection")
        .function("getName", &slang::wgsl::VariableLayoutReflection::getName)
        .function(
            "getTypeLayout",
            &slang::wgsl::VariableLayoutReflection::getTypeLayout,
            allow_raw_pointers())
        .function("getBindingIndex", &slang::wgsl::VariableLayoutReflection::getBindingIndex);

    class_<slang::wgsl::GenericReflection>("GenericReflection")
        .function("getName", &slang::wgsl::GenericReflection::getName)
        .function("getTypeParameterCount", &slang::wgsl::GenericReflection::getTypeParameterCount)
        .function("getValueParameterCount", &slang::wgsl::GenericReflection::getValueParameterCount)
        .function("getInnerKind", &slang::wgsl::GenericReflection::getInnerKind)
        .function("asDecl", &slang::wgsl::GenericReflection::asDecl, allow_raw_pointers())
        // .function(
        //     "getTypeParameterConstraintCount",
        //     &slang::wgsl::GenericReflection::getTypeParameterConstraintCount,
        //     allow_raw_pointers())
        .function(
            "getTypeParameter",
            &slang::wgsl::GenericReflection::getTypeParameter,
            allow_raw_pointers())
        .function(
            "getValueParameter",
            &slang::wgsl::GenericReflection::getValueParameter,
            allow_raw_pointers())
        .function(
            "getInnerDecl",
            &slang::wgsl::GenericReflection::getInnerDecl,
            allow_raw_pointers())
        .function(
            "getOuterGenericContainer",
            &slang::wgsl::GenericReflection::getOuterGenericContainer,
            allow_raw_pointers());

    enum_<SlangDeclKind>("SlangDeclKind")
        .value(
            "SLANG_DECL_KIND_UNSUPPORTED_FOR_REFLECTION",
            SlangDeclKind::SLANG_DECL_KIND_UNSUPPORTED_FOR_REFLECTION)
        .value("SLANG_DECL_KIND_STRUCT", SlangDeclKind::SLANG_DECL_KIND_STRUCT)
        .value("SLANG_DECL_KIND_FUNC", SlangDeclKind::SLANG_DECL_KIND_FUNC)
        .value("SLANG_DECL_KIND_MODULE", SlangDeclKind::SLANG_DECL_KIND_MODULE)
        .value("SLANG_DECL_KIND_GENERIC", SlangDeclKind::SLANG_DECL_KIND_GENERIC)
        .value("SLANG_DECL_KIND_VARIABLE", SlangDeclKind::SLANG_DECL_KIND_VARIABLE)
        .value("SLANG_DECL_KIND_NAMESPACE", SlangDeclKind::SLANG_DECL_KIND_NAMESPACE);

    class_<slang::wgsl::DeclReflection>("DeclReflection")
        .function("getName", &slang::wgsl::DeclReflection::getName)
        .function("getChildrenCount", &slang::wgsl::DeclReflection::getChildrenCount)
        .function("getKind", &slang::wgsl::DeclReflection::getKind)
        .function("getChild", &slang::wgsl::DeclReflection::getChild, allow_raw_pointers())
        .function("getType", &slang::wgsl::DeclReflection::getType, allow_raw_pointers())
        .function("asVariable", &slang::wgsl::DeclReflection::asVariable, allow_raw_pointers())
        .function("asFunction", &slang::wgsl::DeclReflection::asFunction, allow_raw_pointers())
        .function("asGeneric", &slang::wgsl::DeclReflection::asGeneric, allow_raw_pointers())
        .function("getParent", &slang::wgsl::DeclReflection::getParent, allow_raw_pointers());

    enum_<slang::DeclReflection::Kind>("DeclReflectionKind")
        .value("Unsupported", slang::DeclReflection::Kind::Unsupported)
        .value("Struct", slang::DeclReflection::Kind::Struct)
        .value("Func", slang::DeclReflection::Kind::Func)
        .value("Module", slang::DeclReflection::Kind::Module)
        .value("Generic", slang::DeclReflection::Kind::Generic)
        .value("Variable", slang::DeclReflection::Kind::Variable)
        .value("Namespace", slang::DeclReflection::Kind::Namespace);

    enum_<slang::TypeReflection::ScalarType>("ScalarType")
        .value("None", slang::TypeReflection::ScalarType::None)
        .value("Void", slang::TypeReflection::ScalarType::Void)
        .value("Bool", slang::TypeReflection::ScalarType::Bool)
        .value("Int32", slang::TypeReflection::ScalarType::Int32)
        .value("UInt32", slang::TypeReflection::ScalarType::UInt32)
        .value("Int64", slang::TypeReflection::ScalarType::Int64)
        .value("UInt64", slang::TypeReflection::ScalarType::UInt64)
        .value("Float16", slang::TypeReflection::ScalarType::Float16)
        .value("Float32", slang::TypeReflection::ScalarType::Float32)
        .value("Float64", slang::TypeReflection::ScalarType::Float64)
        .value("Int8", slang::TypeReflection::ScalarType::Int8)
        .value("UInt8", slang::TypeReflection::ScalarType::UInt8)
        .value("Int16", slang::TypeReflection::ScalarType::Int16)
        .value("UInt16", slang::TypeReflection::ScalarType::UInt16);

    class_<slang::wgsl::TypeReflection>("TypeReflection")
        .function("getScalarType", &slang::wgsl::TypeReflection::getScalarType)
        .function("getKind", &slang::wgsl::TypeReflection::getKind);

    enum_<slang::TypeReflection::Kind>("TypeReflectionKind")
        .value("None", slang::TypeReflection::Kind::None)
        .value("Struct", slang::TypeReflection::Kind::Struct)
        .value("Array", slang::TypeReflection::Kind::Array)
        .value("Matrix", slang::TypeReflection::Kind::Matrix)
        .value("Vector", slang::TypeReflection::Kind::Vector)
        .value("Scalar", slang::TypeReflection::Kind::Scalar)
        .value("ConstantBuffer", slang::TypeReflection::Kind::ConstantBuffer)
        .value("Resource", slang::TypeReflection::Kind::Resource)
        .value("SamplerState", slang::TypeReflection::Kind::SamplerState)
        .value("TextureBuffer", slang::TypeReflection::Kind::TextureBuffer)
        .value("ShaderStorageBuffer", slang::TypeReflection::Kind::ShaderStorageBuffer)
        .value("ParameterBlock", slang::TypeReflection::Kind::ParameterBlock)
        .value("GenericTypeParameter", slang::TypeReflection::Kind::GenericTypeParameter)
        .value("Interface", slang::TypeReflection::Kind::Interface)
        .value("OutputStream", slang::TypeReflection::Kind::OutputStream)
        .value("Specialized", slang::TypeReflection::Kind::Specialized)
        .value("Feedback", slang::TypeReflection::Kind::Feedback)
        .value("Pointer", slang::TypeReflection::Kind::Pointer)
        .value("DynamicResource", slang::TypeReflection::Kind::DynamicResource);


    class_<slang::wgsl::UserAttribute>("UserAttribute")
        .function("getName", &slang::wgsl::UserAttribute::getName)
        .function("getArgumentCount", &slang::wgsl::UserAttribute::getArgumentCount)
        .function(
            "getArgumentType",
            &slang::wgsl::UserAttribute::getArgumentType,
            allow_raw_pointers())
        .function(
            "getArgumentValueString",
            &slang::wgsl::UserAttribute::getArgumentValueString,
            allow_raw_pointers())
        .function(
            "getArgumentValueFloat",
            &slang::wgsl::UserAttribute::getArgumentValueFloat,
            allow_raw_pointers());

    class_<slang::wgsl::FunctionReflection>("FunctionReflection")
        .function("getName", &slang::wgsl::FunctionReflection::getName)
        .function("getUserAttributeCount", &slang::wgsl::FunctionReflection::getUserAttributeCount)
        .function(
            "getUserAttributeByIndex",
            &slang::wgsl::FunctionReflection::getUserAttributeByIndex,
            allow_raw_pointers());

    class_<slang::wgsl::EntryPointReflection>("EntryPointReflection")
        .function(
            "getComputeThreadGroupSize",
            &slang::wgsl::EntryPointReflection::getComputeThreadGroupSize);

    class_<slang::wgsl::EntryPointReflection::ThreadGroupSize>("ThreadGroupSize")
        .property("x", &slang::wgsl::EntryPointReflection::ThreadGroupSize::x)
        .property("y", &slang::wgsl::EntryPointReflection::ThreadGroupSize::y)
        .property("z", &slang::wgsl::EntryPointReflection::ThreadGroupSize::z);

    class_<slang::wgsl::ProgramLayout>("ProgramLayout")
        .function("toJsonObject", &slang::wgsl::ProgramLayout::toJsonObject)
        .function("getParameterCount", &slang::wgsl::ProgramLayout::getParameterCount)
        .function(
            "getParameterByIndex",
            &slang::wgsl::ProgramLayout::getParameterByIndex,
            allow_raw_pointers())
        .function(
            "getGlobalParamsTypeLayout",
            &slang::wgsl::ProgramLayout::getGlobalParamsTypeLayout,
            allow_raw_pointers())
        .function(
            "findEntryPointByName",
            &slang::wgsl::ProgramLayout::findEntryPointByName,
            allow_raw_pointers())
        .function(
            "findFunctionByName",
            &slang::wgsl::ProgramLayout::findFunctionByName,
            allow_raw_pointers());

    enum_<slang::BindingType>("BindingType")
        .value("Unknown", slang::BindingType::Unknown)
        .value("Texture", slang::BindingType::Texture)
        .value("ConstantBuffer", slang::BindingType::ConstantBuffer)
        .value("MutableRawBuffer", slang::BindingType::MutableRawBuffer)
        .value("MutableTypedBuffer", slang::BindingType::MutableTypedBuffer)
        .value("MutableTexture", slang::BindingType::MutableTexture);

    class_<slang::wgsl::Module, base<slang::wgsl::ComponentType>>("Module")
        .function(
            "findEntryPointByName",
            &slang::wgsl::Module::findEntryPointByName,
            allow_raw_pointers())
        .function(
            "findAndCheckEntryPoint",
            &slang::wgsl::Module::findAndCheckEntryPoint,
            allow_raw_pointers())
        .function(
            "getDefinedEntryPoint",
            &slang::wgsl::Module::getDefinedEntryPoint,
            allow_raw_pointers())
        .function("getDefinedEntryPointCount", &slang::wgsl::Module::getDefinedEntryPointCount);

    value_object<slang::wgsl::Error>("Error")
        .field("type", &slang::wgsl::Error::type)
        .field("result", &slang::wgsl::Error::result)
        .field("message", &slang::wgsl::Error::message);

    class_<slang::wgsl::EntryPoint, base<slang::wgsl::ComponentType>>("EntryPoint")
        .function("getName", &slang::wgsl::EntryPoint::getName, allow_raw_pointers());

    register_vector<std::string>("StringList");
    register_optional<std::vector<std::string>>();

    value_object<slang::wgsl::lsp::Position>("Position")
        .field("line", &slang::wgsl::lsp::Position::line)
        .field("character", &slang::wgsl::lsp::Position::character);

    value_object<slang::wgsl::lsp::Range>("Range")
        .field("start", &slang::wgsl::lsp::Range::start)
        .field("end", &slang::wgsl::lsp::Range::end);

    value_object<slang::wgsl::lsp::Location>("Location")
        .field("uri", &slang::wgsl::lsp::Location::uri)
        .field("range", &slang::wgsl::lsp::Location::range);
    register_vector<slang::wgsl::lsp::Location>("LocationList");
    register_optional<std::vector<slang::wgsl::lsp::Location>>();

    value_object<slang::wgsl::lsp::TextEdit>("TextEdit")
        .field("range", &slang::wgsl::lsp::TextEdit::range)
        .field("text", &slang::wgsl::lsp::TextEdit::text);
    register_optional<slang::wgsl::lsp::TextEdit>();
    register_vector<slang::wgsl::lsp::TextEdit>("TextEditList");
    register_optional<std::vector<slang::wgsl::lsp::TextEdit>>();

    value_object<slang::wgsl::lsp::MarkupContent>("MarkupContent")
        .field("kind", &slang::wgsl::lsp::MarkupContent::kind)
        .field("value", &slang::wgsl::lsp::MarkupContent::value);
    register_optional<slang::wgsl::lsp::MarkupContent>();

    value_object<slang::wgsl::lsp::Hover>("Hover")
        .field("contents", &slang::wgsl::lsp::Hover::contents)
        .field("range", &slang::wgsl::lsp::Hover::range);
    register_optional<slang::wgsl::lsp::Hover>();

    value_object<slang::wgsl::lsp::CompletionItem>("CompletionItem")
        .field("label", &slang::wgsl::lsp::CompletionItem::label)
        .field("kind", &slang::wgsl::lsp::CompletionItem::kind)
        .field("detail", &slang::wgsl::lsp::CompletionItem::detail)
        .field("documentation", &slang::wgsl::lsp::CompletionItem::documentation)
        .field("textEdit", &slang::wgsl::lsp::CompletionItem::textEdit)
        .field("data", &slang::wgsl::lsp::CompletionItem::data)
        .field("commitCharacters", &slang::wgsl::lsp::CompletionItem::commitCharacters);
    register_optional<slang::wgsl::lsp::CompletionItem>();
    register_vector<slang::wgsl::lsp::CompletionItem>("CompletionItemList");
    register_optional<std::vector<slang::wgsl::lsp::CompletionItem>>();

    value_object<slang::wgsl::lsp::CompletionContext>("CompletionContext")
        .field("triggerKind", &slang::wgsl::lsp::CompletionContext::triggerKind)
        .field("triggerCharacter", &slang::wgsl::lsp::CompletionContext::triggerCharacter);

    value_array<std::array<uint32_t, 2>>("array_uint_2")
        .element(emscripten::index<0>())
        .element(emscripten::index<1>());

    value_object<slang::wgsl::lsp::ParameterInformation>("ParameterInformation")
        .field("label", &slang::wgsl::lsp::ParameterInformation::label)
        .field("documentation", &slang::wgsl::lsp::ParameterInformation::documentation);

    register_vector<slang::wgsl::lsp::ParameterInformation>("ParameterInformationList");

    value_object<slang::wgsl::lsp::SignatureInformation>("SignatureInformation")
        .field("label", &slang::wgsl::lsp::SignatureInformation::label)
        .field("documentation", &slang::wgsl::lsp::SignatureInformation::documentation)
        .field("parameters", &slang::wgsl::lsp::SignatureInformation::parameters);

    register_vector<slang::wgsl::lsp::SignatureInformation>("SignatureInformationList");

    value_object<slang::wgsl::lsp::SignatureHelp>("SignatureHelp")
        .field("signatures", &slang::wgsl::lsp::SignatureHelp::signatures)
        .field("activeSignature", &slang::wgsl::lsp::SignatureHelp::activeSignature)
        .field("activeParameter", &slang::wgsl::lsp::SignatureHelp::activeParameter);
    register_optional<slang::wgsl::lsp::SignatureHelp>();

    value_object<slang::wgsl::lsp::DocumentSymbol>("DocumentSymbol")
        .field("name", &slang::wgsl::lsp::DocumentSymbol::name)
        .field("detail", &slang::wgsl::lsp::DocumentSymbol::detail)
        .field("kind", &slang::wgsl::lsp::DocumentSymbol::kind)
        .field("range", &slang::wgsl::lsp::DocumentSymbol::range)
        .field("selectionRange", &slang::wgsl::lsp::DocumentSymbol::selectionRange)
        .field("children", &slang::wgsl::lsp::DocumentSymbol::children);

    register_vector<slang::wgsl::lsp::DocumentSymbol>("DocumentSymbolList");
    register_optional<std::vector<slang::wgsl::lsp::DocumentSymbol>>();

    value_object<slang::wgsl::lsp::Diagnostics>("Diagnostics")
        .field("code", &slang::wgsl::lsp::Diagnostics::code)
        .field("range", &slang::wgsl::lsp::Diagnostics::range)
        .field("severity", &slang::wgsl::lsp::Diagnostics::severity)
        .field("message", &slang::wgsl::lsp::Diagnostics::message);
    register_vector<slang::wgsl::lsp::Diagnostics>("DiagnosticsList");
    register_optional<std::vector<slang::wgsl::lsp::Diagnostics>>();

    register_vector<uint32_t>("Uint32List");
    register_optional<std::vector<uint32_t>>();

    class_<slang::wgsl::lsp::LanguageServer>("LanguageServer")
        .function(
            "didOpenTextDocument",
            &slang::wgsl::lsp::LanguageServer::didOpenTextDocument,
            allow_raw_pointers())
        .function(
            "didCloseTextDocument",
            &slang::wgsl::lsp::LanguageServer::didCloseTextDocument,
            allow_raw_pointers())
        .function(
            "didChangeTextDocument",
            &slang::wgsl::lsp::LanguageServer::didChangeTextDocument,
            allow_raw_pointers())
        .function("hover", &slang::wgsl::lsp::LanguageServer::hover, allow_raw_pointers())
        .function(
            "gotoDefinition",
            &slang::wgsl::lsp::LanguageServer::gotoDefinition,
            allow_raw_pointers())
        .function("completion", &slang::wgsl::lsp::LanguageServer::completion, allow_raw_pointers())
        .function(
            "completionResolve",
            &slang::wgsl::lsp::LanguageServer::completionResolve,
            allow_raw_pointers())
        .function(
            "semanticTokens",
            &slang::wgsl::lsp::LanguageServer::semanticTokens,
            allow_raw_pointers())
        .function(
            "signatureHelp",
            &slang::wgsl::lsp::LanguageServer::signatureHelp,
            allow_raw_pointers())
        .function(
            "documentSymbol",
            &slang::wgsl::lsp::LanguageServer::documentSymbol,
            allow_raw_pointers())
        .function(
            "getDiagnostics",
            &slang::wgsl::lsp::LanguageServer::getDiagnostics,
            allow_raw_pointers());

    function(
        "createLanguageServer",
        &slang::wgsl::lsp::createLanguageServer,
        return_value_policy::take_ownership());
};
