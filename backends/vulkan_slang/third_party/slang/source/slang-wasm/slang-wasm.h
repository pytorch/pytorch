#pragma once

#include <emscripten.h>
#include <emscripten/val.h>
#include <optional>
#include <slang-com-ptr.h>
#include <slang.h>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

/**
The web assembly binding here is designed to make javascript code as simple and native as possible.
The big issue being handled here is lifetime management of objects created in the Slang API.

The idea here is to make lifetime management as coarse grained as possible from the javascript side.
Only two types of objects need to be explicitly deleted by javascript: GlobalSession and Session.

All the remaining objects returned by member functions of Session will have their lifetime managed
by the owning session in the C++ side. This way, the javascript code will never need to worry about
freeing small objects like ComponentType, EntryPoint, Module, TypeLayoutReflection,
VariableLayoutReflection, ProgramLayout etc.

When a Session is no longer needed, the javascript code should explicitly delete it, this will allow
us to free all the objects we allocated from the session in one single explicit call.

By making explicit memory management as coarse grained as possible, we are making memory management
efficient, simple, and less error prone.
*/

namespace Slang
{
class LanguageServerCore;
}

namespace slang
{
namespace wgsl
{

class TypeLayoutReflection;
class TypeReflection;
class VariableLayoutReflection;
class VariableReflection;
class FunctionReflection;
class GenericReflection;


class Error
{
public:
    // Can be
    // "USER": User did not call the API correctly
    // "INTERNAL": Slang failed due to a bug
    std::string type;
    std::string message;
    SlangResult result;
};

Error getLastError();

// returns mapping of codegen target from string to SlangCompileTarget
// in the form of [{name: STRING, value: INT}, ...].
emscripten::val getCompileTargets();

class DeclReflection
{
public:
    std::string getName();
    slang::DeclReflection::Kind getKind();
    uint32_t getChildrenCount();
    slang::wgsl::DeclReflection* getChild(uint32_t index);
    slang::wgsl::TypeReflection* getType();
    slang::wgsl::VariableReflection* asVariable();
    slang::wgsl::FunctionReflection* asFunction();
    slang::wgsl::GenericReflection* asGeneric();
    slang::wgsl::DeclReflection* getParent();

    slang::DeclReflection* interface() const { return (slang::DeclReflection*)this; };
};

class GenericReflection
{

public:
    std::string getName();
    slang::wgsl::DeclReflection* asDecl();
    uint32_t getTypeParameterCount();
    slang::wgsl::VariableReflection* getTypeParameter(unsigned index);
    uint32_t getValueParameterCount();
    slang::wgsl::VariableReflection* getValueParameter(unsigned index);
    // uint32_t getTypeParameterConstraintCount(VariableReflection* typeParam);
    // slang::wgsl::TypeReflection* getTypeParameterConstraintType(
    //     VariableReflection* typeParam,
    //     unsigned index);
    slang::wgsl::DeclReflection* getInnerDecl();
    SlangDeclKind getInnerKind();
    slang::wgsl::GenericReflection* getOuterGenericContainer();
    // TypeReflection* getConcreteType(slang::wgsl::VariableReflection* typeParam);
    // int64_t getConcreteIntVal(slang::wgsl::VariableReflection* valueParam);
    // GenericReflection* applySpecializations(slang::wgsl::GenericReflection* generic);

    slang::GenericReflection* interface() const { return (slang::GenericReflection*)this; };
};

class TypeReflection
{

public:
    slang::TypeReflection::ScalarType getScalarType();
    slang::TypeReflection::Kind getKind();
    uint32_t getFieldCount();
    slang::wgsl::VariableReflection* getFieldByIndex(uint32_t index);

    slang::TypeReflection* interface() const { return (slang::TypeReflection*)this; };
};


class UserAttribute
{

public:
    std::string getName();
    uint32_t getArgumentCount();
    float getArgumentValueFloat(uint32_t index);
    std::string getArgumentValueString(uint32_t index);
    slang::wgsl::TypeReflection* getArgumentType(uint32_t index);
    slang::UserAttribute* interface() const { return (slang::UserAttribute*)this; };
};


class VariableReflection
{
public:
    std::string getName();
    slang::wgsl::TypeReflection* getType();
    Modifier* findModifier(Modifier::ID id);
    uint32_t getUserAttributeCount();
    slang::wgsl::UserAttribute* getUserAttributeByIndex(uint32_t index);
    // slang::wgsl::UserAttribute* findUserAttributeByName(SlangSession* globalSession, std::string
    // name);
    bool hasDefaultValue();
    // slang::wgsl::GenericReflection* getGenericContainer();
    // slang::wgsl::VariableReflection* applySpecializations(slang::wgsl::GenericReflection*
    // generic);
    slang::VariableReflection* interface() const { return (slang::VariableReflection*)this; }
};


class TypeLayoutReflection
{
public:
    BindingType getDescriptorSetDescriptorRangeType(uint32_t setIndex, uint32_t rangeIndex);
    slang::TypeLayoutReflection* interface() const { return (slang::TypeLayoutReflection*)this; }
};

class VariableLayoutReflection
{
public:
    std::string getName();
    slang::wgsl::TypeLayoutReflection* getTypeLayout();
    uint32_t getBindingIndex();

    slang::VariableLayoutReflection* interface() const
    {
        return (slang::VariableLayoutReflection*)this;
    }
};

class EntryPointReflection
{

public:
    struct ThreadGroupSize
    {
        uint32_t x;
        uint32_t y;
        uint32_t z;
    };

    ThreadGroupSize getComputeThreadGroupSize();
    slang::EntryPointReflection* interface() const { return (slang::EntryPointReflection*)this; };
};
class FunctionReflection
{

public:
    std::string getName();
    uint32_t getUserAttributeCount();
    slang::wgsl::UserAttribute* getUserAttributeByIndex(uint32_t index);
    slang::FunctionReflection* interface() const { return (slang::FunctionReflection*)this; };
};

class ProgramLayout
{
public:
    uint32_t getParameterCount();
    slang::wgsl::VariableLayoutReflection* getParameterByIndex(uint32_t index);

    slang::wgsl::TypeLayoutReflection* getGlobalParamsTypeLayout();

    slang::wgsl::EntryPointReflection* findEntryPointByName(std::string name);

    slang::wgsl::FunctionReflection* findFunctionByName(std::string name);

    slang::ProgramLayout* interface() const { return (slang::ProgramLayout*)this; }

    emscripten::val toJsonObject();
};

class Session;
class ComponentType
{
public:
    IComponentType* m_interface;
    Session* m_session;

public:
    ComponentType(slang::IComponentType* interface, Session* session)
        : m_interface(interface), m_session(session)
    {
    }

    // Returns ComponentType or null.
    emscripten::val link();

    std::string getEntryPointCode(int entryPointIndex, int targetIndex);

    // Returns UInt8Array or null.
    emscripten::val getEntryPointCodeBlob(int entryPointIndex, int targetIndex);
    std::string getTargetCode(int targetIndex);

    // Returns UInt8Array or null.
    emscripten::val getTargetCodeBlob(int targetIndex);

    slang::wgsl::ProgramLayout* getLayout(uint32_t targetIndex);

    slang::IComponentType* interface() const { return m_interface; }

    // returns [{hash: HASH, string: STRING}, ...]
    emscripten::val loadStrings();
};

class EntryPoint : public ComponentType
{
public:
    EntryPoint(slang::IComponentType* interface, Session* session)
        : ComponentType(interface, session)
    {
    }
    std::string getName() const
    {
        return entryPointInterface()->getFunctionReflection()->getName();
    }

private:
    slang::IEntryPoint* entryPointInterface() const
    {
        return static_cast<slang::IEntryPoint*>(interface());
    }
};

class Module : public ComponentType
{
public:
    Module(slang::IComponentType* interface, Session* session)
        : ComponentType(interface, session)
    {
    }

    // Returns EntryPoint or null.
    emscripten::val findEntryPointByName(const std::string& name);

    // Returns EntryPoint or null.
    emscripten::val findAndCheckEntryPoint(const std::string& name, int stage);

    // Returns EntryPoint or null.
    emscripten::val getDefinedEntryPoint(int index);

    int getDefinedEntryPointCount();

    slang::IModule* moduleInterface() const { return static_cast<slang::IModule*>(interface()); }
};

class Session
{
public:
    Session(slang::ISession* interface)
        : m_interface(interface)
    {
    }
    ~Session();

    // Returns Module or null.
    emscripten::val loadModuleFromSource(
        const std::string& slangCode,
        const std::string& name,
        const std::string& path);

    // `components` is a javascript array of ComponentType/Module/EntryPoint objects.
    // Returns ComponentType or null.
    emscripten::val createCompositeComponentType(emscripten::val components);

    slang::ISession* interface() const { return m_interface; }

    void addComponentType(slang::IComponentType* componentType)
    {
        m_componentTypes.push_back(Slang::ComPtr<slang::IComponentType>(componentType));
    }

private:
    std::vector<Slang::ComPtr<slang::IComponentType>> m_componentTypes;
    Slang::ComPtr<slang::ISession> m_interface;
};

class GlobalSession
{
public:
    GlobalSession(slang::IGlobalSession* interface)
        : m_interface(interface)
    {
    }

    Session* createSession(int compileTarget);

    slang::IGlobalSession* interface() const { return m_interface; }

private:
    Slang::ComPtr<slang::IGlobalSession> m_interface;
};

GlobalSession* createGlobalSession();

namespace lsp
{
struct Position
{
    int line = -1;
    int character = -1;
};

struct Range
{
    Position start;
    Position end;
};

struct Location
{
    std::string uri;
    Range range;
};

struct TextEdit
{
    Range range;
    std::string text;
};

struct MarkupContent
{
    std::string kind;
    std::string value;
};

struct Hover
{
    MarkupContent contents;
    Range range;
};

struct CompletionItem
{
    std::string label;
    int kind;
    std::string detail;
    std::string data;
    std::optional<MarkupContent> documentation;
    std::optional<TextEdit> textEdit;
    std::optional<std::vector<std::string>> commitCharacters;
};

struct CompletionContext
{
    int triggerKind = 1;
    std::string triggerCharacter;
};

struct ParameterInformation
{
    uint32_t label[2] = {0, 0};
    MarkupContent documentation;
};

struct SignatureInformation
{
    std::string label;
    MarkupContent documentation;
    std::vector<ParameterInformation> parameters;
};

struct SignatureHelp
{
    std::vector<SignatureInformation> signatures;
    uint32_t activeSignature = 0;
    uint32_t activeParameter = 0;
};

struct DocumentSymbol
{
    std::string name;
    std::string detail;
    int kind = 0;
    Range range;
    Range selectionRange;
    std::vector<DocumentSymbol> children;
};

struct Diagnostics
{
    std::string code;
    Range range;
    std::string message;
    int severity;
};

class LanguageServer
{
private:
    Slang::LanguageServerCore* m_core = nullptr;
    void init();

public:
    LanguageServer();
    ~LanguageServer();
    void didOpenTextDocument(std::string uri, std::string text);
    void didCloseTextDocument(std::string uri);
    void didChangeTextDocument(std::string uri, const std::vector<lsp::TextEdit>& changes);
    std::optional<lsp::Hover> hover(std::string uri, lsp::Position position);
    std::optional<std::vector<lsp::Location>> gotoDefinition(
        std::string uri,
        lsp::Position position);
    std::optional<std::vector<lsp::CompletionItem>> completion(
        std::string uri,
        lsp::Position position,
        CompletionContext context);
    std::optional<lsp::CompletionItem> completionResolve(lsp::CompletionItem args);
    std::optional<std::vector<uint32_t>> semanticTokens(std::string uri);
    std::optional<lsp::SignatureHelp> signatureHelp(std::string uri, lsp::Position position);
    std::optional<std::vector<lsp::DocumentSymbol>> documentSymbol(std::string uri);
    std::optional<std::vector<lsp::Diagnostics>> getDiagnostics(std::string uri);
};

LanguageServer* createLanguageServer();
} // namespace lsp

} // namespace wgsl
} // namespace slang
