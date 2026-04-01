#include "unit-test.h"

#include "compiler-core/slang-lexer.h"
#include "compiler-core/slang-name-convention-util.h"
#include "compiler-core/slang-source-loc.h"
#include "core/slang-io.h"
#include "identifier-lookup.h"
#include "node-tree.h"
#include "options.h"
#include "parser.h"

namespace CppParse
{
using namespace Slang;


struct TestState
{
    TestState()
        : m_slicePool(StringSlicePool::Style::Default)
    {
        m_identifierLookup.initDefault(UnownedStringSlice::fromLiteral("SLANG_"));

        m_sourceManager.initialize(nullptr, nullptr);

        m_sink.init(&m_sourceManager, Lexer::sourceLocationLexer);

        m_namePool.setRootNamePool(&m_rootNamePool);

        // We don't require marker
        m_options.m_requireMark = false;
    }

    RootNamePool m_rootNamePool;
    Options m_options;
    SourceManager m_sourceManager;
    DiagnosticSink m_sink;
    NamePool m_namePool;
    StringSlicePool m_slicePool;
    IdentifierLookup m_identifierLookup;
};

static const char someSource[] = "class ISomeInterface\n"
                                 "{\n"
                                 "    public:\n"
                                 "    virtual int SLANG_MCALL someMethod(int a, int b) const = 0;\n"
                                 "    virtual float SLANG_MCALL anotherMethod(float a) = 0;\n"
                                 "};\n"
                                 "\n"
                                 "struct SomeStruct\n"
                                 "{\n"
                                 "    SomeStruct() = default;\n"
                                 "    SomeStruct(float v = 0.0f):b(v) {}\n"
                                 "    ~SomeStruct() {}\n"
                                 "    int a = 10; \n"
                                 "    float b; \n"
                                 "    int another[10];\n"
                                 "    const char* yetAnother = nullptr;\n"
                                 "};\n"
                                 "\n"
                                 "enum SomeEnum\n"
                                 "{\n"
                                 "    Value,\n"
                                 "    Another = 10,\n"
                                 "};\n"
                                 "\n"
                                 "typedef int (*SomeFunc)(int a);\n"
                                 "\n"
                                 "typedef SomeEnum AliasEnum;\n"
                                 "void someFunc(int a, float b) { }\n"
                                 "namespace Blah {\n"
                                 "int add(int a, int b) { return a + b; }\n"
                                 "unsigned add(unsigned a, unsigned b) { return a + b; }\n"
                                 "}\n";


/* static */ SlangResult UnitTestUtil::run()
{
    {
        TestState state;

        NodeTree tree(&state.m_slicePool, &state.m_namePool, &state.m_identifierLookup);

        UnownedStringSlice contents = UnownedStringSlice::fromLiteral(someSource);
        PathInfo pathInfo = PathInfo::makeFromString("source.h");

        SourceManager* sourceManager = &state.m_sourceManager;

        SourceFile* sourceFile = sourceManager->createSourceFileWithString(pathInfo, contents);
        SourceOrigin* sourceOrigin = tree.addSourceOrigin(sourceFile, state.m_options);

        Parser parser(&tree, &state.m_sink);


        {
            const Node::Kind enableKinds[] = {
                Node::Kind::Enum,
                Node::Kind::EnumClass,
                Node::Kind::EnumCase,
                Node::Kind::TypeDef};
            parser.setKindsEnabled(enableKinds, SLANG_COUNT_OF(enableKinds));
        }

        SlangResult res = parser.parse(sourceOrigin, &state.m_options);

        if (state.m_sink.outputBuffer.getLength())
        {
            printf("%s\n", state.m_sink.outputBuffer.getBuffer());
        }

        if (SLANG_FAILED(res))
        {
            return res;
        }

        {
            StringBuilder buf;
            tree.getRootNode()->dump(0, buf);

            SLANG_UNUSED(buf);
        }
    }

    return SLANG_OK;
}

} // namespace CppParse
