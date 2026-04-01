#include "../../source/compiler-core/slang-diagnostic-sink.h"
#include "../../source/compiler-core/slang-lexer.h"
#include "../../source/compiler-core/slang-perfect-hash.h"
#include "../../source/compiler-core/slang-spirv-core-grammar.h"
#include "../../source/core/slang-dictionary.h"
#include "../../source/core/slang-io.h"
#include "../../source/core/slang-writer.h"

#include <cstdio>

using namespace Slang;

//
// Go from a dictionary to a C++ embedding of a perfect hash
//
template<typename S, typename T, typename F>
String dictToPerfectHash(
    const Dictionary<S, T>& dict,
    const UnownedStringSlice& type,
    const UnownedStringSlice& funcName,
    F valueToString)
{
    HashParams hashParams;
    List<String> names;
    for (const auto& [name, val] : dict)
        names.add(name);
    auto r = minimalPerfectHash(names, hashParams);
    SLANG_ASSERT(r == HashFindResult::Success);
    List<String> values;
    values.reserve(hashParams.destTable.getCount());
    for (const auto& v : hashParams.destTable)
    {
        values.add(valueToString(dict.getValue(v.getUnownedSlice())));
    }
    return perfectHashToEmbeddableCpp(hashParams, type, funcName, values);
}

//
// Go from a dictionary to a C++ embedding of switch table
//
template<typename K, typename V, typename F1, typename F2>
void dictToSwitch(
    const Dictionary<K, V>& dict,
    const char* funName,
    const char* keyType,
    const char* valueType,
    const char* unpackKey,
    const F1 keyToString,
    const F2 valueToAssignmentString,
    WriterHelper& w)
{
    const auto line = [&](const auto& l)
    {
        w.put(l);
        w.put("\n");
    };

    w.print("static bool %s(const %s& k, %s& v)\n", funName, keyType, valueType);
    line("{");
    w.print("    switch(%s)\n", unpackKey);
    line("    {");
    for (const auto& [k, v] : dict)
    {
        const auto kStr = keyToString(k);
        const auto vStr = valueToAssignmentString(v);
        w.print(
            "        case %s:\n"
            "        {\n"
            "            %s;\n"
            "            return true;\n"
            "        }\n",
            kStr.getBuffer(),
            vStr.getBuffer());
    }
    line("        default: return false;");
    line("    }");
    line("}");
    line("");
}

//
// Go from a dictionary to a C++ embedding of switch table, specific to the
// two-level table of a QualifiedEnumValue
//
template<typename V, typename F>
void qualifiedEnumValueNameSwitch(
    const Dictionary<Slang::SPIRVCoreGrammarInfo::QualifiedEnumValue, V>& dict,
    const char* funName,
    const char* keyType,
    const char* valueType,
    const char* unpackKey1,
    const F valueToAssignmentString,
    WriterHelper& w)
{
    const auto line = [&](const auto& l)
    {
        w.put(l);
        w.put("\n");
    };

    using K1 = Slang::SPIRVCoreGrammarInfo::OperandKind;
    using K2 = SpvWord;
    Dictionary<K1, Dictionary<K2, V>> stepDict;
    for (const auto& [k, v] : dict)
    {
        const auto& [k1, k2] = k;
        stepDict[k1][k2] = v;
    }

    w.print("static bool %s(const %s& k, %s& v)\n", funName, keyType, valueType);
    line("{");
    line("    const auto& [k1, k2] = k;");
    w.print("    switch(%s)\n", unpackKey1);
    line("    {");
    for (const auto& [k1, inner] : stepDict)
    {
        const auto k1Str = String(k1.index);
        w.print("        case %s:\n", k1Str.getBuffer());

        line("        switch(k2)");
        line("        {");
        for (const auto& [k2, v] : inner)
        {
            const auto k2Str = String(k2);
            const auto vStr = valueToAssignmentString(v);
            w.print("            case %s: %s; return true;\n", k2Str.getBuffer(), vStr.getBuffer());
        }
        line("            default: return false;");
        line("        }");
    }
    line("        default: return false;");
    line("    }");
    line("}");
    line("");
}

static const char* opClassToString(Slang::SPIRVCoreGrammarInfo::OpInfo::Class c)
{
    switch (c)
    {
#define GO(n)                             \
    case SPIRVCoreGrammarInfo::OpInfo::n: \
        return #n;
        GO(Miscellaneous)
        GO(Debug)
        GO(Annotation)
        GO(Extension)
        GO(ModeSetting)
        GO(TypeDeclaration)
        GO(ConstantCreation)
        GO(Memory)
        GO(Function)
        GO(Image)
        GO(Conversion)
        GO(Composite)
        GO(Arithmetic)
        GO(Bit)
        GO(Relational_and_Logical)
        GO(Derivative)
        GO(ControlFlow)
        GO(Atomic)
        GO(Primitive)
        GO(Barrier)
        GO(Group)
        GO(DeviceSideEnqueue)
        GO(Pipe)
        GO(NonUniform)
        GO(Reserved)
    default:
        GO(Other)
#undef GO
    }
}

//
// Write a C++ embedding of the SPIRVCoreGrammarInfo struct
//
void writeInfo(const char* const outCppPath, const SPIRVCoreGrammarInfo& info)
{
    StringBuilder sb;
    StringWriter writer(&sb, WriterFlags(0));
    WriterHelper w(&writer);
    const auto line = [&](const auto& l)
    {
        w.put(l);
        w.put("\n");
    };

    //
    // Intro
    //
    line("// Source embedding for SPIR-V core grammar");
    line("//");
    line("// This file was carefully generated by a machine,");
    line("// don't even think about modifying it yourself!");
    line("//");
    line("");
    line("#include \"core/slang-smart-pointer.h\"");
    line("#include \"compiler-core/slang-spirv-core-grammar.h\"");
    line("namespace Slang");
    line("{");
    line("using OperandKind = SPIRVCoreGrammarInfo::OperandKind;");
    line("using QualifiedEnumName = SPIRVCoreGrammarInfo::QualifiedEnumName;");
    line("using QualifiedEnumValue = SPIRVCoreGrammarInfo::QualifiedEnumValue;");

    //
    // Each block writes the lookup function for a member table
    // Read the memberAssignments addition to see which one
    //
    List<String> memberAssignments;


    {
        memberAssignments.add("info->opcodes.embedded = &lookupSpvOp;");
        w.put("static ");
        w.put(dictToPerfectHash(
                  info.opcodes.dict,
                  UnownedStringSlice("SpvOp"),
                  UnownedStringSlice("lookupSpvOp"),
                  [](const auto n)
                  {
                      const auto radix = 10;
                      return "static_cast<SpvOp>(" + String(n, radix) + ")";
                  })
                  .getBuffer());
    }

    {
        memberAssignments.add("info->capabilities.embedded = &lookupSpvCapability;");
        w.put("static ");
        w.put(dictToPerfectHash(
                  info.capabilities.dict,
                  UnownedStringSlice("SpvCapability"),
                  UnownedStringSlice("lookupSpvCapability"),
                  [](const auto n)
                  {
                      const auto radix = 10;
                      return "static_cast<SpvCapability>(" + String(n, radix) + ")";
                  })
                  .getBuffer());
    }

    {
        memberAssignments.add("info->allEnumsWithTypePrefix.embedded = &lookupEnumWithTypePrefix;");
        w.put("static ");
        w.put(dictToPerfectHash(
                  info.allEnumsWithTypePrefix.dict,
                  UnownedStringSlice("SpvWord"),
                  UnownedStringSlice("lookupEnumWithTypePrefix"),
                  [](const auto n)
                  {
                      const auto radix = 10;
                      return "SpvWord{" + String(n, radix) + "}";
                  })
                  .getBuffer());
    }

    {
        memberAssignments.add("info->opInfos.embedded = &getOpInfo;");
        dictToSwitch(
            info.opInfos.dict,
            "getOpInfo",
            "SpvOp",
            "SPIRVCoreGrammarInfo::OpInfo",
            "k",
            [&](SpvOp o) { return "Spv" + String(info.opNames.dict.getValue(o)); },
            [](const Slang::SPIRVCoreGrammarInfo::OpInfo& i)
            {
                const char* classStr = opClassToString(i.class_);
                String ret;
                if (i.numOperandTypes)
                {
                    ret.append("const static OperandKind operandTypes[] = {");
                    String operandTypes;
                    for (Index o = 0; o < i.numOperandTypes; ++o)
                    {
                        if (o != 0)
                            ret.append(", ");
                        ret.append("{" + String(i.operandTypes[o].index) + "}");
                    }
                    ret.append("};\n            ");
                }
                ret.append(
                    String("v = {SPIRVCoreGrammarInfo::OpInfo::") + classStr + ", " +
                    String(i.resultTypeIndex) + ", " + String(i.resultIdIndex) + ", " +
                    String(i.minOperandCount) + ", " +
                    (i.maxOperandCount == 0xffff ? String("0xffff") : String(i.maxOperandCount)) +
                    ", " + String(i.numOperandTypes) + ", " +
                    (i.numOperandTypes ? "operandTypes" : "nullptr") + "}");
                return ret;
            },
            w);
    }

    {
        memberAssignments.add("info->opNames.embedded = &getOpName;");
        dictToSwitch(
            info.opNames.dict,
            "getOpName",
            "SpvOp",
            "UnownedStringSlice",
            "k",
            [&](SpvOp o) { return "Spv" + String(info.opNames.dict.getValue(o)); },
            [](const UnownedStringSlice& i)
            { return "v = UnownedStringSlice{\"" + String(i) + "\"}"; },
            w);
    }

    {
        memberAssignments.add("info->operandKinds.embedded = &lookupOperandKind;");
        w.put("static ");
        w.put(dictToPerfectHash(
                  info.operandKinds.dict,
                  UnownedStringSlice("OperandKind"),
                  UnownedStringSlice("lookupOperandKind"),
                  [](const auto n)
                  {
                      const auto radix = 10;
                      return "OperandKind{" + String(n.index, radix) + "}";
                  })
                  .getBuffer());
    }

    {
        memberAssignments.add("info->allEnums.embedded = &lookupQualifiedEnum;");

        // First construct a helper function which will lookup an enum name
        // with a hex prefix representing the kind. This allows us to just
        // reuse the existing string-based perfect hasher
        Dictionary<String, SpvWord> enumDict;
        Index maxNameLength = 0;
        for (const auto& [q, v] : info.allEnums.dict)
        {
            const auto i = q.kind.index;
            String k;
            k.appendChar(char((i >> 4) + 'a'));
            k.appendChar(char((i & 0xf) + 'a'));
            k.append(q.name);
            enumDict.add(k, v);
            maxNameLength = std::max(maxNameLength, k.getLength());
        }
        w.put(dictToPerfectHash(
                  enumDict,
                  UnownedStringSlice("SpvWord"),
                  UnownedStringSlice("lookupEnumWithHexPrefix"),
                  [&](const auto n) { return "SpvWord{" + String(n) + "}"; })
                  .getBuffer());

        // Utilise this helper
        line("static bool lookupQualifiedEnum(const QualifiedEnumName& k, SpvWord& v)");
        line("{");
        line("    static_assert(sizeof(k.kind.index) == 1);");
        w.print("    if(k.name.getLength() > %d)\n", (int)maxNameLength);
        line("        return false;");
        w.print("    char name[%d];\n", (int)maxNameLength + 2);
        line("    name[0] = char((k.kind.index >> 4) + 'a');");
        line("    name[1] = char((k.kind.index & 0xf) + 'a');");
        line("    memcpy(name+2, k.name.begin(), k.name.getLength());");
        line("    return lookupEnumWithHexPrefix(UnownedStringSlice(name, k.name.getLength() + 2), "
             "v);");
        line("}");
        line("");
    }

    {
        memberAssignments.add("info->allEnumNames.embedded = &getQualifiedEnumName;");
        qualifiedEnumValueNameSwitch(
            info.allEnumNames.dict,
            "getQualifiedEnumName",
            "QualifiedEnumValue",
            "UnownedStringSlice",
            "k1.index",
            [](const UnownedStringSlice& i)
            { return "v = UnownedStringSlice{\"" + String(i) + "\"}"; },
            w);
    }

    {
        memberAssignments.add("info->operandKindNames.embedded = &getOperandKindName;");
        dictToSwitch(
            info.operandKindNames.dict,
            "getOperandKindName",
            "OperandKind",
            "UnownedStringSlice",
            "k.index",
            [&](Slang::SPIRVCoreGrammarInfo::OperandKind o) { return String(o.index); },
            [](const UnownedStringSlice& i)
            { return "v = UnownedStringSlice{\"" + String(i) + "\"}"; },
            w);
    }

    {
        memberAssignments.add(
            "info->operandKindUnderneathIds.embedded = &getOperandKindUnderneathId;");
        dictToSwitch(
            info.operandKindUnderneathIds.dict,
            "getOperandKindUnderneathId",
            "OperandKind",
            "OperandKind",
            "k.index",
            [](Slang::SPIRVCoreGrammarInfo::OperandKind o) { return String(o.index); },
            [](Slang::SPIRVCoreGrammarInfo::OperandKind i)
            { return "v = OperandKind{" + String(i.index) + "}"; },
            w);
    }

    //
    // Now write out the function which holds onto the static embedded info table
    //
    line("RefPtr<SPIRVCoreGrammarInfo>& SPIRVCoreGrammarInfo::getEmbeddedVersion()");
    line("{");
    line("    static RefPtr<SPIRVCoreGrammarInfo> embedded = [](){");
    line("        RefPtr<SPIRVCoreGrammarInfo> info = new SPIRVCoreGrammarInfo();");
    for (const auto& a : memberAssignments)
        line(("        " + a).getBuffer());

    //
    line("        return info;");
    line("    }();");
    line("    return embedded;");
    line("}");
    line("}");

    File::writeAllTextIfChanged(outCppPath, sb.getUnownedSlice());
}

int main(int argc, const char* const* argv)
{
    using namespace Slang;

    if (argc != 3)
    {
        fprintf(
            stderr,
            "Usage: %s spirv.core.grammar.json output.cpp\n",
            argc >= 1 ? argv[0] : "slang-spirv-embed-generator");
        return 1;
    }

    const char* const inPath = argv[1];
    const char* const outCppPath = argv[2];

    RefPtr<FileWriter> writer(new FileWriter(stderr, WriterFlag::AutoFlush));
    SourceManager sourceManager;
    sourceManager.initialize(nullptr, nullptr);
    DiagnosticSink sink(&sourceManager, Lexer::sourceLocationLexer);
    sink.writer = writer;

    String contents;
    SLANG_RETURN_ON_FAIL(File::readAllText(inPath, contents));
    PathInfo pathInfo = PathInfo::makeFromString(inPath);
    SourceFile* sourceFile = sourceManager.createSourceFileWithString(pathInfo, contents);
    SourceView* sourceView = sourceManager.createSourceView(sourceFile, nullptr, SourceLoc());

    RefPtr<SPIRVCoreGrammarInfo> info = SPIRVCoreGrammarInfo::loadFromJSON(*sourceView, sink);

    writeInfo(outCppPath, *info);

    return 0;
}
