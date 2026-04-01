// parse-diagnostic-util.h

#ifndef PARSE_DIAGNOSTIC_UTIL_H
#define PARSE_DIAGNOSTIC_UTIL_H

#include "../../source/compiler-core/slang-artifact-diagnostic-util.h"
#include "../../source/compiler-core/slang-downstream-compiler.h"
#include "../../source/core/slang-string-util.h"
#include "../../source/core/slang-string.h"
#include "slang-com-ptr.h"

struct ParseDiagnosticUtil
{
    struct OutputInfo
    {
        int resultCode;
        Slang::String stdError;
        Slang::String stdOut;
    };

    /// We need a way to identify downstream compilers and others - specifically here Slang.
    /// Ideally we'd have an enum that included Slang.
    /// Just adding to SlangPassThrough doesn't seem right. If we had an enumeration with a
    /// more appropriate name, then including downstream and slang compilers wouldn't be a problem.
    /// So for now this is punted, and this type is used to represent possible compiler identities.
    struct CompilerIdentity
    {
        typedef CompilerIdentity ThisType;

        enum Type
        {
            Unknown,
            Slang,
            DownstreamCompiler,
        };

        static CompilerIdentity make(Type type, SlangPassThrough downstreamCompiler)
        {
            CompilerIdentity ident;
            ident.m_type = type;
            ident.m_downstreamCompiler = downstreamCompiler;
            return ident;
        }
        static CompilerIdentity make(SlangPassThrough downstreamCompiler)
        {
            return make(Type::DownstreamCompiler, downstreamCompiler);
        }
        static CompilerIdentity makeSlang() { return make(Type::Slang, SLANG_PASS_THROUGH_NONE); }

        bool operator==(const ThisType& rhs) const
        {
            return m_type == rhs.m_type && m_downstreamCompiler == rhs.m_downstreamCompiler;
        }
        bool operator!=(const ThisType& rhs) const { return !(*this == rhs); }

        Type m_type = Type::Unknown;
        SlangPassThrough m_downstreamCompiler = SLANG_PASS_THROUGH_NONE;
    };

    typedef uint32_t EqualityFlags;
    struct EqualityFlag
    {
        enum Enum : EqualityFlags
        {
            IgnoreLineNos = 0x1,
        };
    };

    typedef SlangResult (*LineParser)(
        Slang::SliceAllocator& allocator,
        const Slang::UnownedStringSlice& line,
        Slang::List<Slang::UnownedStringSlice>& lineSlices,
        Slang::ArtifactDiagnostic& outDiagnostic);

    /// Given a compiler identity returns a line parsing function.
    static LineParser getLineParser(const CompilerIdentity& compilerIdentity);

    /// For a 'generic' (as in uses DownstreamCompiler mechanism) line parsing
    static SlangResult parseGenericLine(
        Slang::SliceAllocator& allocator,
        const Slang::UnownedStringSlice& line,
        Slang::List<Slang::UnownedStringSlice>& lineSlices,
        Slang::ArtifactDiagnostic& outDiagnostic);

    /// For parsing diagnostics from Slang
    static SlangResult parseSlangLine(
        Slang::SliceAllocator& allocator,
        const Slang::UnownedStringSlice& line,
        Slang::List<Slang::UnownedStringSlice>& lineSlices,
        Slang::ArtifactDiagnostic& outDiagnostic);

    /// Parse diagnostics into output text
    static SlangResult parseDiagnostics(
        const Slang::UnownedStringSlice& inText,
        Slang::IArtifactDiagnostics* diagnostics);

    /// Parse diagnostics with known compiler identity.
    /// If the prefix is empty, it is assumed there is no prefix and it won't be checked.
    static SlangResult parseDiagnostics(
        const Slang::UnownedStringSlice& inText,
        const CompilerIdentity& identity,
        const Slang::UnownedStringSlice& prefix,
        Slang::IArtifactDiagnostics* diagnostics);

    /// Given the file output style used by tests, get components of the output into Diagnostic
    static SlangResult parseOutputInfo(const Slang::UnownedStringSlice& in, OutputInfo& out);

    /// Given a line split it into slices - taking into account compiler output, path
    /// considerations, and potentially line prefixing
    static SlangResult splitDiagnosticLine(
        const CompilerIdentity& compilerIdentity,
        const Slang::UnownedStringSlice& line,
        const Slang::UnownedStringSlice& linePrefix,
        Slang::List<Slang::UnownedStringSlice>& outSlices);

    /// Give text of diagnostic determine which compiler the output is from
    static SlangResult identifyCompiler(
        const Slang::UnownedStringSlice& in,
        CompilerIdentity& outIdentity);

    /// Determines if the diagnostics in a and b (they are parsed via parseDiagnostics) are equal,
    /// taking into account flags Note! If the parse of either a or b fails, then equality is
    /// returns as false.
    static bool areEqual(
        const Slang::UnownedStringSlice& a,
        const Slang::UnownedStringSlice& b,
        EqualityFlags flags);
};

#endif // PARSE_DIAGNOSTIC_UTIL_H
