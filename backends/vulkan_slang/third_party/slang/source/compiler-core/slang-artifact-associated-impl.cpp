// slang-artifact-associated-impl.cpp
#include "slang-artifact-associated-impl.h"

#include "../core/slang-array-view.h"
#include "../core/slang-char-util.h"
#include "../core/slang-file-system.h"
#include "../core/slang-io.h"
#include "../core/slang-type-text-util.h"
#include "slang-artifact-diagnostic-util.h"

namespace Slang
{

/* !!!!!!!!!!!!!!!!!!!!!!!!!!! ArtifactDiagnostics !!!!!!!!!!!!!!!!!!!!!!!!!!! */

ArtifactDiagnostics::ArtifactDiagnostics(const ThisType& rhs)
    : ComBaseObject()
    , m_result(rhs.m_result)
    , m_diagnostics(rhs.m_diagnostics)
    , m_raw(rhs.m_raw.getLength() + 1)
{
    // We need to be careful with raw, we want a new *copy* not a non atomic ref counting
    // In initialization we should have enough space
    m_raw.append(rhs.m_raw.getUnownedSlice());

    // Reallocate all the strings
    for (auto& diagnostic : m_diagnostics)
    {
        diagnostic.filePath = m_allocator.allocate(diagnostic.filePath);
        diagnostic.code = m_allocator.allocate(diagnostic.code);
        diagnostic.text = m_allocator.allocate(diagnostic.text);
    }
}

void* ArtifactDiagnostics::clone(const Guid& guid)
{
    ThisType* copy = new ThisType(*this);
    if (auto ptr = copy->castAs(guid))
    {
        return ptr;
    }
    // If the cast fails, we delete the item.
    delete copy;
    return nullptr;
}

void* ArtifactDiagnostics::getInterface(const Guid& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == ICastable::getTypeGuid() ||
        guid == IClonable::getTypeGuid() || guid == IArtifactDiagnostics::getTypeGuid())
    {
        return static_cast<IArtifactDiagnostics*>(this);
    }
    return nullptr;
}

void* ArtifactDiagnostics::getObject(const Guid& guid)
{
    SLANG_UNUSED(guid);
    return nullptr;
}

void* ArtifactDiagnostics::castAs(const Guid& guid)
{
    if (auto intf = getInterface(guid))
    {
        return intf;
    }
    return getObject(guid);
}

void ArtifactDiagnostics::reset()
{
    m_diagnostics.clear();
    m_raw.clear();
    m_result = SLANG_OK;

    m_allocator.deallocateAll();
}

void ArtifactDiagnostics::add(const Diagnostic& inDiagnostic)
{
    Diagnostic diagnostic(inDiagnostic);

    diagnostic.text = m_allocator.allocate(inDiagnostic.text);
    diagnostic.code = m_allocator.allocate(inDiagnostic.code);
    diagnostic.filePath = m_allocator.allocate(inDiagnostic.filePath);

    m_diagnostics.add(diagnostic);
}

void ArtifactDiagnostics::setRaw(const CharSlice& slice)
{
    m_raw.clear();
    m_raw << asStringSlice(slice);
}

void ArtifactDiagnostics::appendRaw(const CharSlice& slice)
{
    if (m_raw.getLength() && m_raw[m_raw.getLength() - 1] != '\n')
    {
        m_raw.appendChar('\n');
    }
    m_raw << asStringSlice(slice);
}

Count ArtifactDiagnostics::getCountAtLeastSeverity(Diagnostic::Severity severity)
{
    Index count = 0;
    for (const auto& msg : m_diagnostics)
    {
        count += Index(Index(msg.severity) >= Index(severity));
    }
    return count;
}

Count ArtifactDiagnostics::getCountBySeverity(Diagnostic::Severity severity)
{
    Index count = 0;
    for (const auto& msg : m_diagnostics)
    {
        count += Index(msg.severity == severity);
    }
    return count;
}

bool ArtifactDiagnostics::hasOfAtLeastSeverity(Diagnostic::Severity severity)
{
    for (const auto& msg : m_diagnostics)
    {
        if (Index(msg.severity) >= Index(severity))
        {
            return true;
        }
    }
    return false;
}

Count ArtifactDiagnostics::getCountByStage(
    Diagnostic::Stage stage,
    Count outCounts[Int(Diagnostic::Severity::CountOf)])
{
    Int count = 0;
    ::memset(outCounts, 0, sizeof(Index) * Int(Diagnostic::Severity::CountOf));
    for (const auto& diagnostic : m_diagnostics)
    {
        if (diagnostic.stage == stage)
        {
            count++;
            outCounts[Index(diagnostic.severity)]++;
        }
    }
    return count++;
}

void ArtifactDiagnostics::removeBySeverity(Diagnostic::Severity severity)
{
    Index count = m_diagnostics.getCount();
    for (Index i = 0; i < count; ++i)
    {
        if (m_diagnostics[i].severity == severity)
        {
            m_diagnostics.removeAt(i);
            i--;
            count--;
        }
    }
}

void ArtifactDiagnostics::maybeAddNote(const CharSlice& in)
{
    ArtifactDiagnosticUtil::maybeAddNote(asStringSlice(in), this);
}

void ArtifactDiagnostics::requireErrorDiagnostic()
{
    // If we find an error, we don't need to add a generic diagnostic
    for (const auto& msg : m_diagnostics)
    {
        if (Index(msg.severity) >= Index(Diagnostic::Severity::Error))
        {
            return;
        }
    }

    Diagnostic diagnostic;
    diagnostic.severity = Diagnostic::Severity::Error;
    diagnostic.text = m_allocator.allocate(m_raw);

    // Add the diagnostic
    m_diagnostics.add(diagnostic);
}

/* static */ UnownedStringSlice _getSeverityText(ArtifactDiagnostic::Severity severity)
{
    typedef ArtifactDiagnostic::Severity Severity;
    switch (severity)
    {
    default:
        return UnownedStringSlice::fromLiteral("Unknown");
    case Severity::Info:
        return UnownedStringSlice::fromLiteral("Info");
    case Severity::Warning:
        return UnownedStringSlice::fromLiteral("Warning");
    case Severity::Error:
        return UnownedStringSlice::fromLiteral("Error");
    }
}

static void _appendCounts(
    const Index counts[Int(ArtifactDiagnostic::Severity::CountOf)],
    StringBuilder& out)
{
    typedef ArtifactDiagnostic::Severity Severity;

    for (Index i = 0; i < Int(Severity::CountOf); i++)
    {
        if (counts[i] > 0)
        {
            out << _getSeverityText(Severity(i)) << "(" << counts[i] << ") ";
        }
    }
}

static void _appendSimplified(
    const Index counts[Int(ArtifactDiagnostic::Severity::CountOf)],
    StringBuilder& out)
{
    typedef ArtifactDiagnostic::Severity Severity;
    for (Index i = 0; i < Int(Severity::CountOf); i++)
    {
        if (counts[i] > 0)
        {
            out << _getSeverityText(Severity(i)) << " ";
        }
    }
}

void ArtifactDiagnostics::calcSummary(ISlangBlob** outBlob)
{
    StringBuilder buf;

    Index counts[Int(Diagnostic::Severity::CountOf)];
    if (getCountByStage(Diagnostic::Stage::Compile, counts) > 0)
    {
        buf << "Compile: ";
        _appendCounts(counts, buf);
        buf << "\n";
    }
    if (getCountByStage(Diagnostic::Stage::Link, counts) > 0)
    {
        buf << "Link: ";
        _appendCounts(counts, buf);
        buf << "\n";
    }

    *outBlob = StringBlob::moveCreate(buf).detach();
}

void ArtifactDiagnostics::calcSimplifiedSummary(ISlangBlob** outBlob)
{
    StringBuilder buf;

    Index counts[Int(Diagnostic::Severity::CountOf)];
    if (getCountByStage(Diagnostic::Stage::Compile, counts) > 0)
    {
        buf << "Compile: ";
        _appendSimplified(counts, buf);
        buf << "\n";
    }
    if (getCountByStage(Diagnostic::Stage::Link, counts) > 0)
    {
        buf << "Link: ";
        _appendSimplified(counts, buf);
        buf << "\n";
    }

    *outBlob = StringBlob::moveCreate(buf).detach();
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ArtifactPostEmitMetadata !!!!!!!!!!!!!!!!!!!!!!!!!!! */

void* ArtifactPostEmitMetadata::getInterface(const Guid& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == ICastable::getTypeGuid() ||
        guid == IArtifactPostEmitMetadata::getTypeGuid())
    {
        return static_cast<IArtifactPostEmitMetadata*>(this);
    }
    return nullptr;
}

void* ArtifactPostEmitMetadata::getObject(const Guid& uuid)
{
    if (uuid == getTypeGuid())
    {
        return this;
    }
    return nullptr;
}

void* ArtifactPostEmitMetadata::castAs(const Guid& guid)
{
    if (auto ptr = getInterface(guid))
    {
        return ptr;
    }
    return getObject(guid);
}

Slice<ShaderBindingRange> ArtifactPostEmitMetadata::getUsedBindingRanges()
{
    return Slice<ShaderBindingRange>(m_usedBindings.getBuffer(), m_usedBindings.getCount());
}

Slice<String> ArtifactPostEmitMetadata::getExportedFunctionMangledNames()
{
    return Slice<String>(
        m_exportedFunctionMangledNames.getBuffer(),
        m_exportedFunctionMangledNames.getCount());
}

SlangResult ArtifactPostEmitMetadata::isParameterLocationUsed(
    SlangParameterCategory category,
    SlangUInt spaceIndex,
    SlangUInt registerIndex,
    bool& outUsed)
{
    for (const auto& range : getUsedBindingRanges())
    {
        if (range.containsBinding((slang::ParameterCategory)category, spaceIndex, registerIndex))
        {
            outUsed = true;
            return SLANG_OK;
        }
    }

    outUsed = false;
    return SLANG_OK;
}


} // namespace Slang
