#ifndef SLANG_CORE_COMMAND_OPTIONS_WRITER_H
#define SLANG_CORE_COMMAND_OPTIONS_WRITER_H

#include "slang-command-options.h"

namespace Slang
{

class CommandOptionsWriter : public RefObject
{
public:
    typedef CommandOptions::CategoryKind CategoryKind;
    typedef CommandOptions::NameKey NameKey;
    typedef CommandOptions::LookupKind LookupKind;

    enum class Style
    {
        Text,           ///< Suitable for output to a terminal
        Markdown,       ///< Markdown
        NoLinkMarkdown, ///< Markdown without links
    };

    static ConstArrayView<NamesDescriptionValue> getStyleInfos();

    struct Options
    {
        Style style = Style::Text; ///< The style
        Index lineLength = 120;    ///< The maximum amount of characters on a line
        UnownedStringSlice indent = toSlice("  ");
        ;
    };

    /// Append descirption for a category
    void appendDescriptionForCategory(CommandOptions* options, Index categoryIndex);
    /// Appends a description of all of the options
    void appendDescription(CommandOptions* options);

    /// Get the builder that string is being written to
    StringBuilder& getBuilder() { return m_builder; }

    static RefPtr<CommandOptionsWriter> create(const Options& options);


protected:
    /// Append descirption for a category
    virtual void appendDescriptionForCategoryImpl(Index categoryIndex) = 0;
    /// Appends a description of all of the options
    virtual void appendDescriptionImpl() = 0;

    // Ctor, use create to create a writer
    CommandOptionsWriter(const Options& options);

    /// Get the length of the current line in ascii chars/bytes
    Count _getCurrentLineLength();

    /// Indentation/wrapping
    void _requireIndent(Count indentCount);
    void _appendWrappedIndented(
        Count indentCount,
        List<UnownedStringSlice>& slices,
        const UnownedStringSlice& delimit);

    CommandOptions* m_commandOptions = nullptr;

    StringSlicePool m_pool;
    StringBuilder m_builder;
    Options m_options;
};

} // namespace Slang

#endif
