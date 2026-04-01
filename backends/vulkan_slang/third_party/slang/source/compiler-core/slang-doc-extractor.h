// slang-doc.h
#ifndef SLANG_DOC_EXTRACTOR_H
#define SLANG_DOC_EXTRACTOR_H

#include "../core/slang-basic.h"
#include "slang-lexer.h"
#include "slang-source-loc.h"

namespace Slang
{

enum class MarkupVisibility : uint8_t
{
    Public,   ///< Always available
    Internal, ///< Can be available in more verbose 'internal' documentation
    Hidden,   ///< Not generally available
};

/* Extracts 'markup' from comments in Slang source core. The comments are extracted and associated
in declarations. The association is held in DocMarkup type. The comment style follows the doxygen
style */
class DocMarkupExtractor
{
public:
    typedef uint32_t MarkupFlags;
    struct MarkupFlag
    {
        enum Enum : MarkupFlags
        {
            Before = 0x1,
            After = 0x2,
            IsMultiToken = 0x4, ///< Can use more than one token
            IsBlock = 0x8,      ///<
        };
    };

    // NOTE! Don't change order without fixing isBefore and isAfter
    enum class MarkupType
    {
        None,

        BlockBefore,     /// /**  */ or /*!  */.
        LineBangBefore,  /// //! Can be multiple lines
        LineSlashBefore, /// /// Can be multiple lines
        OrdinaryBlockBefore,
        OrdinaryLineBefore,

        BlockAfter,     /// /*!< */ or /**< */
        LineBangAfter,  /// //!< Can be multiple lines
        LineSlashAfter, /// ///< Can be multiple lines
        OrdinaryLineAfter,
    };

    static bool isBefore(MarkupType type)
    {
        return Index(type) >= Index(MarkupType::BlockBefore) &&
               Index(type) <= Index(MarkupType::OrdinaryLineBefore);
    }
    static bool isAfter(MarkupType type) { return Index(type) >= Index(MarkupType::BlockAfter); }

    struct IndexRange
    {
        SLANG_FORCE_INLINE Index getCount() const { return end - start; }

        Index start;
        Index end;
    };

    enum class Location
    {
        None, ///< No defined location
        Before,
        AfterParam,        ///< Can have trailing , or )
        AfterSemicolon,    ///< Can have a trailing ;
        AfterEnumCase,     ///< Can have a , or before }
        AfterGenericParam, ///< Can have trailing , or >
    };

    static bool isAfter(Location location)
    {
        return Index(location) >= Index(Location::AfterParam);
    }
    static bool isBefore(Location location) { return location == Location::Before; }

    struct FoundMarkup
    {
        void reset()
        {
            location = Location::None;
            type = MarkupType::None;
            range = IndexRange{0, 0};
        }

        Location location = Location::None;
        MarkupType type = MarkupType::None;
        IndexRange range;
    };

    enum SearchStyle
    {
        None,         ///< Cannot be searched for
        EnumCase,     ///< An enum case
        Param,        ///< A parameter in a function/method
        Variable,     ///< A variable-like declaration
        Before,       ///< Only allows before
        Function,     ///< Function/method
        GenericParam, ///< Generic parameter
        Attribute,    ///< Attribute definition
    };

    /// An input search item
    struct SearchItemInput
    {
        SourceLoc sourceLoc;
        SearchStyle searchStyle; ///< The search style when looking for an item
    };

    /// The items will be in source order
    struct SearchItemOutput
    {
        Index viewIndex;            ///< Index into the array of views on the output
        Index inputIndex;           ///< The index to this item in the input
        String text;                ///< The found text
        MarkupVisibility visibilty; ///< Visibility of the item
    };

    struct FindInfo
    {
        SourceView* sourceView; ///< The source view the tokens were generated from
        TokenList* tokenList;   ///< The token list
        Index tokenIndex;       ///< The token index location (where searches start from)
        Index lineIndex;        ///< The line number for the decl
    };

    void setSearchInOrdinaryComments(bool val) { m_searchInOrindaryComments = val; }

    /// Extracts 'markup' doc information for the specified input items
    /// The output is placed in out - with the items now in the source order *not* the order of the
    /// input items The inputIndex on the output holds the input item index The outViews holds the
    /// views specified in viewIndex in the output, which may be useful for determining where the
    /// documentation was placed in source
    SlangResult extract(
        const SearchItemInput* inputItems,
        Index inputCount,
        SourceManager* sourceManager,
        DiagnosticSink* sink,
        List<SourceView*>& outViews,
        List<SearchItemOutput>& out);

    static MarkupFlags getFlags(MarkupType type);
    static MarkupType findMarkupType(const Token& tok);
    static UnownedStringSlice removeStart(MarkupType type, const UnownedStringSlice& comment);

protected:
    /// returns SLANG_E_NOT_FOUND if not found, SLANG_OK on success else an error
    SlangResult _findMarkup(const FindInfo& info, Location location, FoundMarkup& out);

    /// Locations are processed in order, and the first successful used. If found in another
    /// location will issue a warning. returns SLANG_E_NOT_FOUND if not found, SLANG_OK on success
    /// else an error
    SlangResult _findFirstMarkup(
        const FindInfo& info,
        const Location* locs,
        Index locCount,
        FoundMarkup& out,
        Index& outIndex);

    SlangResult _findMarkup(
        const FindInfo& info,
        const Location* locs,
        Index locCount,
        FoundMarkup& out);

    /// Given the decl, the token stream, and the decls tokenIndex, try to find some associated
    /// markup
    SlangResult _findMarkup(const FindInfo& info, SearchStyle searchStyle, FoundMarkup& out);

    /// Given a found markup location extracts the contents of the tokens into out
    SlangResult _extractMarkup(
        const FindInfo& info,
        const FoundMarkup& foundMarkup,
        StringBuilder& out);

    /// Given a location, try to find the first token index that could potentially be markup
    /// Will return -1 if not found
    Index _findStartIndex(const FindInfo& info, Location location);

    /// True if the tok is 'on' lineIndex. Interpretation of 'on' depends on the markup type.
    static bool _isTokenOnLineIndex(
        SourceView* sourceView,
        MarkupType type,
        const Token& tok,
        Index lineIndex);

    DiagnosticSink* m_sink;

    bool m_searchInOrindaryComments = false;
};

} // namespace Slang

#endif
