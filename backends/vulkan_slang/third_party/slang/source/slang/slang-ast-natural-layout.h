#ifndef SLANG_AST_NATURAL_LAYOUT_H
#define SLANG_AST_NATURAL_LAYOUT_H

#include "slang-ast-base.h"

namespace Slang
{

struct NaturalSize
{
    typedef NaturalSize ThisType;

    // We are going to use 0 as invalid for alignment. This has a few nice propeties
    //
    // * Will naturally produce 0 size when used with `calcAligned` operation
    // * Is fast to test
    // * Is easy to make a fast 'max' such that a max with invalid always returns `invalid`
    //
    // We also desire that when invalid the `size` member is 0.
    // This is so that equality testing doesn't require anything special.
    SLANG_FORCE_INLINE static Count calcAligned(Count size, Count alignment)
    {
        return (size + alignment - 1) & ~(alignment - 1);
    }
    // Use to get the max of two alignments. Uses some maths such that `invalid` is always max
    SLANG_FORCE_INLINE static Count maxAlignment(Count a, Count b)
    {
        return (UCount(a) - 1) > (UCount(b) - 1) ? a : b;
    }

    /// Given two sizes, returns a result that can hold the union.
    static NaturalSize calcUnion(NaturalSize a, NaturalSize b);

    /// Value chosen such that normal combining operations produce an invalid result
    /// as typically a max.
    static const Count kInvalidAlignment = 0;

    /// Get the stride, which is equivalent to the size aligned
    SLANG_FORCE_INLINE Count getStride() const { return calcAligned(size, alignment); }

    /// Append rhs to this.
    /// If rhs is invalid or this is the result will also be invalid
    void append(const ThisType& rhs)
    {
        const auto newAlignment = maxAlignment(alignment, rhs.alignment);

        // If the new alignment is valid we calculate the size, else it's 0
        size =
            (newAlignment != kInvalidAlignment) ? (calcAligned(size, rhs.alignment) + rhs.size) : 0;

        // Set the new alignment
        alignment = newAlignment;
    }

    SLANG_FORCE_INLINE bool isInvalid() const { return alignment == kInvalidAlignment; }
    SLANG_FORCE_INLINE bool isValid() const { return !isInvalid(); }

    bool operator==(const ThisType& rhs) const
    {
        return size == rhs.size && alignment == rhs.alignment;
    }
    bool operator!=(const ThisType& rhs) const { return !(*this == rhs); }

    /// Converts to bool to make testing convenient
    operator bool() const { return isValid(); }

    /// An empty size. It consumes 0 bytes and has the lowest alignment (1)
    static ThisType makeEmpty() { return ThisType{0, 1}; }
    /// Make an invalid size.
    static ThisType makeInvalid() { return ThisType{0, kInvalidAlignment}; }
    /// Make a size with an amount of bytes and the alignment
    static ThisType make(Count size, Count alignment) { return ThisType{size, alignment}; }

    /// Given a base type returns it's size
    static ThisType makeFromBaseType(BaseType baseType);

    /// Multiply by a count.
    /// Will return invalid if count < 0 or this is already invalid
    ThisType operator*(Count count) const;

    Count size;
    Count alignment;
};

struct ASTNaturalLayoutContext
{
    /// Given a type returns it's natural size.
    /// Returns invalid size if types size could not be calculated
    NaturalSize calcSize(Type* type);

    /// Ctor
    ASTNaturalLayoutContext(ASTBuilder* astBuilder, DiagnosticSink* sink = nullptr);

protected:
    /// Gets a count (positivie integer including 0).
    /// <0 indicates error
    Count _getCount(IntVal* intVal);

    /// The main implementation, assumes outer `calcSize` will perform caching
    NaturalSize _calcSizeImpl(Type* type);

    Dictionary<Type*, NaturalSize> m_typeToSize;

    ASTBuilder* m_astBuilder;
    DiagnosticSink* m_sink;
};

} // namespace Slang

#endif
