// slang-ast-print.h

#ifndef SLANG_AST_PRINT_H
#define SLANG_AST_PRINT_H

#include "../core/slang-range.h"
#include "slang-ast-all.h"

namespace Slang
{

class ASTPrinter
{
public:
    typedef uint32_t OptionFlags;
    struct OptionFlag
    {
        enum Enum : OptionFlags
        {
            ParamNames = 0x01, ///< If set will output parameter names
            ModuleName = 0x02, ///< Writes out module names
            NoInternalKeywords =
                0x04, ///< Omits internal decoration keywords (e.g. __target_intrinsic).
            SimplifiedBuiltinType = 0x08, ///< Prints simplified builtin generic types (e.g. float3)
                                          ///< instead of its generic form.

            /// Use the original generic type name instead of the specialized
            /// type name defined on an extension when
            /// printing the target type of an extension decl.
            NoSpecializedExtensionTypeName = 0x10,
            // Output `= blah` when function type parameters have a default value
            DefaultParamValues = 0x20,
        };
    };

    /// Note that we could/can have a hierarchy of Parts - with overlapping spans.
    /// Moreover we could have less kinds, if we used the overlaps to signal out sections
    ///
    /// For example we could have a 'Param', 'Generic' span, and then have 'Name', 'Type' and
    /// 'Value'. So a param type, would be the 'Type' defined in a Param span. Moreover you could
    /// have the hierarchy of Types, and then such that you can pull out specific parts that make up
    /// a type.
    ///
    /// This is powerful/flexible - but requires more complexity at the use sites, so for now we use
    /// this simpler mechanism.

    /// Defines part of the structure of the output printed.
    struct Part
    {
        enum class Kind
        {
            None,
            Type,
            Value,
            Name,
        };

        enum class Type
        {
            None,
            ParamType,             ///< The type associated with a parameter
            ParamName,             ///< The name associated with a parameter
            ReturnType,            ///< The return type
            DeclPath,              ///< The declaration path (NOT including the actual decl name)
            GenericParamType,      ///< Generic parameter type
            GenericParamValue,     ///< Generic parameter value
            GenericParamValueType, ///< The type requirement for a value type
        };

        static Kind getKind(Type type);
        static Part make(Type type, Index start, Index end)
        {
            Part part;
            part.type = type;
            part.start = start;
            part.end = end;
            return part;
        }

        Type type = Type::None;
        Index start;
        Index end;
    };

    struct PartPair
    {
        Part first;
        Part second;
    };

    struct ScopePart
    {
        ScopePart(ASTPrinter* printer, Part::Type type)
            : m_printer(printer), m_type(type), m_startIndex(printer->m_builder.getLength())
        {
        }
        ~ScopePart()
        {
            List<Part>* parts = m_printer->m_parts;
            if (parts)
            {
                parts->add(Part::make(m_type, m_startIndex, m_printer->m_builder.getLength()));
            }
        }

        Part::Type m_type;
        Index m_startIndex;
        ASTPrinter* m_printer;
    };

    /// We might want options to change how things are output, for example we may want to output
    /// parameter names if there are any

    /// Get the currently built up string
    StringBuilder& getStringBuilder() { return m_builder; }
    /// Get the current offset, for the end of the string builder - useful for building up ranges
    Index getOffset() const { return m_builder.getLength(); }

    /// Reset the state
    void reset() { m_builder.clear(); }

    /// Get the current string
    String getString() { return m_builder.produceString(); }

    /// Get contents as a slice
    UnownedStringSlice getSlice() const { return m_builder.getUnownedSlice(); }

    /// Add a type
    void addType(Type* type);
    /// Add an expr
    void addExpr(Expr* type);
    /// Add a value
    void addVal(Val* val);

    /// Add the path to the declaration including the declaration name
    void addDeclPath(const DeclRef<Decl>& declRef);

    /// Add the path such that it encapsulates all overridable decls (ie is without terminal generic
    /// parameters)
    void addOverridableDeclPath(const DeclRef<Decl>& declRef);

    /// Add just the parameters from a declaration.
    /// Will output the generic parameters (if it's a generic) in <> before the parameters ()
    void addDeclParams(const DeclRef<Decl>& declRef, List<Range<Index>>* outParamRange = nullptr);

    /// Add a prefix that describes the kind of declaration
    void addDeclKindPrefix(Decl* decl);

    /// Add the result type
    /// Should be called after the decl params
    void addDeclResultType(const DeclRef<Decl>& inDeclRef);

    /// Add the signature for the decl
    void addDeclSignature(const DeclRef<Decl>& declRef);

    /// Add generic parameters
    void addGenericParams(const DeclRef<GenericDecl>& genericDeclRef);

    /// Get the specified part type. Returns empty slice if not found
    UnownedStringSlice getPartSlice(Part::Type partType) const;
    /// Get the slice for a part
    UnownedStringSlice getPartSlice(const Part& part) const { return getPart(getSlice(), part); }

    /// Gets the specified part type
    static UnownedStringSlice getPart(const UnownedStringSlice& slice, const Part& part)
    {
        return (part.type != Part::Type::None)
                   ? UnownedStringSlice(slice.begin() + part.start, slice.begin() + part.end)
                   : UnownedStringSlice();
    }
    static UnownedStringSlice getPart(
        Part::Type partType,
        const UnownedStringSlice& slice,
        const List<Part>& parts);

    static void appendDeclName(Decl* decl, StringBuilder& out);

    /// Ctor
    ASTPrinter(ASTBuilder* astBuilder, OptionFlags optionFlags = 0, List<Part>* parts = nullptr)
        : m_astBuilder(astBuilder), m_parts(parts), m_optionFlags(optionFlags)
    {
    }

    static String getDeclSignatureString(const LookupResultItem& item, ASTBuilder* astBuilder);
    static String getDeclSignatureString(DeclRef<Decl> declRef, ASTBuilder* astBuilder);

protected:
    void _addDeclPathRec(const DeclRef<Decl>& declRef, Index depth);
    void _addDeclName(Decl* decl);

    OptionFlags m_optionFlags; ///< Flags controlling output
    List<Part>* m_parts;       ///< Optional parts list
    ASTBuilder* m_astBuilder;  ///< Required as types are setup as part of printing
    StringBuilder m_builder;   ///< The output of the 'printing' process
};

} // namespace Slang

#endif // SLANG_AST_PRINT_H
