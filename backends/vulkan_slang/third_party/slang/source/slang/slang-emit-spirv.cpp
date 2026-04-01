// slang-emit-spirv.cpp

#include "../core/slang-memory-arena.h"
#include "slang-compiler.h"
#include "slang-emit-base.h"
#include "slang-ir-call-graph.h"
#include "slang-ir-entry-point-decorations.h"
#include "slang-ir-insts.h"
#include "slang-ir-layout.h"
#include "slang-ir-redundancy-removal.h"
#include "slang-ir-spirv-legalize.h"
#include "slang-ir-spirv-snippet.h"
#include "slang-ir-util.h"
#include "slang-ir.h"
#include "slang-lookup-spirv.h"
#include "spirv/unified1/spirv.h"

#include <type_traits>

namespace Slang
{

// Our goal in this file is to convert a module in the Slang IR over to an
// equivalent module in the SPIR-V intermediate language.
//
// The Slang IR is (intentionally) similar to SPIR-V in many ways, and both
// can represent shaders at similar levels of abstraction, so much of the
// translation involves one-to-one translation of Slang IR instructions
// to their SPIR-V equivalents.
//
// SPIR-V differs from Slang IR in some key ways, and the SPIR-V
// specification places many restrictions on how the IR can be encoded.
// In some cases we will rely on earlier IR passes to convert Slang IR
// into a form closer to what SPIR-V expects (e.g., by moving all
// varying entry point parameters to global scope), but other differences
// will be handled during the translation process.
//
// The logic in this file relies on the formal [SPIR-V Specification].
// When we are making use of or enforcing some property from the spec,
// we will try to refer to the relevant section in comments.
//
// [SPIR-V Specification]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html

// [2.3: Physical Layout of a SPIR-V Module and Instruction]
//
// > A SPIR-V module is a single linear stream of words.


// [2.3: Physical Layout of a SPIR-V Module and Instruction]
//
// > All remaining words are a linear sequence of instructions.
// > Each instruction is a stream of words
//
// After a fixed-size header, the contents of a SPIR-V module
// is just a flat sequence of instructions, each of which is
// just a sequence of words.
//
// In principle we could try to emit instructions directly
// in one pass as a stream of words, but there are additional
// constraints placed by the SPIR-V encoding that would make
// a single-pass strategy very hard, so we don't attempt it.
//
// [2.4 Logical Layout of a Module]
//
// SPIR-V imposes some global ordering constraints on instructions,
// such that certain instructions must come before or after others.
// For example, all `OpCapability` instructions must come before any
// `OpEntryPoint` instructions.
//
// While the SPIR-V spec doesn't use such a term, we will take
// the enumeration of the ordering in Section 2.4 and use it to
// define a list of *logical sections* that make up a SPIR-V module.

/// Logical sections of a SPIR-V module.
enum class SpvLogicalSectionID
{
    Capabilities,
    Extensions,
    ExtIntInstImports,
    MemoryModel,
    EntryPoints,
    ExecutionModes,
    DebugStringsAndSource,
    DebugNames,
    Annotations,
    ConstantsAndTypes,
    GlobalVariables,
    FunctionDeclarations,
    FunctionDefinitions,

    Count,
};

// The registered id for the Slang compiler.
static const uint32_t kSPIRVSlangCompilerId = 40 << 16;

// While the SPIR-V module is nominally (according to the spec) just
// a flat sequence of instructions, in practice some of the instructions
// are logically in a parent/child relationship.
//
// In particular, functions "own" the instructions between an `OpFunction`
// and the matching `OpFunctionEnd`. We can also think of basic
// blocks within a function as owning the instructions between
// an `OpLabel` (which represents the bloc) and the next label
// or the end of the function.
//
// Furthermore, the common case is SPIR-V is that an instruction
// that defines some value must appear before any instruction
// that uses that value as an operand. This property is often true
// in a Slang IR module, but isn't strictly enforced for things at
// the global scope.
//
// To deal with the above issues, our strategy will be to emit
// SPIR-V instructions into a lightweight intermediate structure
// that simplifies dealing with ordering constraiints on
// instructions.
//
// We will start by forward-declaring the type we will
// use to represent instructions:
//
struct SpvInst;

// Next, we will define a base type that can serve as a parent
// to SPIR-V instructions. Both the logical sections defined
// earlier and instructions such as functions will be used
// as parents.

/// Base type for SPIR-V instructions and logical sections of a module
///
/// Holds and supports appending to a list of child instructions.
struct SpvInstParent
{
public:
    /// Add an instruction to the end of the list of children
    void addInst(SpvInst* inst);

    /// Dump all children, recursively, to a flattened list of SPIR-V words
    void dumpTo(List<SpvWord>& ioWords);

    /// The first child, if any.
    SpvInst* m_firstChild = nullptr;

    /// A pointer to the null pointer at the end of the linked list.
    ///
    /// If the list of children is empty this points to `m_firstChild`,
    /// while if it is non-empty it points to the `nextSibling` field
    /// of the last instruction.
    ///
    SpvInst* m_lastChild = nullptr;
};

// A SPIR-V instruction is then (in the general case) a potential
// parent to other instructions.

/// A type to represent a SPIR-V instruction to be emitted.
///
/// This type alows the instruction to be built up across
/// multiple steps in a mutable fashion.
///
struct SpvInst : SpvInstParent
{
    // [2.3: Physical Layout of a SPIR-V Module and Instruction]
    //
    // > Each instruction is a stream of words
    //
    // > Opcode: The 16 high-order bits are the WordCount of the instruction.
    // >         The 16 low-order bits are the opcode enumerant.
    //
    // We will store the "opcode enumerant" directly in our
    // intermediate structure, and compute the word count on
    // the fly when writing an instruction to an output buffer.

    /// The SPIR-V opcode for the instruction
    SpvOp opcode;

    // [2.3: Physical Layout of a SPIR-V Module and Instruction]
    //
    // > Optional instruction type <id> (presence determined by opcode)
    // > Optional instruction Result <id> (presence determined by opcode)
    // > Operand 1 (if needed)
    // > Operand 2 (if needed)
    // > ...
    //
    // We represent the remaining words of the instruction (after
    // the opcode word) as an undifferentiated array. Any code
    // that encodes an instruction is responsible for knowing the
    // opcode-specific data that is required.
    //
    // Our code does not need to process instruction operands after
    // they have been written into a `SpvInst`. If we ever had
    // cases where we needed to do post-processing, then we would
    // need to store a more refined representation here.

    /// The additional words of the instruction after the opcode
    SpvWord* operandWords = nullptr;
    /// The amount of operand words
    uint32_t operandWordsCount = 0;

    // We will store the instructions in a given `SpvInstParent`
    // using an intrusive linked list.

    SpvInstParent* parent = nullptr;

    /// The next instruction in the same `SpvInstParent`
    SpvInst* nextSibling = nullptr;

    SpvInst* prevSibling = nullptr;

    /// The result <id> produced by this instruction, or zero if it has no result.
    SpvWord id = 0;

    /// Dump the instruction (and any children, recursively) into the flat array of SPIR-V words.
    void dumpTo(List<SpvWord>& ioWords)
    {
        // [2.2: Terms]
        //
        // > Word Count: The complete number of words taken by an instruction,
        // > including the word holding the word count and opcode, and any optional
        // > operands. An instruction’s word count is the total space taken by the instruction.
        //
        SpvWord wordCount = 1 + SpvWord(operandWordsCount);

        // [2.3: Physical Layout of a SPIR-V Module and Instruction]
        //
        // > Opcode: The 16 high-order bits are the WordCount of the instruction.
        // >         The 16 low-order bits are the opcode enumerant.
        //
        ioWords.add(wordCount << 16 | opcode);

        // The operand words simply follow the opcode word.
        //
        ioWords.addRange(operandWords, operandWordsCount);

        // In our representation choice, the children of a
        // parent instruction will always follow the encoded
        // words of a parent:
        //
        // * The instructions inside a function always follow the `OpFunction`
        // * The instructions inside a block always follow the `OpLabel`
        //
        SpvInstParent::dumpTo(ioWords);
    }

    void removeFromParent()
    {
        auto oldParent = parent;

        // If we don't currently have a parent, then
        // we are doing fine.
        if (!oldParent)
            return;

        auto pp = prevSibling;
        auto nn = nextSibling;

        if (pp)
        {
            SLANG_ASSERT(pp->parent == oldParent);
            pp->nextSibling = nn;
        }
        else
        {
            oldParent->m_firstChild = nn;
        }

        if (nn)
        {
            SLANG_ASSERT(nn->parent == oldParent);
            nn->prevSibling = pp;
        }
        else
        {
            oldParent->m_lastChild = pp;
        }

        prevSibling = nullptr;
        nextSibling = nullptr;
        parent = nullptr;
    }
};

/// A logical section of a SPIR-V module
struct SpvLogicalSection : SpvInstParent
{
};

// Now that we've filled in the definition of `SpvInst`, we can
// go back and define the key operations on `SpvInstParent`.

void SpvInstParent::addInst(SpvInst* inst)
{
    SLANG_ASSERT(inst);
    SLANG_ASSERT(!inst->nextSibling);

    if (m_firstChild == nullptr)
    {
        m_firstChild = m_lastChild = inst;
        return;
    }

    // The user shouldn't be trying to add multiple instructions at once.
    // If they really want that then they probably wanted to give `inst`
    // some children.
    //
    m_lastChild->nextSibling = inst;
    inst->prevSibling = m_lastChild;
    inst->parent = this;
    m_lastChild = inst;
}

void SpvInstParent::dumpTo(List<SpvWord>& ioWords)
{
    for (auto child = m_firstChild; child; child = child->nextSibling)
    {
        child->dumpTo(ioWords);
    }
}

/// The context for inlining a SPV assembly snippet.
struct SpvSnippetEmitContext
{
    SpvInst* resultType;
    IRType* irResultType;
    // True if resultType is float or vector of float.
    bool isResultTypeFloat;
    // True if resultType is signed.
    bool isResultTypeSigned;
    List<SpvWord> argumentIds;
};

// A structure which can hold an integer literal, either one word or several
struct SpvLiteralInteger
{
    static SpvLiteralInteger from32(int32_t value) { return from32(uint32_t(value)); }
    static SpvLiteralInteger from32(uint32_t value) { return SpvLiteralInteger{{value}}; }
    static SpvLiteralInteger from64(int64_t value) { return from64(uint64_t(value)); }
    static SpvLiteralInteger from64(uint64_t value)
    {
        return SpvLiteralInteger{{SpvWord(value), SpvWord(value >> 32)}};
    }
    List<SpvWord> value; // Words, stored low words to high (TODO, SmallArray or something here)
};

// A structure which can hold bitwise literal, either one word or several
struct SpvLiteralBits
{
    static SpvLiteralBits from32(uint32_t value) { return SpvLiteralBits{{value}}; }
    static SpvLiteralBits from64(uint64_t value)
    {
        return SpvLiteralBits{{SpvWord(value), SpvWord(value >> 32)}};
    }
    static SpvLiteralBits fromUnownedStringSlice(UnownedStringSlice text)
    {
        SpvLiteralBits result;

        // [Section 2.2.1 : Instructions]
        //
        // > Literal String: A nul-terminated stream of characters consuming
        // > an integral number of words. The character set is Unicode in the
        // > UTF-8 encoding scheme. The UTF-8 octets (8-bit bytes) are packed
        // > four per word, following the little-endian convention (i.e., the
        // > first octet is in the lowest-order 8 bits of the word).
        // > The final word contains the string’s nul-termination character (0), and
        // > all contents past the end of the string in the final word are padded with 0.

        // First work out the amount of words we'll need
        const Index textCount = text.getLength();
        // Calculate the minimum amount of bytes needed - which needs to include terminating 0
        const Index minByteCount = textCount + 1;
        // Calculate the amount of words including padding if necessary
        const Index wordCount = (minByteCount + 3) >> 2;

        // Make space on the operand stack, keeping the free space start in operandStartIndex
        result.value.setCount(wordCount);

        // Set dst to the start of the operand memory
        char* dst = (char*)(result.value.getBuffer());

        // Copy the text
        SLANG_ASSUME(textCount >= 0);
        memcpy(dst, text.begin(), textCount);

        // Set terminating 0, and remaining buffer 0s
        memset(dst + textCount, 0, wordCount * sizeof(SpvWord) - textCount);

        return result;
    }
    List<SpvWord> value; // Words, stored low words to high (TODO, SmallArray or something here)
};

// As a convenience, there are often cases where
// we will want to emit all of the operands of some
// IR instruction as <id> operands of a SPIR-V
// instruction. This is handy in cases where the
// Slang IR and SPIR-V instructions agree on the
// number, order, and meaning of their operands.
/// Helper type for emitting all the operands of the current IR instruction
struct OperandsOf
{
    OperandsOf(IRInst* irInst)
        : irInst(irInst)
    {
    }

    IRInst* irInst = nullptr;
};

/// Helper type for not emitting an operand in this position
struct SkipThisOptionalOperand
{
};

template<typename T>
struct OptionalOperand
{
    static_assert(std::is_trivial_v<T>);
    OptionalOperand(SkipThisOptionalOperand)
        : present(false)
    {
    }
    OptionalOperand(T value)
        : present(true), value(value)
    {
    }
    bool present;
    T value;
};

template<typename T>
OptionalOperand<T> nullOptionOperand()
{
    return OptionalOperand<T>{false};
}

template<typename T>
OptionalOperand<T> someOptionOperand(T t)
{
    return OptionalOperand<T>{true, t};
}

template<typename T>
constexpr bool isPlural = false;
template<typename T>
constexpr bool isPlural<List<T>> = true;
template<typename T>
constexpr bool isPlural<IROperandList<T>> = true;
template<typename T, Index N>
constexpr bool isPlural<Array<T, N>> = true;
template<>
constexpr bool isPlural<OperandsOf> = true;
template<>
constexpr bool isPlural<IRUse*> = true;
template<typename T>
constexpr bool isSingular = !isPlural<T>;


// Now that we've defined the intermediate data structures we will
// use to represent SPIR-V code during emission, we will move on
// to defining the main context type that will drive SPIR-V
// code generation.

/// Context used for translating a Slang IR module to SPIR-V
struct SPIRVEmitContext : public SourceEmitterBase, public SPIRVEmitSharedContext
{
    /// The Slang IR module being translated
    IRModule* m_irModule;

    // [2.2: Terms]
    //
    // > <id>: A numerical name; the name used to refer to an object, a type,
    // > a function, a label, etc. An <id> always consumes one word.
    // > The <id>s defined by a module obey SSA.
    //
    // [2.3: Physical Layout of a SPIR-V Module and Instruction]
    //
    // > Bound; where all <id>s in this module are guaranteed to satisfy
    // > 0 < id < Bound
    // > Bound should be small, smaller is better, with all <id> in a module being densely packed
    // and near 0.
    //
    // Instructions will be referred to by their <id>s.
    // We need to generate <id>s for instructions, and also
    // compute the "bound" value that will be stored in
    // the module header.
    //
    // We will use a single counter and allocate <id>s
    // on demand. There may be some slop where we allocate
    // an <id> for something that never gets referenced,
    // but we expect the amount of slop to be small (and
    // it can be cleaned up by other tools/passes).

    /// The next destination `<id>` to allocate.
    SpvWord m_nextID = 1;

    OrderedDictionary<SpvInst*, IRPtrTypeBase*> m_forwardDeclaredPointers;

    SpvInst* m_nullDwarfExpr = nullptr;

    // A hash set to prevent redecorating the same spv inst.
    HashSet<SpvId> m_decoratedSpvInsts;

    SpvAddressingModel m_addressingMode = SpvAddressingModelLogical;

    // We will store the logical sections of the SPIR-V module
    // in a single array so that we can easily look up a
    // section by its `SpvLogicalSectionID`.

    /// The logical sections of the SPIR-V module
    SpvLogicalSection m_sections[int(SpvLogicalSectionID::Count)];

    /// Get a logical section based on its `SpvLogicalSectionID`
    SpvLogicalSection* getSection(SpvLogicalSectionID id) { return &m_sections[int(id)]; }

    // At the end of emission we need a single linear stream of words,
    // so we will eventually flatten `m_sections` into a single array.

    /// The final array of SPIR-V words that defines the encoded module
    List<SpvWord> m_words;

    /// Emit the concrete words that make up the binary SPIR-V module.
    ///
    /// This function fills in `m_words` based on the data in `m_sections`.
    /// This function should only be called once.
    ///
    void emitPhysicalLayout()
    {
        // [2.3: Physical Layout of a SPIR-V Module and Instruction]
        //
        // > Magic Number
        //
        m_words.add(SpvMagicNumber);

        // > Version nuumber
        //
        m_words.add(m_spvVersion);

        // > Generator's magic number.
        //
        m_words.add(kSPIRVSlangCompilerId);

        // > Bound
        //
        // As described above, we use `m_nextID` to allocate
        // <id>s, so its value when we are done emitting code
        // can serve as the bound.
        //
        m_words.add(m_nextID);

        // > 0 (Reserved for instruction schema, if needed.)
        //
        m_words.add(0);

        // > First word of instruction stream
        // > All remaining words are a linear sequence of instructions.
        //
        // Once we are done emitting the header, we emit all
        // the instructions in our logical sections.
        //
        for (int ii = 0; ii < int(SpvLogicalSectionID::Count); ++ii)
        {
            m_sections[ii].dumpTo(m_words);
        }
    }

    // We will often need to refer to an instrcition by its
    // <id>, given only the Slang IR instruction that represents
    // it (e.g., when it is used as an operand of another
    // instruction).
    //
    // To that end we will keep a map of instructions that
    // have been emitted, where a Slang IR instruction maps
    // to the corresponding SPIR-V instruction.

    /// Map a Slang IR instruction to the corresponding SPIR-V instruction
    Dictionary<IRInst*, SpvInst*> m_mapIRInstToSpvInst;

    // Sometimes we need to reserve an ID for an `IRInst` without actually
    // emitting it. We use `m_mapIRInstToSpvID` to hold all reserved SpvIDs.
    // Use `getIRInstSpvID` to obtain an SpvID for an `IRInst` if the
    // `IRInst` may not have been emitted.
    Dictionary<IRInst*, SpvWord> m_mapIRInstToSpvID;

    // Map a Slang IR instruction to the corresponding SPIR-V debug instruction.
    Dictionary<IRInst*, SpvInst*> m_mapIRInstToSpvDebugInst;

    /// Register that `irInst` maps to `spvInst`
    void registerInst(IRInst* irInst, SpvInst* spvInst)
    {
        m_mapIRInstToSpvInst.add(irInst, spvInst);

        // If we have reserved an SpvID for `irInst`, make sure to use it.
        SpvWord reservedID = 0;
        m_mapIRInstToSpvID.tryGetValue(irInst, reservedID);

        if (reservedID)
        {
            SLANG_ASSERT(spvInst->id == 0);
            spvInst->id = reservedID;
        }
    }

    /// Register that `irInst` has debug info represented by `spvDebugInst`.
    void registerDebugInst(IRInst* irInst, SpvInst* spvDebugInst)
    {
        m_mapIRInstToSpvDebugInst.add(irInst, spvDebugInst);
    }

    SpvInst* findDebugScope(IRInst* inst)
    {
        for (auto parent = inst; parent; parent = parent->getParent())
        {
            if (!as<IRFunc>(parent) && !as<IRModuleInst>(parent))
                continue;

            SpvInst* spvInst = nullptr;
            if (m_mapIRInstToSpvDebugInst.tryGetValue(parent, spvInst))
                return spvInst;
        }
        return nullptr;
    }

    /// Get or reserve a SpvID for an IR value.
    SpvWord getIRInstSpvID(IRInst* inst)
    {
        // If we have already emitted an SpvInst for `inst`, return its ID.
        SpvInst* spvInst = nullptr;
        if (m_mapIRInstToSpvInst.tryGetValue(inst, spvInst))
            return getID(spvInst);
        // Check if we have reserved an ID for `inst`.
        SpvWord result = 0;
        if (m_mapIRInstToSpvID.tryGetValue(inst, result))
            return result;
        // Otherwise, reserve a new ID for inst, and register it in `m_mapIRInstToSpvID`.
        result = m_nextID;
        ++m_nextID;
        m_mapIRInstToSpvID[inst] = result;
        return result;
    }

    // When we are emitting an instruction that can produce
    // a result, we will allocate an <id> to it so that other
    // instructions can refer to it.
    //
    // We will allocate <id>s on emand as they are needed.

    SpvWord freshID() { return m_nextID++; }

    /// Get the <id> for `inst`, or assign one if it doesn't have one yet
    SpvWord getID(SpvInst* inst)
    {
        auto id = inst->id;
        if (!id)
        {
            id = freshID();
            inst->id = id;
        }
        return id;
    }

    // We will build up `SpvInst`s in a stateful fashion,
    // mostly for convenience. We could in theory compute
    // the number of words each instruction needs, then allocate
    // the words, then fill them in, but that would make the
    // emit logic more complicated and we'd like to keep it simple
    // until we are sure performance is an issue.
    //
    // Emitting an instruction starts with picking the opcode
    // and allocating the `SpvInst`.

    // Holds a stack of instructions operands *BEFORE* they added to the instruction.
    List<SpvWord> m_operandStack;
    // The current instruction being constructed. Cannot add operands unless it
    // is set, or we are peeking at some operands to see if we have them memoized
    SpvInst* m_currentInst = nullptr;
    bool m_peekingOperands = false;

    // Operands can only be added when inside of a InstConstructScope or...
    struct InstConstructScope
    {
        SLANG_FORCE_INLINE operator SpvInst*() const { return m_inst; }

        InstConstructScope(SPIRVEmitContext* context, SpvOp opcode, IRInst* irInst = nullptr)
            : m_context(context)
        {
            m_context->_beginInst(opcode, irInst, *this);
        }
        ~InstConstructScope() { m_context->_endInst(*this); }

        SpvInst* m_inst;             ///< The instruction associated with this scope
        SPIRVEmitContext* m_context; ///< The context
        SpvInst* m_previousInst;     ///< The previously live inst
        Index m_operandsStartIndex;  ///< The start index for operands of m_inst
    };

    // ...If we're speculatively adding them to see if we have a memoized results
    struct OperandMemoizeScope
    {
        OperandMemoizeScope(SPIRVEmitContext* context)
            : m_context(context)
        {
            m_tmpOperandStack.swapWith(m_context->m_operandStack);
            std::swap(m_tmpPeeking, m_context->m_peekingOperands);
            std::swap(m_tmpInst, m_context->m_currentInst);
        }
        ~OperandMemoizeScope()
        {
            std::swap(m_tmpInst, m_context->m_currentInst);
            std::swap(m_tmpPeeking, m_context->m_peekingOperands);
            m_tmpOperandStack.swapWith(m_context->m_operandStack);
        }

        SPIRVEmitContext* m_context;
        List<SpvWord> m_tmpOperandStack;
        bool m_tmpPeeking = true;
        SpvInst* m_tmpInst = nullptr;
    };

    /// Holds memory for instructions and operands.
    MemoryArena m_memoryArena;

    /// Begin emitting an instruction with the given SPIR-V `opcode`.
    ///
    /// If `irInst` is non-null, then the resulting SPIR-V instruction
    /// will be registered as corresponding to `irInst`.
    ///
    /// The created instruction is stored in m_currentInst.
    ///
    /// Should not typically be called directly use InstConstructScope to scope construction
    void _beginInst(SpvOp opcode, IRInst* irInst, InstConstructScope& ioScope)
    {
        SLANG_ASSERT(this == ioScope.m_context);

        // Allocate the instruction
        auto spvInst = new (m_memoryArena.allocate(sizeof(SpvInst))) SpvInst();
        spvInst->opcode = opcode;

        if (irInst)
        {
            registerInst(irInst, spvInst);
        }

        // Set up the scope
        ioScope.m_inst = spvInst;
        ioScope.m_previousInst = m_currentInst;
        ioScope.m_operandsStartIndex = m_operandStack.getCount();

        // Set the current instruction
        m_currentInst = spvInst;
    }

    /// End emitting an instruction
    /// Should not typically be called directly use InstConstructScope to scope construction
    void _endInst(const InstConstructScope& scope)
    {
        SLANG_ASSERT(scope.m_inst == m_currentInst);

        const Index operandsStartIndex = scope.m_operandsStartIndex;
        // Work out how many operands were added
        const Index operandsCount = m_operandStack.getCount() - operandsStartIndex;


        if (operandsCount)
        {
            // Allocate the operands
            m_currentInst->operandWords = m_memoryArena.allocateAndCopyArray(
                m_operandStack.getBuffer() + operandsStartIndex,
                operandsCount);
            // Set the count
            m_currentInst->operandWordsCount = uint32_t(operandsCount);
        }

        // Make the previous inst active
        m_currentInst = scope.m_previousInst;
        // Reset the operand stack
        m_operandStack.setCount(operandsStartIndex);
    }

    /// Ensure that an instruction has been emitted
    SpvInst* ensureInst(IRInst* irInst)
    {
        SpvInst* spvInst = nullptr;
        if (!m_mapIRInstToSpvInst.tryGetValue(irInst, spvInst))
        {
            // If the `irInst` hasn't already been emitted,
            // then we will assume that is is a global instruction
            // (a constant, type, function, etc.) and we should make
            // sure it gets emitted now.
            //
            // Note: this step means that emitting an instruction
            // can be re-entrant/recursive. Because we emit the SPIR-V
            // words for an instruction into an intermediate structure
            // we don't have to worry about the re-entrancy causing
            // the ordering of instruction words to be interleaved.
            //
            spvInst = emitGlobalInst(irInst);
        }
        return spvInst;
    }

    // Whilst an instruction has been created, we append the operand
    // words to it with `emitOperand`. There are a few different
    // case of operands that we handle.
    //
    // The simplest case is when an instruction takes an operand
    // that is just a literal SPIR-V word.

    /// Emit a literal `word` as an operand to the current instruction
    void emitOperand(SpvWord word)
    {
        // Can only add operands if we are constructing an instruction (ie in _beginInst/_endInst)
        SLANG_ASSERT(m_currentInst || m_peekingOperands);
        m_operandStack.add(word);
    }

    // The most common case of operand is an <id> that represents
    // some other instruction. In cases where we already have
    // an <id> we can emit it as a literal and the meaning is
    // the same. If we have a `SpvInst` we can look up or
    // generate an <id> for it.

    /// Emit an operand to the current instruction, which references `src` by its <id>
    void emitOperand(SpvInst* src) { emitOperand(getID(src)); }

    // Commonly, we will have an operand in the form of an `IRInst`
    // which might either represent an instruction we've already
    // emitted (e.g., because it came earlier in a function body)
    // or which we have yet to emit (because it is a global-scope
    // instruction that has not been referenced before).

    /// Emit an operand to the current instruction, which references `src` by its <id>
    void emitOperand(IRInst* src)
    {
        SpvInst* spvSrc = ensureInst(src);
        emitOperand(getID(spvSrc));
    }

    // Some instructions take a string as a literal operand,
    // which requires us to follow the SPIR-V rules to
    // encode the string into multiple operand words.

    /// Emit an operand that is encoded as a literal string
    void emitOperand(UnownedStringSlice const& text)
    {
        // Can only emitOperands if we are in an instruction
        SLANG_ASSERT(m_currentInst || m_peekingOperands);
        SLANG_COMPILE_TIME_ASSERT(sizeof(SpvWord) == 4);

        // Assert that `text` doesn't contain any embedded nul bytes, since they
        // could lead to invalid encoded results.
        SLANG_ASSERT(text.indexOf(0) < 0);
        emitOperand(SpvLiteralBits::fromUnownedStringSlice(text));
    }

    // Sometimes we will want to pass down an argument that
    // represents a result <id> operand, but we won't yet
    // have access to the `SpvInst` that will get the <id>.
    // We will use a dummy `enum` type to support this case.

    enum ResultIDToken
    {
        kResultID
    };

    void emitOperand(ResultIDToken)
    {
        // This is the one case we shouldn't be peeking at operands, as it
        // depends on having an instruction under construction
        SLANG_ASSERT(m_currentInst);

        // A result <id> operand uses the <id> of the instruction itself (which is m_currentInst)
        emitOperand(getID(m_currentInst));
    }

    void emitOperand(const SpvLiteralBits& bits)
    {
        for (const auto v : bits.value)
            emitOperand(v);
    }

    void emitOperand(const SpvLiteralInteger& integer)
    {
        for (const auto v : integer.value)
            emitOperand(v);
    }

    template<typename T>
    void emitOperand(const List<T>& os)
    {
        for (const auto& o : os)
            emitOperand(o);
    }

    template<typename T>
    void emitOperand(const IROperandList<T>& os)
    {
        for (const auto& o : os)
            emitOperand(o);
    }

    template<typename T, Index N>
    void emitOperand(const Array<T, N>& os)
    {
        for (const auto& o : os)
            emitOperand(o);
    }

    template<typename T>
    void emitOperand(const ArrayView<T>& os)
    {
        for (const auto& o : os)
            emitOperand(o);
    }

    template<typename TConstant>
    struct ConstantValueKey
    {
        IRType* type;
        TConstant value;
        HashCode getHashCode() const
        {
            return combineHash(Slang::getHashCode(type), Slang::getHashCode(value));
        }
        bool operator==(const ConstantValueKey& other) const
        {
            return type == other.type && value == other.value;
        }
    };
    Dictionary<ConstantValueKey<IRIntegerValue>, SpvInst*> m_spvIntConstants;
    Dictionary<ConstantValueKey<IRFloatingPointValue>, SpvInst*> m_spvFloatConstants;

    // Get an SpvLiteralBits from an IRConstant.
    SpvLiteralBits getLiteralBits(IRInst* type, IRInst* inst)
    {
        switch (type->getOp())
        {
        case kIROp_DoubleType:
            {
                if (auto fval = as<IRFloatLit>(inst))
                    return SpvLiteralBits::from64(DoubleAsInt64(fval->getValue()));
                break;
            }
        case kIROp_HalfType:
            {
                if (auto fval = as<IRFloatLit>(inst))
                    return SpvLiteralBits::from32(uint32_t(FloatToHalf((float)fval->getValue())));
                break;
            }
        case kIROp_FloatType:
            {
                if (auto fval = as<IRFloatLit>(inst))
                    return SpvLiteralBits::from32(FloatAsInt((float)fval->getValue()));
                break;
            }
        case kIROp_Int64Type:
        case kIROp_UInt64Type:
#if SLANG_PTR_IS_64
        case kIROp_PtrType:
        case kIROp_UIntPtrType:
#endif
            {
                if (auto val = as<IRIntLit>(inst))
                    return SpvLiteralBits::from64(uint64_t(val->getValue()));
                break;
            }
        default:
            {
                if (auto val = as<IRIntLit>(inst))
                    return SpvLiteralBits::from32(uint32_t(val->getValue()));
                break;
            }
        }
        return SpvLiteralBits::from32(0);
    }

    SpvInst* emitIntConstant(IRIntegerValue val, IRType* type, IRInst* inst = nullptr)
    {
        ConstantValueKey<IRIntegerValue> key;
        key.value = val;
        key.type = type;
        SpvInst* result = nullptr;
        if (m_spvIntConstants.tryGetValue(key, result))
        {
            m_mapIRInstToSpvInst[inst] = result;
            return result;
        }
        switch (type->getOp())
        {
        case kIROp_Int64Type:
        case kIROp_UInt64Type:
#if SLANG_PTR_IS_64
        case kIROp_PtrType:
        case kIROp_UIntPtrType:
#endif
            {
                result = emitOpConstant(inst, type, SpvLiteralBits::from64(uint64_t(val)));
                break;
            }
        default:
            {
                result = emitOpConstant(inst, type, SpvLiteralBits::from32(uint32_t(val)));
                break;
            }
        }
        m_spvIntConstants[key] = result;
        m_mapIRInstToSpvInst[inst] = result;
        return result;
    }
    SpvInst* emitFloatConstant(IRFloatingPointValue val, IRType* type, IRInst* inst = nullptr)
    {
        ConstantValueKey<IRFloatingPointValue> key;
        key.value = val;
        key.type = type;
        SpvInst* result = nullptr;
        if (m_spvFloatConstants.tryGetValue(key, result))
        {
            m_mapIRInstToSpvInst[inst] = result;
            return result;
        }
        if (type->getOp() == kIROp_DoubleType)
        {
            result =
                emitOpConstant(inst, type, SpvLiteralBits::from64(uint64_t(DoubleAsInt64(val))));
        }
        else if (type->getOp() == kIROp_FloatType)
        {
            result = emitOpConstant(
                inst,
                type,
                SpvLiteralBits::from32(uint32_t(FloatAsInt(float(val)))));
        }
        else if (type->getOp() == kIROp_HalfType)
        {
            result = emitOpConstant(
                inst,
                type,
                SpvLiteralBits::from32(uint32_t(FloatToHalf(float(val)))));
        }
        else
        {
            SLANG_UNEXPECTED("missing case in SPIR-V emitFloatConstant");
        }
        m_mapIRInstToSpvInst[inst] = result;
        m_spvFloatConstants[key] = result;
        return result;
    }

    /// Emit operand words for all the operands of a given IR instruction
    void emitOperand(OperandsOf const& other)
    {
        auto irInst = other.irInst;
        auto operandCount = irInst->getOperandCount();
        for (UInt ii = 0; ii < operandCount; ++ii)
        {
            emitOperand(irInst->getOperand(ii));
        }
    }

    /// Do nothing
    void emitOperand(SkipThisOptionalOperand) {}

    template<typename T>
    void emitOperand(OptionalOperand<T> o)
    {
        if (o.present)
            emitOperand(o.value);
    }

    // With the above routines, code can easily construct a SPIR-V
    // instruction with arbitrary operands over multiple lines of code.
    //
    // The safe way to call these routines is encoded in the below `emitInst`
    // function.
    //
    // This allows one to generically output a SPIR-V instruction with any
    // desired operands.
    //
    // This function performs no checks that it is actually being used
    // correctly with respect to the SPIR-V rules for each opcode. As such, a
    // more type safe function for each opcode is included in
    // 'slang-emit-spirv-ops.h', and available in this class. You are
    // encouraged to use these instead.
    //
    template<typename... Operands>
    SpvInst* emitInst(SpvInstParent* parent, IRInst* irInst, SpvOp opcode, const Operands&... ops)
    {
        return emitInstCustomOperandFunc(
            parent,
            irInst,
            opcode,
            [&]() { (emitOperand(ops), ...); });
    }

    template<typename OperandEmitFunc>
    SpvInst* emitInstCustomOperandFunc(
        SpvInstParent* parent,
        IRInst* irInst,
        SpvOp opcode,
        const OperandEmitFunc& f)
    {
        InstConstructScope scopeInst(this, opcode, irInst);
        SpvInst* spvInst = scopeInst;
        f();
        parent->addInst(spvInst);
        return spvInst;
    }

    // Emits a SPV Inst with deduplication
    // This is used where our IR doesn't guarantee uniqueness but SPIR-V
    // requires it
    template<typename... Operands>
    SpvInst* emitInstMemoized(
        SpvInstParent* parent,
        IRInst* irInst,
        SpvOp opcode,
        // We take the resultId here explicitly here to make sure we don't try
        // and memoize its value.
        ResultIDToken resultId,
        const Operands&... ops)
    {
        return emitInstMemoizedCustomOperandFunc(
            parent,
            irInst,
            opcode,
            resultId,
            [&]() { (emitOperand(ops), ...); });
    }

    template<typename OperandEmitFunc>
    SpvInst* emitInstMemoizedCustomOperandFunc(
        SpvInstParent* parent,
        IRInst* irInst,
        SpvOp opcode,
        // We take the resultId here explicitly here to make sure we don't try
        // and memoize its value.
        ResultIDToken resultId,
        const OperandEmitFunc& f)
    {
        List<SpvWord> ourOperands;
        {
            auto scopePeek = OperandMemoizeScope(this);
            f();
            // Steal our operands back, so we don't have to calculate them
            // again
            ourOperands = std::move(m_operandStack);
        }

        // Hash the whole global stack and opcode
        SpvTypeInstKey key;
        key.words.add(opcode);
        key.words.addRange(ourOperands);

        // If we have seen this before, return the memoized instruction
        if (SpvInst** memoized = m_spvTypeInsts.tryGetValue(key))
        {
            // There could be another different slang IR inst that translates to
            // the same spir-v inst.
            // For example, both Ptr<T> and Ref<T> translates to the same pointer
            // type in spirv.
            // In this case we need to make sure we also
            // register `inst` to map it to the memoized spir-v inst.
            if (irInst)
                m_mapIRInstToSpvInst.addIfNotExists(irInst, *memoized);
            return *memoized;
        }

        // Otherwise, we can construct our instruction and record the result
        InstConstructScope scopeInst(this, opcode, irInst);
        SpvInst* spvInst = scopeInst;
        m_spvTypeInsts[key] = spvInst;

        // Emit our operands, this time with the resultId too
        emitOperand(resultId);
        m_operandStack.addRange(ourOperands);

        parent->addInst(spvInst);
        return spvInst;
    }

    template<typename OperandEmitFunc>
    SpvInst* emitInstMemoizedNoResultIDCustomOperandFunc(
        SpvInstParent* parent,
        IRInst* irInst,
        SpvOp opcode,
        const OperandEmitFunc& f)
    {
        List<SpvWord> ourOperands;
        {
            auto scopePeek = OperandMemoizeScope(this);
            f();
            // Steal our operands back, so we don't have to calculate them
            // again
            ourOperands = std::move(m_operandStack);
        }

        // Hash the whole global stack and opcode
        SpvTypeInstKey key;
        key.words.add(opcode);
        key.words.addRange(ourOperands);

        // If we have seen this before, return the memoized instruction
        if (SpvInst** memoized = m_spvTypeInsts.tryGetValue(key))
            return *memoized;

        // Otherwise, we can construct our instruction and record the result
        InstConstructScope scopeInst(this, opcode, irInst);
        SpvInst* spvInst = scopeInst;
        m_spvTypeInsts[key] = spvInst;

        m_operandStack.addRange(ourOperands);

        parent->addInst(spvInst);
        return spvInst;
    }
    //
    // Specific emit funcs
    //

#define SLANG_IN_SPIRV_EMIT_CONTEXT
#include "slang-emit-spirv-ops-debug-info-ext.h"
#include "slang-emit-spirv-ops.h"
#undef SLANG_IN_SPIRV_EMIT_CONTEXT

    /// The SPIRV OpExtInstImport inst that represents the GLSL450
    /// extended instruction set.
    SpvInst* m_glsl450ExtInst = nullptr;

    SpvInst* getGLSL450ExtInst()
    {
        if (m_glsl450ExtInst)
            return m_glsl450ExtInst;
        m_glsl450ExtInst = emitOpExtInstImport(
            getSection(SpvLogicalSectionID::ExtIntInstImports),
            nullptr,
            UnownedStringSlice("GLSL.std.450"));
        return m_glsl450ExtInst;
    }

    /// The SPIRV OpExtInstImport inst that represents the NonSemantic debug info
    /// extended instruction set.
    SpvInst* m_NonSemanticDebugInfoExtInst = nullptr;

    SpvInst* getNonSemanticDebugInfoExtInst()
    {
        if (m_NonSemanticDebugInfoExtInst)
            return m_NonSemanticDebugInfoExtInst;
        m_NonSemanticDebugInfoExtInst = emitOpExtInstImport(
            getSection(SpvLogicalSectionID::ExtIntInstImports),
            nullptr,
            UnownedStringSlice("NonSemantic.Shader.DebugInfo.100"));
        return m_NonSemanticDebugInfoExtInst;
    }

    /// The SPIRV OpExtInstImport inst that represents the NonSemantic debug info
    /// extended instruction set.
    SpvInst* m_NonSemanticDebugPrintfExtInst = nullptr;

    SpvInst* getNonSemanticDebugPrintfExtInst()
    {
        if (m_NonSemanticDebugPrintfExtInst)
            return m_NonSemanticDebugPrintfExtInst;
        m_NonSemanticDebugPrintfExtInst = emitOpExtInstImport(
            getSection(SpvLogicalSectionID::ExtIntInstImports),
            nullptr,
            UnownedStringSlice("NonSemantic.DebugPrintf"));
        return m_NonSemanticDebugPrintfExtInst;
    }

    static SpvStorageClass addressSpaceToStorageClass(AddressSpace addrSpace)
    {
        SLANG_EXHAUSTIVE_SWITCH_BEGIN
        switch (addrSpace)
        {
        case AddressSpace::Generic:
            return SpvStorageClassMax;
        case AddressSpace::ThreadLocal:
            return SpvStorageClassPrivate;
        case AddressSpace::GroupShared:
            return SpvStorageClassWorkgroup;
        case AddressSpace::Uniform:
            return SpvStorageClassUniform;
        case AddressSpace::Input:
        case AddressSpace::BuiltinInput:
            return SpvStorageClassInput;
        case AddressSpace::Output:
        case AddressSpace::BuiltinOutput:
            return SpvStorageClassOutput;
        case AddressSpace::TaskPayloadWorkgroup:
            return SpvStorageClassTaskPayloadWorkgroupEXT;
        case AddressSpace::Function:
            return SpvStorageClassFunction;
        case AddressSpace::StorageBuffer:
            return SpvStorageClassStorageBuffer;
        case AddressSpace::PushConstant:
            return SpvStorageClassPushConstant;
        case AddressSpace::RayPayloadKHR:
            return SpvStorageClassRayPayloadKHR;
        case AddressSpace::IncomingRayPayload:
            return SpvStorageClassIncomingRayPayloadKHR;
        case AddressSpace::CallableDataKHR:
            return SpvStorageClassCallableDataKHR;
        case AddressSpace::IncomingCallableData:
            return SpvStorageClassIncomingCallableDataKHR;
        case AddressSpace::HitObjectAttribute:
            return SpvStorageClassHitObjectAttributeNV;
        case AddressSpace::HitAttribute:
            return SpvStorageClassHitAttributeKHR;
        case AddressSpace::ShaderRecordBuffer:
            return SpvStorageClassShaderRecordBufferKHR;
        case AddressSpace::UniformConstant:
            return SpvStorageClassUniformConstant;
        case AddressSpace::Image:
            return SpvStorageClassImage;
        case AddressSpace::UserPointer:
            return SpvStorageClassPhysicalStorageBuffer;
        case AddressSpace::NodePayloadAMDX:
            return SpvStorageClassNodePayloadAMDX;
        case AddressSpace::Global:
        case AddressSpace::MetalObjectData:
        case AddressSpace::SpecializationConstant:
            // msvc is limiting us from putting the UNEXPECTED macro here, so
            // just fall out
            ;
        }
        SLANG_UNEXPECTED("Unhandled AddressSpace in addressSpaceToStorageClass");
        SLANG_EXHAUSTIVE_SWITCH_END
    }

    // Now that we've gotten the core infrastructure out of the way,
    // let's start looking at emitting some instructions that make
    // up a SPIR-V module.
    //
    // We will start with certain instructions that are required
    // to appear in a well-formed SPIR-V module for Vulkan, but
    // which do not directly relate to any instruction in the
    // Slang IR.

    /// Emit the mandatory "front-matter" instructions that
    /// the SPIR-V module must include to make it usable.
    void emitFrontMatter()
    {
        // TODO: We should ideally add SPIR-V capabilities to
        // the module as we emit instructions that require them.
        // For now we will always emit the `Shader` capability,
        // since every Vulkan shader module will use it.
        //
        emitOpCapability(
            getSection(SpvLogicalSectionID::Capabilities),
            nullptr,
            SpvCapabilityShader);

        // [2.4: Logical Layout of a Module]
        //
        // > The single required OpMemoryModel instruction.
        //
        // A memory model is always required in SPIR-V module.
        //
        // The Vulkan spec further says:
        //
        // > The `Logical` addressing model must be selected
        //
        // It isn't clear if the GLSL450 memory model is also
        // a requirement, but it is what glslang produces,
        // so we will use it for now.
        //
        emitOpMemoryModel(
            getSection(SpvLogicalSectionID::MemoryModel),
            nullptr,
            m_addressingMode,
            SpvMemoryModelGLSL450);
    }

    IRInst* m_defaultDebugSource = nullptr;

    Dictionary<UnownedStringSlice, SpvInst*> m_extensionInsts;
    SpvInst* ensureExtensionDeclaration(UnownedStringSlice name)
    {
        SpvInst* result = nullptr;
        if (m_extensionInsts.tryGetValue(name, result))
            return result;
        result = emitOpExtension(getSection(SpvLogicalSectionID::Extensions), nullptr, name);
        m_extensionInsts[name] = result;
        return result;
    }

    List<List<UnownedStringSlice>> m_anyExtension;
    void ensureAnyExtensionDeclaration(List<UnownedStringSlice> extensions)
    {
        if (!m_anyExtension.contains(extensions))
        {
            m_anyExtension.add(extensions);
        }
    }

    void emitSPIRVAnyExtension()
    {
        for (const auto& options : m_anyExtension)
        {
            bool found = false;
            for (UnownedStringSlice option : options)
            {
                if (m_extensionInsts.tryGetValue(option))
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                ensureExtensionDeclaration(options[0]);
            }
        }
    }

    SpvInst* ensureExtensionDeclarationBeforeSpv14(UnownedStringSlice name)
    {
        if (isSpirv14OrLater())
            return nullptr;
        return ensureExtensionDeclaration(name);
    }

    SpvInst* ensureExtensionDeclarationBeforeSpv15(UnownedStringSlice name)
    {
        if (isSpirv15OrLater())
            return nullptr;
        return ensureExtensionDeclaration(name);
    }

    SpvInst* ensureExtensionDeclarationBeforeSpv16(UnownedStringSlice name)
    {
        if (isSpirv16OrLater())
            return nullptr;
        return ensureExtensionDeclaration(name);
    }

    bool hasExtensionDeclaration(const UnownedStringSlice& name)
    {
        return m_extensionInsts.containsKey(name);
    }

    struct SpvTypeInstKey
    {
        List<SpvWord> words;
        bool operator==(const SpvTypeInstKey& other) const { return words == other.words; }
        const static bool kHasUniformHash = true;
        auto getHashCode() const
        {
            return Slang::getHashCode(
                reinterpret_cast<const char*>(words.getBuffer()),
                words.getCount() * sizeof(SpvWord));
        }
    };

    Dictionary<SpvTypeInstKey, SpvInst*> m_spvTypeInsts;

    bool shouldEmitSPIRVReflectionInfo()
    {
        return m_targetProgram->getOptionSet().getBoolOption(
            CompilerOptionName::VulkanEmitReflection);
    }

    void requirePhysicalStorageAddressing()
    {
        if (m_addressingMode == SpvAddressingModelPhysicalStorageBuffer64)
            return;
        ensureExtensionDeclaration(UnownedStringSlice("SPV_KHR_physical_storage_buffer"));
        requireSPIRVCapability(SpvCapabilityPhysicalStorageBufferAddresses);
        m_addressingMode = SpvAddressingModelPhysicalStorageBuffer64;
    }

    bool shouldEmitArrayStride(IRInst* elementType)
    {
        for (auto decor : elementType->getDecorations())
        {
            switch (decor->getOp())
            {
            case kIROp_SPIRVBufferBlockDecoration:
            case kIROp_SPIRVBlockDecoration:
                return false;
            }
        }
        return true;
    }

    // Next, let's look at emitting some of the instructions
    // that can occur at global scope.

    /// Emit an instruction that is expected to appear at the global scope of the SPIR-V module.
    ///
    /// Returns the corresponding SPIR-V instruction.
    ///
    SpvInst* emitGlobalInst(IRInst* inst)
    {
        switch (inst->getOp() & kIROpMask_OpMask)
        {
            // [3.32.6: Type-Declaration Instructions]
            //

        case kIROp_VoidType:
            return emitOpTypeVoid(inst);
        case kIROp_BoolType:
            return emitOpTypeBool(inst);

            // > OpTypeInt

        case kIROp_UInt16Type:
        case kIROp_Int16Type:
        case kIROp_UInt8Type:
        case kIROp_UIntType:
        case kIROp_UInt64Type:
        case kIROp_Int8Type:
        case kIROp_IntType:
        case kIROp_Int64Type:
            {
                const IntInfo i = getIntTypeInfo(as<IRType>(inst));
                if (i.width == 16)
                    requireSPIRVCapability(SpvCapabilityInt16);
                else if (i.width == 64)
                    requireSPIRVCapability(SpvCapabilityInt64);
                else if (i.width == 8)
                    requireSPIRVCapability(SpvCapabilityInt8);
                return emitOpTypeInt(
                    inst,
                    SpvLiteralInteger::from32(int32_t(i.width)),
                    SpvLiteralInteger::from32(i.isSigned));
            }

            // > OpTypeFloat

        case kIROp_HalfType:
        case kIROp_FloatType:
        case kIROp_DoubleType:
            {
                const FloatInfo i = getFloatingTypeInfo(as<IRType>(inst));
                if (inst->getOp() == kIROp_DoubleType)
                    requireSPIRVCapability(SpvCapabilityFloat64);
                else if (inst->getOp() == kIROp_HalfType)
                    requireSPIRVCapability(SpvCapabilityFloat16);
                return emitOpTypeFloat(inst, SpvLiteralInteger::from32(int32_t(i.width)));
            }
        case kIROp_PtrType:
        case kIROp_RefType:
        case kIROp_ConstRefType:
        case kIROp_OutType:
        case kIROp_InOutType:
            {
                SpvStorageClass storageClass = SpvStorageClassFunction;
                auto ptrType = as<IRPtrTypeBase>(inst);
                SLANG_ASSERT(ptrType);
                if (ptrType->hasAddressSpace())
                    storageClass = addressSpaceToStorageClass(ptrType->getAddressSpace());

                switch (storageClass)
                {
                case SpvStorageClassStorageBuffer:
                    ensureExtensionDeclaration(
                        UnownedStringSlice("SPV_KHR_storage_buffer_storage_class"));
                    break;
                case SpvStorageClassPhysicalStorageBuffer:
                    requirePhysicalStorageAddressing();
                    break;
                case SpvStorageClassNodePayloadAMDX:
                    requireSPIRVCapability(SpvCapabilityShaderEnqueueAMDX);
                    ensureExtensionDeclaration(UnownedStringSlice("SPV_AMDX_shader_enqueue"));
                    break;
                }

                auto valueType = ptrType->getValueType();
                // If we haven't emitted the inner type yet, we need to emit a forward declaration.
                bool useForwardDeclaration =
                    (!m_mapIRInstToSpvInst.containsKey(valueType) && as<IRStructType>(valueType) &&
                     storageClass == SpvStorageClassPhysicalStorageBuffer);
                SpvId valueTypeId;
                if (as<IRVoidType>(valueType))
                {
                    // Emit void* as uint*.
                    IRBuilder builder(valueType);
                    builder.setInsertBefore(valueType);
                    valueTypeId = getID(ensureInst(builder.getUIntType()));
                }
                else if (useForwardDeclaration)
                {
                    valueTypeId = getIRInstSpvID(valueType);
                }
                else if (storageClass == SpvStorageClassNodePayloadAMDX)
                {
                    auto spvValueType = ensureInst(valueType);
                    auto spvNodePayloadType = emitOpTypeNodePayloadArray(inst, spvValueType);
                    valueTypeId = getID(spvNodePayloadType);
                }
                else
                {
                    auto spvValueType = ensureInst(valueType);
                    valueTypeId = getID(spvValueType);
                }

                auto resultSpvType = emitOpTypePointer(inst, storageClass, valueTypeId);
                if (useForwardDeclaration)
                {
                    // After everything has been emitted, we will move the pointer definition to the
                    // end of the Types & Constants section.
                    if (m_forwardDeclaredPointers.addIfNotExists(
                            resultSpvType,
                            (IRPtrTypeBase*)inst))
                        emitOpTypeForwardPointer(resultSpvType, storageClass);
                }
                if (storageClass == SpvStorageClassPhysicalStorageBuffer)
                {
                    if (m_decoratedSpvInsts.add(getID(resultSpvType)))
                    {
                        IRSizeAndAlignment sizeAndAlignment;
                        uint32_t stride;

                        getNaturalSizeAndAlignment(
                            m_targetProgram->getOptionSet(),
                            valueType,
                            &sizeAndAlignment);
                        uint64_t valueSize = sizeAndAlignment.size;

                        // Any unsized data type (e.g. struct or array) will have size of
                        // kIndeterminateSize, in such case the stride is invalid, so we have to
                        // provide a non-zero value to pass the spirv validator.
                        stride = (valueSize >= (uint64_t)sizeAndAlignment.kIndeterminateSize)
                                     ? 0xFFFF
                                     : (uint32_t)sizeAndAlignment.getStride();
                        emitOpDecorateArrayStride(
                            getSection(SpvLogicalSectionID::Annotations),
                            nullptr,
                            resultSpvType,
                            SpvLiteralInteger::from32(stride));
                    }
                }
                return resultSpvType;
            }
        case kIROp_ConstantBufferType:
            SLANG_UNEXPECTED("Constant buffer type remaining in spirv emit");
        case kIROp_StructType:
            {
                List<IRType*> types;
                for (auto field : static_cast<IRStructType*>(inst)->getFields())
                {
                    types.add(field->getFieldType());
                }
                auto spvStructType = emitOpTypeStruct(inst, types);
                emitDecorations(inst, getID(spvStructType));

                auto structType = as<IRStructType>(inst);
                uint64_t structSize = 0;
                if (auto layoutDecor = structType->findDecoration<IRSizeAndAlignmentDecoration>())
                {
                    structSize = layoutDecor->getSize();
                }

                if (structSize >= (uint64_t)IRSizeAndAlignment::kIndeterminateSize)
                {
                    IRBuilder builder(inst);
                    if ((isSpirv14OrLater() ||
                         !inst->findDecorationImpl(kIROp_SPIRVBufferBlockDecoration)) &&
                        !inst->findDecorationImpl(kIROp_SPIRVBlockDecoration))
                    {
                        auto decoration = builder.addDecoration(inst, kIROp_SPIRVBlockDecoration);
                        emitDecoration(getID(spvStructType), decoration);
                    }
                }
                emitLayoutDecorations(as<IRStructType>(inst), getID(spvStructType));
                return spvStructType;
            }
        case kIROp_VectorType:
            {
                auto vectorType = static_cast<IRVectorType*>(inst);
                return ensureVectorType(
                    static_cast<IRBasicType*>(vectorType->getElementType())->getBaseType(),
                    static_cast<IRIntLit*>(vectorType->getElementCount())->getValue(),
                    vectorType);
            }
        case kIROp_CoopVectorType:
            {
                auto coopVecType = static_cast<IRCoopVectorType*>(inst);
                requireSPIRVCapability(SpvCapabilityCooperativeVectorNV);
                ensureExtensionDeclaration(UnownedStringSlice("SPV_NV_cooperative_vector"));
                return ensureCoopVecType(
                    static_cast<IRBasicType*>(coopVecType->getElementType())->getBaseType(),
                    static_cast<IRIntLit*>(coopVecType->getElementCount())->getValue(),
                    coopVecType);
            }
        case kIROp_CoopMatrixType:
            {
                requireSPIRVCapability(SpvCapabilityCooperativeMatrixKHR);
                ensureExtensionDeclaration(UnownedStringSlice("SPV_KHR_cooperative_matrix"));

                IRBuilder builder(m_irModule);
                auto coopMatType = static_cast<IRCoopMatrixType*>(inst);
                return emitOpTypeCoopMat(
                    coopMatType,
                    coopMatType->getElementType(),
                    emitIntConstant(
                        static_cast<IRIntLit*>(coopMatType->getScope())->getValue(),
                        builder.getIntType()),
                    emitIntConstant(
                        static_cast<IRIntLit*>(coopMatType->getRowCount())->getValue(),
                        builder.getIntType()),
                    emitIntConstant(
                        static_cast<IRIntLit*>(coopMatType->getColumnCount())->getValue(),
                        builder.getIntType()),
                    emitIntConstant(
                        static_cast<IRIntLit*>(coopMatType->getMatrixUse())->getValue(),
                        builder.getIntType()));
            }
        case kIROp_MatrixType:
            {
                auto matrixType = static_cast<IRMatrixType*>(inst);
                auto vectorSpvType = ensureVectorType(
                    static_cast<IRBasicType*>(matrixType->getElementType())->getBaseType(),
                    static_cast<IRIntLit*>(matrixType->getColumnCount())->getValue(),
                    nullptr);
                const auto columnCount =
                    static_cast<IRIntLit*>(matrixType->getRowCount())->getValue();
                auto matrixSPVType = emitOpTypeMatrix(
                    inst,
                    vectorSpvType,
                    SpvLiteralInteger::from32(int32_t(columnCount)));
                return matrixSPVType;
            }
        case kIROp_ArrayType:
        case kIROp_UnsizedArrayType:
            {
                auto irArrayType = static_cast<IRArrayTypeBase*>(inst);
                const auto elementType = irArrayType->getElementType();
                const auto arrayType =
                    inst->getOp() == kIROp_ArrayType
                        ? emitOpTypeArray(inst, elementType, irArrayType->getElementCount())
                        : emitOpTypeRuntimeArray(inst, elementType);
                auto strideInst = irArrayType->getArrayStride();
                if (strideInst && shouldEmitArrayStride(irArrayType->getElementType()))
                {
                    int stride = (int)getIntVal(strideInst);
                    emitOpDecorateArrayStride(
                        getSection(SpvLogicalSectionID::Annotations),
                        nullptr,
                        arrayType,
                        SpvLiteralInteger::from32(stride));
                }
                return arrayType;
            }
        case kIROp_AtomicType:
            {
                auto result = ensureInst(as<IRAtomicType>(inst)->getElementType());
                registerInst(inst, result);
                return result;
            }
        case kIROp_DescriptorHandleType:
            {
                IRBuilder builder(inst);
                builder.setInsertBefore(inst);
                return emitOpTypeVector(inst, builder.getUIntType(), SpvLiteralInteger::from32(2));
            }
        case kIROp_SubpassInputType:
            return ensureSubpassInputType(inst, cast<IRSubpassInputType>(inst));
        case kIROp_TextureType:
            return ensureTextureType(inst, cast<IRTextureType>(inst));
        case kIROp_SamplerStateType:
        case kIROp_SamplerComparisonStateType:
            return emitOpTypeSampler(inst);

        case kIROp_RaytracingAccelerationStructureType:
            requireSPIRVAnyCapability({SpvCapabilityRayTracingKHR, SpvCapabilityRayQueryKHR});
            ensureAnyExtensionDeclaration(
                {UnownedStringSlice("SPV_KHR_ray_tracing"),
                 UnownedStringSlice("SPV_KHR_ray_query")});
            return emitOpTypeAccelerationStructure(inst);

        case kIROp_RayQueryType:
            ensureExtensionDeclaration(UnownedStringSlice("SPV_KHR_ray_query"));
            requireSPIRVCapability(SpvCapabilityRayQueryKHR);
            return emitOpTypeRayQuery(inst);

        case kIROp_HitObjectType:
            ensureExtensionDeclaration(UnownedStringSlice("SPV_NV_shader_invocation_reorder"));
            requireSPIRVCapability(SpvCapabilityShaderInvocationReorderNV);
            return emitOpTypeHitObject(inst);

        case kIROp_FuncType:
            // > OpTypeFunction
            //
            // Both Slang and SPIR-V encode a function type
            // with the result-type operand coming first,
            // followed by operand sfor all the parameter types.
            //
            return emitOpTypeFunction(
                inst,
                static_cast<IRFuncType*>(inst)->getResultType(),
                static_cast<IRFuncType*>(inst)->getParamTypes());

        case kIROp_RateQualifiedType:
            {
                auto result = ensureInst(as<IRRateQualifiedType>(inst)->getValueType());
                registerInst(inst, result);
                return result;
            }
            // > OpTypeForwardPointer

        case kIROp_Func:
            // [3.32.6: Function Instructions]
            //
            // > OpFunction
            //
            // Functions are complex enough that we'll handle
            // them in a dedicated subroutine.
            //
            return emitFunc(as<IRFunc>(inst));

        case kIROp_BoolLit:
        case kIROp_IntLit:
        case kIROp_FloatLit:
        case kIROp_StringLit:
        case kIROp_PtrLit:
            {
                return emitLit(inst);
            }
        case kIROp_MakeVectorFromScalar:
            {
                const auto scalar = inst->getOperand(0);
                const auto vecTy = as<IRVectorType>(inst->getDataType());
                SLANG_ASSERT(vecTy);
                const auto numElems = as<IRIntLit>(vecTy->getElementCount());
                SLANG_ASSERT(numElems);
                return emitSplat(
                    getSection(SpvLogicalSectionID::ConstantsAndTypes),
                    inst,
                    scalar,
                    numElems->getValue());
            }
        case kIROp_MakeVector:
        case kIROp_MakeCoopVector:
        case kIROp_MakeArray:
        case kIROp_MakeStruct:
            return emitCompositeConstruct(getSection(SpvLogicalSectionID::ConstantsAndTypes), inst);
        case kIROp_MakeArrayFromElement:
            return emitMakeArrayFromElement(
                getSection(SpvLogicalSectionID::ConstantsAndTypes),
                inst);
        case kIROp_MakeMatrix:
            return emitMakeMatrix(getSection(SpvLogicalSectionID::ConstantsAndTypes), inst);
        case kIROp_MakeMatrixFromScalar:
            return emitMakeMatrixFromScalar(
                getSection(SpvLogicalSectionID::ConstantsAndTypes),
                inst);
        case kIROp_GlobalParam:
            return emitGlobalParam(as<IRGlobalParam>(inst));
        case kIROp_GlobalVar:
            return emitGlobalVar(as<IRGlobalVar>(inst));
        case kIROp_SPIRVAsmOperandBuiltinVar:
            return emitBuiltinVar(inst);
        case kIROp_Var:
            return emitVar(getSection(SpvLogicalSectionID::GlobalVariables), inst);
            // ...

        case kIROp_Specialize:
            {
                const auto s = as<IRSpecialize>(inst);
                const auto g = s->getBase();
                const auto e = "Specialize instruction remains in IR for SPIR-V emit, is something "
                               "undefined?\n" +
                               dumpIRToString(g);
                SLANG_UNEXPECTED(e.getBuffer());
            }

        case kIROp_DebugSource:
            {
                ensureExtensionDeclaration(UnownedStringSlice("SPV_KHR_non_semantic_info"));
                auto debugSource = as<IRDebugSource>(inst);
                auto sourceStr = as<IRStringLit>(debugSource->getSource())->getStringSlice();
                // If source content is empty, skip the content operand.
                if (sourceStr.getLength() == 0)
                {
                    return emitOpDebugSource(
                        getSection(SpvLogicalSectionID::ConstantsAndTypes),
                        inst,
                        inst->getFullType(),
                        getNonSemanticDebugInfoExtInst(),
                        debugSource->getFileName());
                }
                // SPIRV does not allow string lits longer than 65535, so we need to split the
                // source string in OpDebugSourceContinued instructions.
                auto sourceStrHead =
                    sourceStr.getLength() > 65535 ? sourceStr.head(65535) : sourceStr;
                auto spvStrHead = emitInst(
                    getSection(SpvLogicalSectionID::DebugStringsAndSource),
                    nullptr,
                    SpvOpString,
                    kResultID,
                    SpvLiteralBits::fromUnownedStringSlice(sourceStrHead));

                auto result = emitOpDebugSource(
                    getSection(SpvLogicalSectionID::ConstantsAndTypes),
                    inst,
                    inst->getFullType(),
                    getNonSemanticDebugInfoExtInst(),
                    debugSource->getFileName(),
                    spvStrHead);

                for (Index start = 65535; start < sourceStr.getLength(); start += 65535)
                {
                    auto slice = sourceStr.tail(start);
                    slice = slice.getLength() > 65535 ? slice.head(65535) : slice;
                    auto sliceSpvStr = emitInst(
                        getSection(SpvLogicalSectionID::DebugStringsAndSource),
                        nullptr,
                        SpvOpString,
                        kResultID,
                        SpvLiteralBits::fromUnownedStringSlice(slice));
                    emitOpDebugSourceContinued(
                        getSection(SpvLogicalSectionID::ConstantsAndTypes),
                        nullptr,
                        m_voidType,
                        getNonSemanticDebugInfoExtInst(),
                        sliceSpvStr);
                }

                auto moduleInst = inst->getModule()->getModuleInst();
                if (!m_defaultDebugSource)
                    m_defaultDebugSource = debugSource;
                if (!m_mapIRInstToSpvDebugInst.containsKey(moduleInst))
                {
                    IRBuilder builder(inst);
                    builder.setInsertBefore(inst);
                    auto translationUnit = emitOpDebugCompilationUnit(
                        getSection(SpvLogicalSectionID::ConstantsAndTypes),
                        moduleInst,
                        inst->getFullType(),
                        getNonSemanticDebugInfoExtInst(),
                        emitIntConstant(100, builder.getUIntType()), // ExtDebugInfo version.
                        emitIntConstant(5, builder.getUIntType()),   // DWARF version.
                        result,
                        emitIntConstant(
                            SpvSourceLanguageSlang,
                            builder.getUIntType())); // Language.
                    registerDebugInst(moduleInst, translationUnit);
                }
                return result;
            }
        case kIROp_GetStringHash:
            return emitGetStringHash(inst);
        case kIROp_AttributedType:
            return ensureInst(as<IRAttributedType>(inst)->getBaseType());
        case kIROp_AllocateOpaqueHandle:
            return nullptr;
        case kIROp_HLSLTriangleStreamType:
        case kIROp_HLSLLineStreamType:
        case kIROp_HLSLPointStreamType:
        case kIROp_VerticesType:
        case kIROp_IndicesType:
        case kIROp_PrimitivesType:
            return nullptr;
        default:
            {
                if (as<IRSPIRVAsmOperand>(inst))
                    return nullptr;
                String e = "Unhandled global inst in spirv-emit:\n" +
                           dumpIRToString(inst, {IRDumpOptions::Mode::Detailed, 0});
                SLANG_UNIMPLEMENTED_X(e.begin());
            }
        }
    }

    SpvImageFormat getSpvImageFormat(IRTextureTypeBase* type)
    {
        ImageFormat imageFormat =
            type->hasFormat() ? (ImageFormat)type->getFormat() : ImageFormat::unknown;
        switch (imageFormat)
        {
        case ImageFormat::unknown:
            return SpvImageFormatUnknown;
        case ImageFormat::rgba32f:
            return SpvImageFormatRgba32f;
        case ImageFormat::rgba16f:
            return SpvImageFormatRgba16f;
        case ImageFormat::rg32f:
            return SpvImageFormatRg32f;
        case ImageFormat::rg16f:
            return SpvImageFormatRg16f;
        case ImageFormat::r11f_g11f_b10f:
            return SpvImageFormatR11fG11fB10f;
        case ImageFormat::r32f:
            return SpvImageFormatR32f;
        case ImageFormat::r16f:
            return SpvImageFormatR16f;
        case ImageFormat::rgba16:
            return SpvImageFormatRgba16;
        case ImageFormat::rgb10_a2:
            return SpvImageFormatRgb10A2;
        case ImageFormat::rgba8:
            return SpvImageFormatRgba8;
        case ImageFormat::rg16:
            return SpvImageFormatRg16;
        case ImageFormat::rg8:
            return SpvImageFormatRg8;
        case ImageFormat::r16:
            return SpvImageFormatR16;
        case ImageFormat::r8:
            return SpvImageFormatR8;
        case ImageFormat::rgba16_snorm:
            return SpvImageFormatRgba16Snorm;
        case ImageFormat::rgba8_snorm:
            return SpvImageFormatRgba8Snorm;
        case ImageFormat::rg16_snorm:
            return SpvImageFormatRg16Snorm;
        case ImageFormat::rg8_snorm:
            return SpvImageFormatRg8Snorm;
        case ImageFormat::r16_snorm:
            return SpvImageFormatR16Snorm;
        case ImageFormat::r8_snorm:
            return SpvImageFormatR8Snorm;
        case ImageFormat::rgba32i:
            return SpvImageFormatRgba32i;
        case ImageFormat::rgba16i:
            return SpvImageFormatRgba16i;
        case ImageFormat::rgba8i:
            return SpvImageFormatRgba8i;
        case ImageFormat::rg32i:
            return SpvImageFormatRg32i;
        case ImageFormat::rg16i:
            return SpvImageFormatRg16i;
        case ImageFormat::rg8i:
            return SpvImageFormatRg8i;
        case ImageFormat::r32i:
            return SpvImageFormatR32i;
        case ImageFormat::r16i:
            return SpvImageFormatR16i;
        case ImageFormat::r8i:
            return SpvImageFormatR8i;
        case ImageFormat::rgba32ui:
            return SpvImageFormatRgba32ui;
        case ImageFormat::rgba16ui:
            return SpvImageFormatRgba16ui;
        case ImageFormat::rgb10_a2ui:
            return SpvImageFormatRgb10a2ui;
        case ImageFormat::rgba8ui:
            return SpvImageFormatRgba8ui;
        case ImageFormat::rg32ui:
            return SpvImageFormatRg32ui;
        case ImageFormat::rg16ui:
            return SpvImageFormatRg16ui;
        case ImageFormat::rg8ui:
            return SpvImageFormatRg8ui;
        case ImageFormat::r32ui:
            return SpvImageFormatR32ui;
        case ImageFormat::r16ui:
            return SpvImageFormatR16ui;
        case ImageFormat::r8ui:
            return SpvImageFormatR8ui;
        case ImageFormat::r64ui:
            return SpvImageFormatR64ui;
        case ImageFormat::r64i:
            return SpvImageFormatR64i;
        default:
            const auto imageFormatInfo = getImageFormatInfo(imageFormat);
            m_sink->diagnose(
                SourceLoc(),
                Diagnostics::imageFormatUnsupportedByBackend,
                imageFormatInfo.name,
                "SPIRV",
                "unknown");
            return SpvImageFormatUnknown;
        }
    }

    static SpvStorageClass getSpvStorageClass(IRPtrTypeBase* ptrType)
    {
        SpvStorageClass storageClass = SpvStorageClassFunction;
        if (ptrType && ptrType->hasAddressSpace())
        {
            storageClass = addressSpaceToStorageClass(ptrType->getAddressSpace());
        }
        return storageClass;
    }

    // https://registry.khronos.org/vulkan/specs/1.3/html/chap37.html#VUID-StandaloneSpirv-DescriptorSet-06491
    // Only UniformConstant, Uniform or StorageBuffer storage class are allowed to be decorated with
    // descriptor set or binding.
    static inline bool isBindingAllowed(SpvStorageClass storageClass)
    {
        switch (storageClass)
        {
        case SpvStorageClassUniformConstant:
        case SpvStorageClassUniform:
        case SpvStorageClassStorageBuffer:
            return true;
        default:
            return false;
        }
    }

    SpvCapability getImageFormatCapability(SpvImageFormat format)
    {
        switch (format)
        {
        case SpvImageFormatUnknown:
        case SpvImageFormatRgba32f:
        case SpvImageFormatRgba16f:
        case SpvImageFormatR32f:
        case SpvImageFormatRgba8:
        case SpvImageFormatRgba8Snorm:
        case SpvImageFormatRgba32i:
        case SpvImageFormatRgba16i:
        case SpvImageFormatRgba8i:
        case SpvImageFormatR32i:
        case SpvImageFormatRgba32ui:
        case SpvImageFormatRgba16ui:
        case SpvImageFormatRgba8ui:
        case SpvImageFormatR32ui:
            return SpvCapabilityShader;
        case SpvImageFormatR64ui:
        case SpvImageFormatR64i:
            return SpvCapabilityInt64ImageEXT;
        default:
            return SpvCapabilityStorageImageExtendedFormats;
        }
    }

    void setImageFormatCapabilityAndExtension(
        SpvImageFormat format,
        SpvCapability_ setCapabilityMask)
    {
        switch (format)
        {
        case SpvImageFormatUnknown:
        case SpvImageFormatRgba32f:
        case SpvImageFormatRgba16f:
        case SpvImageFormatR32f:
        case SpvImageFormatRgba8:
        case SpvImageFormatRgba8Snorm:
        case SpvImageFormatRgba32i:
        case SpvImageFormatRgba16i:
        case SpvImageFormatRgba8i:
        case SpvImageFormatR32i:
        case SpvImageFormatRgba32ui:
        case SpvImageFormatRgba16ui:
        case SpvImageFormatRgba8ui:
        case SpvImageFormatR32ui:
            if (setCapabilityMask == SpvCapabilityShader)
                return;
            requireSPIRVCapability(SpvCapabilityShader);
            return;
        case SpvImageFormatR64ui:
        case SpvImageFormatR64i:
            if (setCapabilityMask == SpvCapabilityInt64ImageEXT)
                return;
            ensureExtensionDeclaration(UnownedStringSlice("SPV_EXT_shader_image_int64"));
            requireSPIRVCapability(SpvCapabilityInt64ImageEXT);
            return;
        default:
            if (setCapabilityMask == SpvCapabilityStorageImageExtendedFormats)
                return;
            requireSPIRVCapability(SpvCapabilityStorageImageExtendedFormats);
            return;
        }
    }

    struct ImageOpConstants
    {
        enum : SpvWord
        {
            // Some untyped constants from OpTypeImage
            // https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpTypeImage

            // indicates not a depth image
            notDepthImage = 0,
            // indicates a depth image
            isDepthImage = 1,
            // means no indication as to whether this is a depth or non-depth image
            unknownDepthImage = 2,

            // indicates non-arrayed content
            notArrayed = 0,
            // indicates arrayed content
            isArrayed = 1,

            // indicates this is only known at run time, not at compile time
            sampledUnknown = 0,
            // indicates an image compatible with sampling operations
            sampledImage = 1,
            // indicates an image compatible with read/write operations (a storage or subpass data
            // image).
            readWriteImage = 2,

            // indicates single-sampled content
            notMultisampled = 0,
            // indicates multisampled content
            isMultisampled = 1,
        };
    };

    SpvInst* ensureSubpassInputType(IRInst* assignee, IRSubpassInputType* inst)
    {
        IRInst* sampledType = inst->getElementType();
        SpvDim dim = SpvDimSubpassData;
        SpvWord ms = inst->isMultisample() ? ImageOpConstants::isMultisampled
                                           : ImageOpConstants::notMultisampled;
        SpvWord sampled = 2;
        requireSPIRVCapability(SpvCapabilityInputAttachment);
        requireSPIRVCapability(SpvCapabilityStorageImageReadWithoutFormat);
        setImageFormatCapabilityAndExtension(SpvImageFormatUnknown, SpvCapabilityShader);
        return emitOpTypeImage(
            assignee,
            getVectorElementType((IRType*)sampledType),
            dim,
            SpvLiteralInteger::from32(ImageOpConstants::unknownDepthImage),
            SpvLiteralInteger::from32(0),
            SpvLiteralInteger::from32(ms),
            SpvLiteralInteger::from32(sampled),
            SpvImageFormatUnknown);
    }

    SpvInst* ensureTextureType(IRInst* assignee, IRTextureTypeBase* inst)
    {
        IRInst* sampledType = inst->getElementType();
        SpvDim dim = SpvDim1D; // Silence uninitialized warnings from msvc...
        switch (inst->GetBaseShape())
        {
        case SLANG_TEXTURE_1D:
            dim = SpvDim1D;
            break;
        case SLANG_TEXTURE_2D:
            dim = SpvDim2D;
            break;
        case SLANG_TEXTURE_3D:
            dim = SpvDim3D;
            break;
        case SLANG_TEXTURE_CUBE:
            dim = SpvDimCube;
            break;
        case SLANG_TEXTURE_BUFFER:
            dim = SpvDimBuffer;
            break;
        }
        SpvWord arrayed =
            inst->isArray() ? ImageOpConstants::isArrayed : ImageOpConstants::notArrayed;

        // Vulkan spec 16.1: "The “Depth” operand of OpTypeImage is ignored."
        SpvWord depth =
            ImageOpConstants::unknownDepthImage; // No knowledge of if this is a depth image
        SpvWord ms = inst->isMultisample() ? ImageOpConstants::isMultisampled
                                           : ImageOpConstants::notMultisampled;

        SpvWord sampled = ImageOpConstants::sampledUnknown;
        switch (inst->getAccess())
        {
        case SlangResourceAccess::SLANG_RESOURCE_ACCESS_READ_WRITE:
        case SlangResourceAccess::SLANG_RESOURCE_ACCESS_RASTER_ORDERED:
        case SlangResourceAccess::SLANG_RESOURCE_ACCESS_WRITE:
            sampled = ImageOpConstants::readWriteImage;
            break;
        case SlangResourceAccess::SLANG_RESOURCE_ACCESS_NONE:
        case SlangResourceAccess::SLANG_RESOURCE_ACCESS_READ:
            sampled = ImageOpConstants::sampledImage;
            break;
        }

        SpvImageFormat format = getSpvImageFormat(inst);
        // If format is unknown, we need to deduce the format if there is
        // unorm or snorm attributes on the sampled type.
        if (auto attribType = as<IRAttributedType>(sampledType))
        {
            sampledType = unwrapAttributedType(sampledType);
            if (format == SpvImageFormatUnknown)
            {
                IRIntegerValue vectorSize = 1;
                if (auto vecType = as<IRVectorType>(sampledType))
                    vectorSize = getIntVal(vecType->getElementCount());

                for (auto attr : attribType->getAllAttrs())
                {
                    switch (attr->getOp())
                    {
                    case kIROp_UNormAttr:
                        switch (vectorSize)
                        {
                        case 1:
                            format = SpvImageFormatR8;
                            break;
                        case 2:
                            format = SpvImageFormatRg8;
                            break;
                        case 3:
                            format = SpvImageFormatRgba8;
                            break;
                        case 4:
                            format = SpvImageFormatRgba8;
                            break;
                        }
                        break;
                    case kIROp_SNormAttr:
                        switch (vectorSize)
                        {
                        case 1:
                            format = SpvImageFormatR8Snorm;
                            break;
                        case 2:
                            format = SpvImageFormatRg8Snorm;
                            break;
                        case 3:
                            format = SpvImageFormatRgba8Snorm;
                            break;
                        case 4:
                            format = SpvImageFormatRgba8Snorm;
                            break;
                        }
                        break;
                    }
                }
            }
        }

        //
        // Capabilities, according to section 3.8
        //
        // SPIR-V requires that the sampled/rw info on the image isn't unknown
        SLANG_ASSERT(
            sampled == ImageOpConstants::sampledImage ||
            sampled == ImageOpConstants::readWriteImage);
        if (ms == ImageOpConstants::isMultisampled)
            requireSPIRVCapability(SpvCapabilityStorageImageMultisample);
        switch (dim)
        {
        case SpvDim1D:
            requireSPIRVCapability(
                sampled == ImageOpConstants::sampledImage ? SpvCapabilitySampled1D
                                                          : SpvCapabilityImage1D);
            break;
        case SpvDim2D:
            // Also requires Shader or Kernel, but these are a given (?)
            if (sampled == ImageOpConstants::readWriteImage &&
                ms == ImageOpConstants::isMultisampled && arrayed == ImageOpConstants::isArrayed)
                requireSPIRVCapability(SpvCapabilityImageMSArray);
            break;
        case SpvDim3D:
            break;
        case SpvDimCube:
            // Requires shader also
            if (sampled == ImageOpConstants::readWriteImage &&
                arrayed == ImageOpConstants::isArrayed)
                requireSPIRVCapability(SpvCapabilityImageCubeArray);
            break;
        case SpvDimRect:
            requireSPIRVCapability(
                sampled == ImageOpConstants::sampledImage ? SpvCapabilitySampledRect
                                                          : SpvCapabilityImageRect);
            break;
        case SpvDimBuffer:
            requireSPIRVCapability(
                sampled == ImageOpConstants::sampledImage ? SpvCapabilitySampledBuffer
                                                          : SpvCapabilityImageBuffer);
            break;
        case SpvDimTileImageDataEXT:
            SLANG_UNIMPLEMENTED_X("OpTypeImage Capabilities for SpvDimTileImageDataEXT");
            break;
        }
        if (format == SpvImageFormatUnknown && sampled == ImageOpConstants::readWriteImage)
        {
            // TODO: It may not be necessary to have both of these
            // depending on if we read or write
            requireSPIRVCapability(SpvCapabilityStorageImageReadWithoutFormat);
            requireSPIRVCapability(SpvCapabilityStorageImageWriteWithoutFormat);
        }

        setImageFormatCapabilityAndExtension(format, SpvCapabilityShader);

        //
        // The op itself
        //
        auto sampledElementType = getSPIRVSampledElementType(sampledType);
        if (inst->isCombined())
        {
            auto imageType = emitOpTypeImage(
                nullptr,
                sampledElementType,
                dim,
                SpvLiteralInteger::from32(depth),
                SpvLiteralInteger::from32(arrayed),
                SpvLiteralInteger::from32(ms),
                SpvLiteralInteger::from32(sampled),
                format);
            return emitOpTypeSampledImage(assignee, imageType);
        }

        return emitOpTypeImage(
            assignee,
            sampledElementType,
            dim,
            SpvLiteralInteger::from32(depth),
            SpvLiteralInteger::from32(arrayed),
            SpvLiteralInteger::from32(ms),
            SpvLiteralInteger::from32(sampled),
            format);
    }

    // Ensures an SpvInst for the specified vector type is emitted.
    // `inst` represents an optional `IRVectorType` inst representing the vector type, if
    // it is nullptr, this function will create one.
    SpvInst* ensureVectorType(BaseType baseType, IRIntegerValue elementCount, IRVectorType* inst)
    {
        if (!inst)
        {
            IRBuilder builder(m_irModule);
            builder.setInsertInto(m_irModule->getModuleInst());
            inst = builder.getVectorType(
                builder.getBasicType(baseType),
                builder.getIntValue(builder.getIntType(), elementCount));
        }
        auto result = emitOpTypeVector(
            inst,
            inst->getElementType(),
            SpvLiteralInteger::from32(int32_t(elementCount)));
        return result;
    }

    /// Similar to ensureVectorType but for CoopVecType
    SpvInst* ensureCoopVecType(
        BaseType baseType,
        IRIntegerValue elementCount,
        IRCoopVectorType* inst)
    {
        IRBuilder builder(m_irModule);
        if (!inst)
        {
            builder.setInsertInto(m_irModule->getModuleInst());
            inst = builder.getCoopVectorType(
                builder.getBasicType(baseType),
                builder.getIntValue(builder.getIntType(), elementCount));
        }
        auto result = emitOpTypeCoopVec(
            inst,
            inst->getElementType(),
            emitIntConstant(elementCount, builder.getIntType()));
        return result;
    }

    bool _maybeEmitInterpolationModifierDecoration(IRInterpolationMode mode, SpvId varInst)
    {
        switch (mode)
        {
        case IRInterpolationMode::NoInterpolation:
            emitOpDecorate(
                getSection(SpvLogicalSectionID::Annotations),
                nullptr,
                varInst,
                SpvDecorationFlat);
            return true;
        case IRInterpolationMode::NoPerspective:
            emitOpDecorate(
                getSection(SpvLogicalSectionID::Annotations),
                nullptr,
                varInst,
                SpvDecorationNoPerspective);
            return true;
        case IRInterpolationMode::Linear:
            return true;
        case IRInterpolationMode::Sample:
            emitOpDecorate(
                getSection(SpvLogicalSectionID::Annotations),
                nullptr,
                varInst,
                SpvDecorationSample);
            return true;
        case IRInterpolationMode::Centroid:
            emitOpDecorate(
                getSection(SpvLogicalSectionID::Annotations),
                nullptr,
                varInst,
                SpvDecorationCentroid);
            return true;
        default:
            return false;
        }
    }
    void emitSystemVarDecoration(IRInst* var, SpvInst* varInst)
    {
        for (auto decor : var->getDecorations())
        {
            switch (decor->getOp())
            {
            case kIROp_GLSLPrimitivesRateDecoration:
                emitOpDecorate(
                    getSection(SpvLogicalSectionID::Annotations),
                    decor,
                    varInst,
                    SpvDecorationPerPrimitiveEXT);
                break;
            }
        }
    }

    void emitVarLayout(IRInst* var, SpvInst* varInst, IRVarLayout* layout)
    {
        auto dataType = as<IRPtrTypeBase>(var->getDataType());
        SpvStorageClass storageClass = getSpvStorageClass(dataType);

        bool isBindingDecorationAllowed = isBindingAllowed(storageClass);
        bool needDefaultSetBindingDecoration = false;
        bool hasExplicitSetBinding = false;
        bool isDescirptorSetDecorated = false;
        for (auto rr : layout->getOffsetAttrs())
        {
            UInt index = rr->getOffset();
            UInt space = rr->getSpace();
            switch (rr->getResourceKind())
            {
            case LayoutResourceKind::Uniform:
                break;

            case LayoutResourceKind::VaryingInput:
                emitOpDecorateLocation(
                    getSection(SpvLogicalSectionID::Annotations),
                    nullptr,
                    varInst,
                    SpvLiteralInteger::from32(int32_t(index)));
                if (space != 0)
                {
                    emitOpDecorateIndex(
                        getSection(SpvLogicalSectionID::Annotations),
                        nullptr,
                        varInst,
                        SpvLiteralInteger::from32(int32_t(space)));
                }
                break;
            case LayoutResourceKind::VaryingOutput:
                emitOpDecorateLocation(
                    getSection(SpvLogicalSectionID::Annotations),
                    nullptr,
                    varInst,
                    SpvLiteralInteger::from32(int32_t(index)));
                if (space != 0)
                {
                    emitOpDecorateIndex(
                        getSection(SpvLogicalSectionID::Annotations),
                        nullptr,
                        varInst,
                        SpvLiteralInteger::from32(int32_t(space)));
                }
                break;

            case LayoutResourceKind::SpecializationConstant:
                emitOpDecorateSpecId(
                    getSection(SpvLogicalSectionID::Annotations),
                    nullptr,
                    varInst,
                    SpvLiteralInteger::from32(int32_t(index)));
                break;

            case LayoutResourceKind::ConstantBuffer:
            case LayoutResourceKind::ShaderResource:
            case LayoutResourceKind::UnorderedAccess:
            case LayoutResourceKind::SamplerState:
            case LayoutResourceKind::DescriptorTableSlot:
                if (!isBindingDecorationAllowed)
                    break;

                emitOpDecorateBinding(
                    getSection(SpvLogicalSectionID::Annotations),
                    nullptr,
                    varInst,
                    SpvLiteralInteger::from32(int32_t(index)));

                if (!isDescirptorSetDecorated)
                {
                    if (space)
                    {
                        emitOpDecorateDescriptorSet(
                            getSection(SpvLogicalSectionID::Annotations),
                            nullptr,
                            varInst,
                            SpvLiteralInteger::from32(int32_t(space)));
                        isDescirptorSetDecorated = true;
                    }
                    else
                    {
                        needDefaultSetBindingDecoration = true;
                    }
                }
                break;
            case LayoutResourceKind::RegisterSpace:
                if (!isDescirptorSetDecorated && isBindingDecorationAllowed)
                {
                    emitOpDecorateDescriptorSet(
                        getSection(SpvLogicalSectionID::Annotations),
                        nullptr,
                        varInst,
                        SpvLiteralInteger::from32(int32_t(index)));
                    hasExplicitSetBinding = true;
                    isDescirptorSetDecorated = true;
                }
                break;
            case LayoutResourceKind::InputAttachmentIndex:
                {
                    emitOpDecorateInputAttachmentIndex(
                        getSection(SpvLogicalSectionID::Annotations),
                        nullptr,
                        varInst,
                        SpvLiteralInteger::from32((int32_t)index));
                }
                break;
            default:
                break;
            }
        }
        if (needDefaultSetBindingDecoration && !hasExplicitSetBinding && !isDescirptorSetDecorated)
        {
            emitOpDecorateDescriptorSet(
                getSection(SpvLogicalSectionID::Annotations),
                nullptr,
                varInst,
                SpvLiteralInteger::from32(int32_t(0)));
        }

        bool anyModifiers = (var->findDecoration<IRInterpolationModeDecoration>() != nullptr);

        // If the user didn't explicitly qualify a varying
        // with integer type, then we need to explicitly
        // add the `flat` modifier for GLSL.
        if (!anyModifiers)
        {
            // Only emit a default `flat` for fragment
            // stage varying inputs.
            if (layout && layout->getStage() == Stage::Fragment &&
                layout->usesResourceKind(LayoutResourceKind::VaryingInput))
            {
                const auto ptrType = as<IRPtrTypeBase>(var->getDataType());
                if (ptrType && isIntegralScalarOrCompositeType(ptrType->getValueType()))
                    emitOpDecorate(
                        getSection(SpvLogicalSectionID::Annotations),
                        nullptr,
                        varInst,
                        SpvDecorationFlat);
            }
        }

        for (auto decor : var->getDecorations())
        {
            switch (decor->getOp())
            {
            case kIROp_GLSLPrimitivesRateDecoration:
                emitOpDecorate(
                    getSection(SpvLogicalSectionID::Annotations),
                    decor,
                    varInst,
                    SpvDecorationPerPrimitiveEXT);
                break;
            case kIROp_RequireSPIRVDescriptorIndexingExtensionDecoration:
                ensureExtensionDeclarationBeforeSpv15(
                    UnownedStringSlice("SPV_EXT_descriptor_indexing"));
                requireSPIRVCapability(SpvCapabilityRuntimeDescriptorArray);
                break;
            }
        }
    }

    void maybeEmitName(SpvInst* spvInst, IRInst* irInst)
    {
        if (auto nameDecor = irInst->findDecoration<IRNameHintDecoration>())
        {
            emitOpName(
                getSection(SpvLogicalSectionID::DebugNames),
                nullptr,
                spvInst,
                nameDecor->getName());
        }
    }

    /// Emit a specialization constant.
    SpvInst* emitSpecializationConstant(IRGlobalParam* param, IRVarOffsetAttr* offset)
    {
        IRInst* defaultVal = nullptr;
        if (auto defaultValDecor = param->findDecoration<IRDefaultValueDecoration>())
        {
            defaultVal = defaultValDecor->getOperand(0);
        }
        else
        {
            IRBuilder builder(param);
            builder.setInsertBefore(param);
            defaultVal = builder.emitDefaultConstruct(param->getDataType());
        }

        SpvInst* result = nullptr;
        if (as<IRBoolType>(defaultVal->getDataType()))
        {
            bool value = false;
            if (auto boolLit = as<IRBoolLit>(defaultVal))
            {
                value = boolLit->getValue();
            }
            if (value)
            {
                result = emitOpSpecConstantTrue(
                    getSection(SpvLogicalSectionID::ConstantsAndTypes),
                    param,
                    param->getDataType());
            }
            else
            {
                result = emitOpSpecConstantFalse(
                    getSection(SpvLogicalSectionID::ConstantsAndTypes),
                    param,
                    param->getDataType());
            }
        }
        else if (auto type = as<IRBasicType>(defaultVal->getDataType()))
        {
            result = emitOpSpecConstant(
                getSection(SpvLogicalSectionID::ConstantsAndTypes),
                param,
                param->getDataType(),
                getLiteralBits(type, defaultVal));
        }
        else if (as<IRVectorType>(defaultVal->getDataType()))
        {
            List<IRInst*> operands;
            auto makeVector = as<IRMakeVector>(defaultVal);
            for (UInt i = 0; i < makeVector->getOperandCount(); i++)
            {
                operands.add(makeVector->getOperand(i));
            }
            result = emitOpSpecConstantComposite(
                getSection(SpvLogicalSectionID::ConstantsAndTypes),
                param,
                param->getDataType(),
                operands);
        }

        emitOpDecorateSpecId(
            getSection(SpvLogicalSectionID::Annotations),
            nullptr,
            result,
            SpvLiteralInteger::from32((uint32_t)offset->getOffset()));

        maybeEmitName(result, param);
        return result;
    }

    /// Emit a global parameter definition.
    SpvInst* emitGlobalParam(IRGlobalParam* param)
    {
        auto layout = getVarLayout(param);
        if (layout)
        {
            if (auto offset = layout->findOffsetAttr(LayoutResourceKind::SpecializationConstant))
            {
                return emitSpecializationConstant(param, offset);
            }
        }
        auto storageClass = SpvStorageClassUniform;
        if (auto ptrType = as<IRPtrTypeBase>(param->getDataType()))
        {
            if (ptrType->hasAddressSpace())
                storageClass = addressSpaceToStorageClass(ptrType->getAddressSpace());
        }
        if (auto systemValInst = maybeEmitSystemVal(param))
        {
            emitSystemVarDecoration(param, systemValInst);
            registerInst(param, systemValInst);
            return systemValInst;
        }
        auto varInst = emitOpVariable(
            getSection(SpvLogicalSectionID::GlobalVariables),
            param,
            param->getDataType(),
            storageClass);
        maybeEmitPointerDecoration(varInst, param);
        maybeEmitWriteOnlyImageDecoration(varInst, param);
        if (layout)
            emitVarLayout(param, varInst, layout);
        emitDecorations(param, getID(varInst));
        return varInst;
    }

    /// Emit a global variable definition.
    SpvInst* emitGlobalVar(IRGlobalVar* globalVar)
    {
        auto layout = getVarLayout(globalVar);
        auto storageClass = SpvStorageClassUniform;
        if (auto ptrType = as<IRPtrTypeBase>(globalVar->getDataType()))
        {
            if (ptrType->hasAddressSpace())
                storageClass = addressSpaceToStorageClass(ptrType->getAddressSpace());
        }
        auto varInst = emitOpVariable(
            getSection(SpvLogicalSectionID::GlobalVariables),
            globalVar,
            globalVar->getDataType(),
            storageClass);
        maybeEmitPointerDecoration(varInst, globalVar);
        if (layout)
            emitVarLayout(globalVar, varInst, layout);
        emitDecorations(globalVar, getID(varInst));
        return varInst;
    }

    SpvInst* emitBuiltinVar(IRInst* spvAsmBuiltinVar)
    {
        const auto kind = (SpvBuiltIn)(getIntVal(spvAsmBuiltinVar->getOperand(0)));
        IRBuilder builder(spvAsmBuiltinVar);
        builder.setInsertBefore(spvAsmBuiltinVar);
        auto varInst = getBuiltinGlobalVar(
            builder.getPtrType(
                kIROp_PtrType,
                spvAsmBuiltinVar->getDataType(),
                AddressSpace::BuiltinInput),
            kind,
            spvAsmBuiltinVar);
        registerInst(spvAsmBuiltinVar, varInst);
        return varInst;
    }

    String getDebugInfoCommandLineArgumentForEntryPoint(IREntryPointDecoration* entryPointDecor)
    {
        StringBuilder sb;
        sb << "-target spirv ";
        m_targetProgram->getOptionSet().writeCommandLineArgs(
            m_targetProgram->getTargetReq()->getSession(),
            sb);
        sb << " -stage " << getStageName(entryPointDecor->getProfile().getStage());
        if (auto entryPointName = as<IRStringLit>(getName(entryPointDecor->getParent())))
        {
            sb << " -entry " << entryPointName->getStringSlice();
        }
        sb << " -g2";
        return sb.produceString();
    }

    /// Emit the given `irFunc` to SPIR-V
    SpvInst* emitFunc(IRFunc* irFunc)
    {
        // [2.4: Logical Layout of a Module]
        //
        // > All function declarations ("declarations" are functions
        // > without a body; there is no forward declaration to a
        // > function with a body).
        // > ...
        // > All function definitions (functions with a body).
        //
        // We need to treat functions differently based
        // on whether they have a body or not, since these
        // are encoded differently (and to different sections).
        //
        if (isDefinition(irFunc))
        {
            return emitFuncDefinition(irFunc);
        }
        else
        {
            return emitFuncDeclaration(irFunc);
        }
    }

    /// Emit a declaration for the given `irFunc`
    SpvInst* emitFuncDeclaration(IRFunc* irFunc)
    {
        // [2.4: Logical Layout of a Module]
        //
        // > All function declarations("declarations" are functions without a
        //   body; there is no forward declaration to a function with a body).
        //
        auto section = getSection(SpvLogicalSectionID::FunctionDeclarations);

        // > A function declaration is as follows.
        // > * Function declaration, using OpFunction.
        // > * Function parameter declarations, using OpFunctionParameter.
        // > * Function end, using OpFunctionEnd.
        //

        // [3.24. Function Control]
        //
        // TODO: We should eventually support emitting the "function control"
        // mask to include inline and other hint bits based on decorations
        // set on `irFunc`.
        //
        SpvFunctionControlMask spvFunctionControl = SpvFunctionControlMaskNone;

        // [3.32.9. Function Instructions]
        //
        // > OpFunction
        //
        // Note that the type <id> of a SPIR-V function uses the
        // *result* type of the function, while the actual function
        // type is given as a later operand. Slan IR instead uses
        // the type of a function instruction store, you know, its *type*.
        //
        SpvInst* spvFunc = emitOpFunction(
            section,
            irFunc,
            irFunc->getDataType()->getResultType(),
            spvFunctionControl,
            irFunc->getDataType());

        // > OpFunctionParameter
        //
        // Though parameters always belong to blocks in Slang, there are no
        // blocks in a function declaration, so we will emit the parameters
        // as derived from the function's type.
        //
        auto funcType = irFunc->getDataType();
        auto paramCount = funcType->getParamCount();
        for (UInt pp = 0; pp < paramCount; ++pp)
        {
            auto paramType = funcType->getParamType(pp);
            SpvInst* spvParam = emitOpFunctionParameter(spvFunc, nullptr, paramType);
            maybeEmitPointerDecoration(spvParam, paramType, false, kIROp_Param);
        }

        // [3.32.9. Function Instructions]
        //
        // > OpFunctionEnd
        //
        // In the SPIR-V encoding a function is logically the parent of any
        // instructions up to a matching `OpFunctionEnd`. In our intermediate
        // structure we will make the `OpFunctionEnd` be the last child of
        // the `OpFunction`.
        //
        emitOpFunctionEnd(spvFunc, nullptr);

        // We will emit any decorations pertinent to the function to the
        // appropriate section of the module.
        //
        emitDecorations(irFunc, getID(spvFunc));

        return spvFunc;
    }

    /// Emit a SPIR-V function definition for the Slang IR function `irFunc`.
    SpvInst* emitFuncDefinition(IRFunc* irFunc)
    {
        if (!irFunc->getFirstBlock())
            m_sink->diagnose(irFunc, Diagnostics::noBlocksOrIntrinsic, "spirv");

        // [2.4: Logical Layout of a Module]
        //
        // > All function definitions (functions with a body).
        //
        auto section = getSection(SpvLogicalSectionID::FunctionDefinitions);
        //
        // > A function definition is as follows.
        // > * Function definition, using OpFunction.
        // > * Function parameter declarations, using OpFunctionParameter.
        // > * Block
        // > * Block
        // > * ...
        // > * Function end, using OpFunctionEnd.
        //

        // [3.24. Function Control]
        //
        // TODO: We should eventually support emitting the "function control"
        // mask to include inline and other hint bits based on decorations
        // set on `irFunc`.
        //
        SpvFunctionControlMask spvFunctionControl = SpvFunctionControlMaskNone;

        // [3.32.9. Function Instructions]
        //
        // > OpFunction
        //
        // Note that the type <id> of a SPIR-V function uses the
        // *result* type of the function, while the actual function
        // type is given as a later operand. Slan IR instead uses
        // the type of a function instruction store, you know, its *type*.
        //
        SpvInst* spvFunc = emitOpFunction(
            section,
            irFunc,
            irFunc->getDataType()->getResultType(),
            spvFunctionControl,
            irFunc->getDataType());

        // > OpFunctionParameter
        //
        // Unlike Slang, where parameters always belong to blocks,
        // the parameters of a SPIR-V function must appear as direct
        // children of the function instruction, and before any basic blocks.
        //
        for (auto irParam : irFunc->getParams())
        {
            emitParam(spvFunc, irParam);
        }

        // [3.32.17. Control-Flow Instructions]
        //
        // > OpLabel
        //
        // A Slang `IRBlock` corresponds to a SPIR-V `OpLabel`:
        // each represents a basic block in the control flow
        // graph of a parent function.
        //
        // We will allocate SPIR-V instructions to represent
        // all of the blocks in a function before we emit
        // body instructions into any of them. We do this
        // because it is possible for one block to make
        // forward reference to another (wheras that is
        // not possible for ordinary instructions within
        // the blocks in the Slang IR)
        //
        SpvInst* funcDebugScope = nullptr;
        for (auto irBlock : irFunc->getBlocks())
        {
            auto spvBlock = emitOpLabel(spvFunc, irBlock);
            if (irBlock == irFunc->getFirstBlock())
            {
                // OpVariable and OpDebugVariable
                // All variables used in the function must be declared before anything else.
                for (auto block : irFunc->getBlocks())
                {
                    for (auto inst : block->getChildren())
                    {
                        switch (inst->getOp())
                        {
                        case kIROp_Var:
                            emitLocalInst(spvBlock, inst);
                            break;
                        case kIROp_DebugVar:
                            // Declare an ordinary local variable for debugDeclare association
                            // of a debug variable. This variable is what we will actually write
                            // values to upon a `kIROp_DebugValue` inst.
                            emitDebugVarBackingLocalVarDeclaration(spvBlock, as<IRDebugVar>(inst));
                            break;
                        }
                    }
                }
                // DebugInfo.
                funcDebugScope = emitDebugFunction(spvBlock, spvFunc, irFunc);
            }
            if (funcDebugScope)
            {
                emitOpDebugScope(
                    spvBlock,
                    nullptr,
                    m_voidType,
                    getNonSemanticDebugInfoExtInst(),
                    funcDebugScope);
            }
            // In addition to normal basic blocks,
            // all loops gets a header block.
            for (auto irInst : irBlock->getChildren())
            {
                if (irInst->getOp() == kIROp_loop)
                {
                    emitOpLabel(spvFunc, irInst);
                }
            }
        }

        if (funcDebugScope)
        {
            if (auto entryPointDecor = irFunc->findDecoration<IREntryPointDecoration>())
            {
                if (auto debugScope = findDebugScope(irFunc->getModule()->getModuleInst()))
                {
                    IRBuilder builder(irFunc);
                    String cmdArgs = getDebugInfoCommandLineArgumentForEntryPoint(entryPointDecor);
                    emitOpDebugEntryPoint(
                        getSection(SpvLogicalSectionID::ConstantsAndTypes),
                        m_voidType,
                        getNonSemanticDebugInfoExtInst(),
                        funcDebugScope,
                        debugScope,
                        builder.getStringValue(toSlice("slangc")),
                        builder.getStringValue(cmdArgs.getUnownedSlice()));
                }
            }
        }

        // Once all the basic blocks have had instructions allocated
        // for them, we go through and fill them in with their bodies.
        //
        // Each loop inst results in a loop header block.
        // We will defer the emit of the contents in loop header block
        // until all Phi insts are emitted.
        List<IRLoop*> pendingLoopInsts;
        for (auto irBlock : irFunc->getBlocks())
        {
            // Note: because we already created the block above,
            // we can be sure that it will have been registred.
            //
            SpvInst* spvBlock = nullptr;
            m_mapIRInstToSpvInst.tryGetValue(irBlock, spvBlock);
            SLANG_ASSERT(spvBlock);

            // [3.32.17. Control-Flow Instructions]
            //
            // > OpPhi
            if (irBlock != irFunc->getFirstBlock())
            {
                for (auto irParam : irBlock->getParams())
                {
                    emitPhi(spvBlock, irParam);
                }
            }
            for (auto irInst : irBlock->getOrdinaryInsts())
            {
                // Any instructions local to the block will be emitted as children
                // of the block.
                //
                // Skip vars because they are already emitted.
                if (as<IRVar>(irInst))
                    continue;
                emitLocalInst(spvBlock, irInst);
                if (irInst->getOp() == kIROp_loop)
                    pendingLoopInsts.add(as<IRLoop>(irInst));
                if (irInst->getOp() == kIROp_discard && !shouldEmitDiscardAsDemote())
                {
                    // If we emitted OpKill for discard, we should stop emitting anything
                    // after this inst in the block, because OpKill is a terminator inst.
                    break;
                }
            }
        }

        // Finally, we generate the body of loop header blocks.
        for (auto loopInst : pendingLoopInsts)
        {
            SpvInst* headerBlock = nullptr;
            m_mapIRInstToSpvInst.tryGetValue(loopInst, headerBlock);
            SLANG_ASSERT(headerBlock);
            emitLoopHeaderBlock(loopInst, headerBlock);
        }

        // [3.32.9. Function Instructions]
        //
        // > OpFunctionEnd
        //
        // In the SPIR-V encoding a function is logically the parent of any
        // instructions up to a matching `OpFunctionEnd`. In our intermediate
        // structure we will make the `OpFunctionEnd` be the last child of
        // the `OpFunction`.
        //
        emitOpFunctionEnd(spvFunc, nullptr);

        // We will emit any decorations pertinent to the function to the
        // appropriate section of the module.
        //
        emitDecorations(irFunc, getID(spvFunc));

        return spvFunc;
    }

    /// Check if a block is a loop's target block.
    bool isLoopTargetBlock(IRInst* block, IRInst*& loopInst)
    {
        for (auto use = block->firstUse; use; use = use->nextUse)
        {
            if (use->getUser()->getOp() == kIROp_loop &&
                as<IRLoop>(use->getUser())->getTargetBlock() == block)
            {
                loopInst = use->getUser();
                return true;
            }
        }
        return false;
    }

    bool shouldEmitDiscardAsDemote()
    {
        return (isSpirv16OrLater() || m_useDemoteToHelperInvocationExtension);
    }

    SpvInst* emitMemorySemanticMask(IRInst* memoryOrderInst, IRInst* ptrInst)
    {
        IRBuilder builder(memoryOrderInst);
        auto memoryOrder = (IRMemoryOrder)getIntVal(memoryOrderInst);
        if (memoryOrder == kIRMemoryOrder_Relaxed)
        {
            return emitIntConstant(
                IRIntegerValue{SpvMemorySemanticsMaskNone},
                builder.getUIntType());
        }
        uint32_t memoryClass = 0;
        if (auto ptrType = as<IRPtrTypeBase>(ptrInst->getDataType()))
        {
            if (ptrType->hasAddressSpace())
            {
                switch (ptrType->getAddressSpace())
                {
                case AddressSpace::StorageBuffer:
                case AddressSpace::UserPointer:
                    memoryClass = SpvMemorySemanticsUniformMemoryMask;
                    break;
                case AddressSpace::Image:
                    memoryClass = SpvMemorySemanticsImageMemoryMask;
                    break;
                case AddressSpace::Output:
                    memoryClass = SpvMemorySemanticsOutputMemoryKHRMask;
                    break;
                case AddressSpace::GroupShared:
                    memoryClass = SpvMemorySemanticsWorkgroupMemoryMask;
                    break;
                }
            }
        }
        switch (memoryOrder)
        {
        case kIRMemoryOrder_Acquire:
            return emitIntConstant(
                IRIntegerValue{SpvMemorySemanticsAcquireMask | memoryClass},
                builder.getUIntType());
        case kIRMemoryOrder_Release:
            return emitIntConstant(
                IRIntegerValue{SpvMemorySemanticsReleaseMask | memoryClass},
                builder.getUIntType());
        case kIRMemoryOrder_AcquireRelease:
            return emitIntConstant(
                IRIntegerValue{SpvMemorySemanticsAcquireReleaseMask | memoryClass},
                builder.getUIntType());
        case kIRMemoryOrder_SeqCst:
            return emitIntConstant(
                IRIntegerValue{SpvMemorySemanticsSequentiallyConsistentMask | memoryClass},
                builder.getUIntType());
        default:
            SLANG_UNEXPECTED("unhandled memory order");
            UNREACHABLE_RETURN(nullptr);
        }
    }

    SpvOp getSpvAtomicOp(IRInst* atomicInst, bool& outNegateOperand)
    {
        auto typeSelect = [&](SpvOp sop, SpvOp uop, SpvOp fop)
        {
            auto scalarType = getVectorElementType(atomicInst->getDataType());
            if (isIntegralType(scalarType))
            {
                auto intInfo = getIntTypeInfo(scalarType);
                if (intInfo.isSigned)
                    return sop;
                return uop;
            }
            return fop;
        };
        outNegateOperand = false;
        switch (atomicInst->getOp())
        {
        case kIROp_AtomicAdd:
            return typeSelect(SpvOpAtomicIAdd, SpvOpAtomicIAdd, SpvOpAtomicFAddEXT);
        case kIROp_AtomicSub:
            if (isFloatingType(getVectorElementType(atomicInst->getDataType())))
                outNegateOperand = true;
            return typeSelect(SpvOpAtomicISub, SpvOpAtomicISub, SpvOpAtomicFAddEXT);
        case kIROp_AtomicMin:
            return typeSelect(SpvOpAtomicSMin, SpvOpAtomicUMin, SpvOpAtomicFMinEXT);
        case kIROp_AtomicMax:
            return typeSelect(SpvOpAtomicSMax, SpvOpAtomicUMax, SpvOpAtomicFMaxEXT);
        case kIROp_AtomicAnd:
            return SpvOpAtomicAnd;
        case kIROp_AtomicOr:
            return SpvOpAtomicOr;
        case kIROp_AtomicXor:
            return SpvOpAtomicXor;
        default:
            SLANG_UNEXPECTED("unhandled atomic op");
            UNREACHABLE_RETURN(SpvOpNop);
        }
    }

    void ensureAtomicCapability(IRInst* atomicInst, SpvOp op)
    {
        auto typeOp = atomicInst->getDataType()->getOp();
        if (typeOp == kIROp_VoidType)
        {
            auto ptrType = atomicInst->getOperand(0)->getDataType();
            IRBuilder builder(atomicInst);
            if (auto valType = tryGetPointedToType(&builder, ptrType))
            {
                if (auto atomicType = as<IRAtomicType>(valType))
                    valType = atomicType->getElementType();
                typeOp = valType->getOp();
            }
        }
        switch (op)
        {
        case SpvOpAtomicFAddEXT:
            {
                switch (typeOp)
                {
                case kIROp_FloatType:
                    ensureExtensionDeclaration(toSlice("SPV_EXT_shader_atomic_float_add"));
                    requireSPIRVCapability(SpvCapabilityAtomicFloat32AddEXT);
                    break;
                case kIROp_DoubleType:
                    ensureExtensionDeclaration(toSlice("SPV_EXT_shader_atomic_float_add"));
                    requireSPIRVCapability(SpvCapabilityAtomicFloat64AddEXT);
                    break;
                case kIROp_HalfType:
                    ensureExtensionDeclaration(toSlice("SPV_EXT_shader_atomic_float16_add"));
                    requireSPIRVCapability(SpvCapabilityAtomicFloat16AddEXT);
                    break;
                case kIROp_VectorType:
                    if (as<IRVectorType>(atomicInst->getDataType())->getElementType()->getOp() ==
                        kIROp_HalfType)
                    {
                        ensureExtensionDeclaration(toSlice("VK_NV_shader_atomic_float16_vector"));
                        requireSPIRVCapability(SpvCapabilityAtomicFloat16VectorNV);
                    }
                    break;
                }
            }
            break;
        case SpvOpAtomicFMinEXT:
        case SpvOpAtomicFMaxEXT:
            {
                switch (typeOp)
                {
                case kIROp_FloatType:
                    ensureExtensionDeclaration(toSlice("SPV_EXT_shader_atomic_float_min_max"));
                    requireSPIRVCapability(SpvCapabilityAtomicFloat32MinMaxEXT);
                    break;
                case kIROp_DoubleType:
                    ensureExtensionDeclaration(toSlice("SPV_EXT_shader_atomic_float_min_max"));
                    requireSPIRVCapability(SpvCapabilityAtomicFloat64MinMaxEXT);
                    break;
                case kIROp_HalfType:
                    ensureExtensionDeclaration(toSlice("SPV_EXT_shader_atomic_float_min_max"));
                    requireSPIRVCapability(SpvCapabilityAtomicFloat16MinMaxEXT);
                    break;
                case kIROp_VectorType:
                    if (as<IRVectorType>(atomicInst->getDataType())->getElementType()->getOp() ==
                        kIROp_HalfType)
                    {
                        ensureExtensionDeclaration(toSlice("VK_NV_shader_atomic_float16_vector"));
                        requireSPIRVCapability(SpvCapabilityAtomicFloat16VectorNV);
                    }
                    break;
                }
            }
            break;
        }
        switch (typeOp)
        {
        case kIROp_UInt64Type:
        case kIROp_Int64Type:
            requireSPIRVCapability(SpvCapabilityInt64Atomics);
            break;
        }
    }

    SpvInst* emitDebugVarDeclaration(SpvInstParent* parent, IRDebugVar* debugVar)
    {
        // For every DebugVar, we will declare:
        // - An OpDebugLocalVariable `spvDebugLocalVar`
        // - An OpVariable `actualHelperVar`
        // - An OpDebugDeclare to associate `spvDebugLocalVar` with `actualHelperVar`
        // The `actualHelperVar` is used to update the actual value of the variable
        // at each kIROp_DebugValue instruction.
        //
        auto scope = findDebugScope(debugVar);
        if (!scope)
            return nullptr;

        bool hasBackingVar = m_mapIRInstToSpvInst.containsKey(debugVar);

        IRBuilder builder(debugVar);
        builder.setInsertBefore(debugVar);

        auto name = getName(debugVar);
        auto varType = tryGetPointedToType(&builder, debugVar->getDataType());
        auto debugType = emitDebugType(varType);

        auto spvDebugLocalVar = emitOpDebugLocalVariable(
            getSection(SpvLogicalSectionID::ConstantsAndTypes),
            hasBackingVar ? nullptr : debugVar,
            m_voidType,
            getNonSemanticDebugInfoExtInst(),
            name,
            debugType,
            debugVar->getSource(),
            debugVar->getLine(),
            debugVar->getCol(),
            scope,
            builder.getIntValue(builder.getUIntType(), 0),
            debugVar->getArgIndex());

        if (hasBackingVar)
        {
            emitOpDebugDeclare(
                parent,
                nullptr,
                m_voidType,
                getNonSemanticDebugInfoExtInst(),
                spvDebugLocalVar,
                debugVar,
                getDwarfExpr(),
                List<SpvInst*>());
        }

        return spvDebugLocalVar;
    }

    bool isLegalType(IRInst* type)
    {
        switch (type->getOp())
        {
        case kIROp_UnsizedArrayType:
            return false;
        case kIROp_ArrayType:
            return isLegalType(as<IRArrayType>(type)->getElementType());
        case kIROp_VectorType:
        case kIROp_StructType:
        case kIROp_MatrixType:
            return true;
        case kIROp_PtrType:
            return as<IRPtrTypeBase>(type)->getAddressSpace() == AddressSpace::UserPointer;
        default:
            if (as<IRBasicType>(type))
                return true;
            return false;
        }
    }

    SpvInst* emitDebugVarBackingLocalVarDeclaration(SpvInstParent* parent, IRDebugVar* debugVar)
    {
        auto scope = findDebugScope(debugVar);
        if (!scope)
            return nullptr;

        IRBuilder builder(debugVar);
        builder.setInsertBefore(debugVar);
        auto varType = tryGetPointedToType(&builder, debugVar->getDataType());

        if (!isLegalType(varType))
            return nullptr;

        IRSizeAndAlignment sizeAlignment;
        getNaturalSizeAndAlignment(this->m_targetRequest->getOptionSet(), varType, &sizeAlignment);
        if (sizeAlignment.size != IRSizeAndAlignment::kIndeterminateSize)
        {
            auto debugVarPtrType = builder.getPtrType(varType, AddressSpace::Function);
            auto actualHelperVar =
                emitOpVariable(parent, debugVar, debugVarPtrType, SpvStorageClassFunction);
            maybeEmitPointerDecoration(actualHelperVar, debugVar);
            return actualHelperVar;
        }
        return nullptr;
    }

    SpvInst* emitMakeUInt64(SpvInstParent* parent, IRInst* inst)
    {
        IRBuilder builder(inst);
        builder.setInsertBefore(inst);
        auto vec = emitOpCompositeConstruct(
            parent,
            nullptr,
            builder.getVectorType(builder.getUIntType(), 2),
            inst->getOperand(0),
            inst->getOperand(1));
        return emitOpBitcast(parent, inst, inst->getDataType(), vec);
    }

    bool isAtomicableAddressSpace(IRInst* type)
    {
        auto ptrType = as<IRPtrTypeBase>(type);
        if (!ptrType)
            return false;
        switch (ptrType->getAddressSpace())
        {
        case AddressSpace::Global:
        case AddressSpace::StorageBuffer:
        case AddressSpace::UserPointer:
        case AddressSpace::GroupShared:
        case AddressSpace::Image:
        case AddressSpace::TaskPayloadWorkgroup:
            return true;
        default:
            return false;
        }
    }

    // The instructions that appear inside the basic blocks of
    // functions are what we will call "local" instructions.
    //
    // When emitting global instructions, we usually have to
    // pick the right logical section to emit them into, while
    // for local instructions they will usually emit into
    // a known parent (the basic block that contains them).

    /// Emit an instruction that is local to the body of the given `parent`.
    SpvInst* emitLocalInst(SpvInstParent* parent, IRInst* inst)
    {
        SpvInst* result = nullptr;
        switch (inst->getOp())
        {
        default:
            {
                if (as<IRSPIRVAsmOperand>(inst))
                    return nullptr;
                String e = "Unhandled local inst in spirv-emit:\n" +
                           dumpIRToString(inst, {IRDumpOptions::Mode::Detailed, 0});
                SLANG_UNIMPLEMENTED_X(e.getBuffer());
            }
        case kIROp_Specialize:
        case kIROp_MissingReturn:
        case kIROp_StaticAssert:
        case kIROp_Unmodified:
            break;
        case kIROp_Var:
            result = emitVar(parent, inst);
            break;
        case kIROp_Call:
            result = emitCall(parent, static_cast<IRCall*>(inst));
            break;
        case kIROp_FieldAddress:
            result = emitFieldAddress(parent, as<IRFieldAddress>(inst));
            break;
        case kIROp_FieldExtract:
            result = emitFieldExtract(parent, as<IRFieldExtract>(inst));
            break;
        case kIROp_GetElementPtr:
            result = emitGetElementPtr(parent, as<IRGetElementPtr>(inst));
            break;
        case kIROp_GetOffsetPtr:
            result = emitGetOffsetPtr(parent, inst);
            break;
        case kIROp_GetElement:
            result = emitGetElement(parent, as<IRGetElement>(inst));
            break;
        case kIROp_MakeStruct:
            result = emitCompositeConstruct(parent, inst);
            break;
        case kIROp_MakeArrayFromElement:
            result = emitMakeArrayFromElement(parent, inst);
            break;
        case kIROp_MakeMatrixFromScalar:
            result = emitMakeMatrixFromScalar(parent, inst);
            break;
        case kIROp_MakeMatrix:
            result = emitMakeMatrix(parent, inst);
            break;
        case kIROp_Load:
            result = emitLoad(parent, as<IRLoad>(inst));
            break;
        case kIROp_Store:
            result = emitStore(parent, as<IRStore>(inst));
            break;
        case kIROp_SwizzledStore:
            result = emitSwizzledStore(parent, as<IRSwizzledStore>(inst));
            break;
        case kIROp_swizzleSet:
            result = emitSwizzleSet(parent, as<IRSwizzleSet>(inst));
            break;
        case kIROp_RWStructuredBufferGetElementPtr:
            result = emitStructuredBufferGetElementPtr(parent, inst);
            break;
        case kIROp_StructuredBufferGetDimensions:
            result = emitStructuredBufferGetDimensions(parent, inst);
            break;
        case kIROp_GetStructuredBufferPtr:
        case kIROp_GetUntypedBufferPtr:
            result = emitGetBufferPtr(parent, inst);
            break;
        case kIROp_swizzle:
            result = emitSwizzle(parent, as<IRSwizzle>(inst));
            break;
        case kIROp_IntCast:
            result = emitIntCast(parent, as<IRIntCast>(inst));
            break;
        case kIROp_FloatCast:
            result = emitFloatCast(parent, as<IRFloatCast>(inst));
            break;
        case kIROp_CastIntToFloat:
            result = emitIntToFloatCast(parent, as<IRCastIntToFloat>(inst));
            break;
        case kIROp_CastFloatToInt:
            result = emitFloatToIntCast(parent, as<IRCastFloatToInt>(inst));
            break;
        case kIROp_CastPtrToInt:
            result = emitCastPtrToInt(parent, inst);
            break;
        case kIROp_CastPtrToBool:
            result = emitCastPtrToBool(parent, inst);
            break;
        case kIROp_CastIntToPtr:
            result = emitCastIntToPtr(parent, inst);
            break;
        case kIROp_PtrCast:
        case kIROp_BitCast:
            result = emitOpBitcast(parent, inst, inst->getDataType(), inst->getOperand(0));
            break;
        case kIROp_BitfieldExtract:
            result = emitBitfieldExtract(parent, inst);
            break;
        case kIROp_BitfieldInsert:
            result = emitBitfieldInsert(parent, inst);
            break;
        case kIROp_MakeUInt64:
            result = emitMakeUInt64(parent, inst);
            break;
        case kIROp_Add:
        case kIROp_Sub:
        case kIROp_Mul:
        case kIROp_Div:
        case kIROp_IRem:
        case kIROp_FRem:
        case kIROp_Neg:
        case kIROp_Not:
        case kIROp_And:
        case kIROp_Or:
        case kIROp_BitNot:
        case kIROp_BitAnd:
        case kIROp_BitOr:
        case kIROp_BitXor:
        case kIROp_Less:
        case kIROp_Leq:
        case kIROp_Eql:
        case kIROp_Neq:
        case kIROp_Greater:
        case kIROp_Geq:
        case kIROp_Rsh:
        case kIROp_Lsh:
            result = emitArithmetic(parent, inst);
            break;
        case kIROp_CastDescriptorHandleToUInt2:
        case kIROp_CastUInt2ToDescriptorHandle:
        case kIROp_GlobalValueRef:
            {
                auto inner = ensureInst(inst->getOperand(0));
                registerInst(inst, inner);
                result = inner;
                break;
            }
        case kIROp_GetVulkanRayTracingPayloadLocation:
            {
                IRInst* location = getVulkanPayloadLocation(inst->getOperand(0));
                if (!location)
                {
                    SLANG_DIAGNOSE_UNEXPECTED(m_sink, inst, "no payload location assigned.");
                    IRBuilder builder(inst);
                    builder.setInsertBefore(inst);
                    location = builder.getIntValue(builder.getIntType(), 0);
                }
                auto inner = ensureInst(location);
                registerInst(inst, inner);
                result = inner;
                break;
            }
        case kIROp_RequireComputeDerivative:
            {
                auto parentFunc = getParentFunc(inst);

                HashSet<IRFunc*>* entryPointsUsingInst =
                    getReferencingEntryPoints(m_referencingEntryPoints, parentFunc);
                for (IRFunc* entryPoint : *entryPointsUsingInst)
                {
                    bool isQuad = true;
                    IREntryPointDecoration* entryPointDecor = nullptr;
                    for (auto dec : entryPoint->getDecorations())
                    {
                        if (auto maybeEntryPointDecor = as<IREntryPointDecoration>(dec))
                            entryPointDecor = maybeEntryPointDecor;
                        if (as<IRDerivativeGroupLinearDecoration>(dec))
                            isQuad = false;
                    }
                    if (!entryPointDecor ||
                        entryPointDecor->getProfile().getStage() != Stage::Compute)
                        continue;

                    ensureExtensionDeclaration(
                        UnownedStringSlice("SPV_KHR_compute_shader_derivatives"));
                    auto numThreadsDecor =
                        entryPointDecor->findDecoration<IRNumThreadsDecoration>();
                    if (isQuad)
                    {
                        verifyComputeDerivativeGroupModifiers(
                            this->m_sink,
                            inst->sourceLoc,
                            true,
                            false,
                            numThreadsDecor);
                        requireSPIRVExecutionMode(
                            nullptr,
                            getIRInstSpvID(entryPoint),
                            SpvExecutionModeDerivativeGroupQuadsKHR);
                        requireSPIRVCapability(SpvCapabilityComputeDerivativeGroupQuadsKHR);
                    }
                    else
                    {
                        verifyComputeDerivativeGroupModifiers(
                            this->m_sink,
                            inst->sourceLoc,
                            false,
                            true,
                            numThreadsDecor);
                        requireSPIRVExecutionMode(
                            nullptr,
                            getIRInstSpvID(entryPoint),
                            SpvExecutionModeDerivativeGroupLinearKHR);
                        requireSPIRVCapability(SpvCapabilityComputeDerivativeGroupLinearKHR);
                    }
                }

                break;
            }

        case kIROp_RequireMaximallyReconverges:
            ensureExtensionDeclaration(UnownedStringSlice("SPV_KHR_maximal_reconvergence"));
            requireSPIRVExecutionModeOnReferencingEntryPoints(
                nullptr,
                getParentFunc(inst),
                SpvExecutionModeMaximallyReconvergesKHR);
            break;
        case kIROp_RequireQuadDerivatives:
            ensureExtensionDeclaration(UnownedStringSlice("SPV_KHR_quad_control"));
            requireSPIRVCapability(SpvCapabilityQuadControlKHR);
            requireSPIRVExecutionModeOnReferencingEntryPoints(
                nullptr,
                getParentFunc(inst),
                SpvExecutionModeQuadDerivativesKHR);
            break;
        case kIROp_Return:
            if (as<IRReturn>(inst)->getVal()->getOp() == kIROp_VoidLit)
                result = emitOpReturn(parent, inst);
            else
                result = emitOpReturnValue(parent, inst, as<IRReturn>(inst)->getVal());
            break;
        case kIROp_discard:
            if (shouldEmitDiscardAsDemote())
            {
                ensureExtensionDeclarationBeforeSpv16(
                    toSlice("SPV_EXT_demote_to_helper_invocation"));
                requireSPIRVCapability(SpvCapabilityDemoteToHelperInvocation);
                result = emitOpDemoteToHelperInvocation(parent, inst);
            }
            else
            {
                result = emitOpKill(parent, inst);
            }
            break;
        case kIROp_BeginFragmentShaderInterlock:
            ensureExtensionDeclaration(UnownedStringSlice("SPV_EXT_fragment_shader_interlock"));
            requireSPIRVCapability(SpvCapabilityFragmentShaderPixelInterlockEXT);
            requireSPIRVExecutionModeOnReferencingEntryPoints(
                nullptr,
                getParentFunc(inst),
                SpvExecutionModePixelInterlockOrderedEXT);
            result = emitOpBeginInvocationInterlockEXT(parent, inst);
            break;
        case kIROp_EndFragmentShaderInterlock:
            result = emitOpEndInvocationInterlockEXT(parent, inst);
            break;
        case kIROp_unconditionalBranch:
            {
                // If we are jumping to the main block of a loop,
                // emit a branch to the loop header instead.
                // The SPV id of the resulting loop header block is associated with the loop inst.
                auto targetBlock = as<IRUnconditionalBranch>(inst)->getTargetBlock();
                IRInst* loopInst = nullptr;
                if (isLoopTargetBlock(targetBlock, loopInst))
                    return emitOpBranch(parent, inst, getIRInstSpvID(loopInst));
                // Otherwise, emit a normal branch inst into the target block.
                result = emitOpBranch(parent, inst, getIRInstSpvID(targetBlock));
                break;
            }
        case kIROp_loop:
            {
                // Return loop header block in its own block.
                auto blockId = getIRInstSpvID(inst);
                SpvInst* block = nullptr;
                m_mapIRInstToSpvInst.tryGetValue(inst, block);
                SLANG_ASSERT(block);

                // Emit a jump to the loop header block.
                // Note: the body of the loop header block is emitted
                // after everything else to ensure Phi instructions (which come
                // from the actual loop target block) are emitted first.
                emitOpBranch(parent, nullptr, blockId);

                result = block;
                break;
            }
        case kIROp_ifElse:
            {
                auto ifelseInst = as<IRIfElse>(inst);
                auto afterBlockID = getIRInstSpvID(ifelseInst->getAfterBlock());
                emitOpSelectionMerge(parent, nullptr, afterBlockID, SpvSelectionControlMaskNone);
                auto falseLabel = ifelseInst->getFalseBlock();
                result = emitOpBranchConditional(
                    parent,
                    inst,
                    ifelseInst->getCondition(),
                    ifelseInst->getTrueBlock(),
                    falseLabel ? getID(ensureInst(falseLabel)) : afterBlockID,
                    makeArray<SpvLiteralInteger>());
                break;
            }
        case kIROp_Switch:
            {
                auto switchInst = as<IRSwitch>(inst);
                auto mergeBlockID = getIRInstSpvID(switchInst->getBreakLabel());
                emitOpSelectionMerge(parent, nullptr, mergeBlockID, SpvSelectionControlMaskNone);
                result = emitInstCustomOperandFunc(
                    parent,
                    inst,
                    SpvOpSwitch,
                    [&]()
                    {
                        emitOperand(switchInst->getCondition());
                        auto defaultLabel = switchInst->getDefaultLabel();
                        emitOperand(defaultLabel ? getID(ensureInst(defaultLabel)) : mergeBlockID);
                        for (UInt c = 0; c < switchInst->getCaseCount(); c++)
                        {
                            auto value = switchInst->getCaseValue(c);
                            auto intLit = as<IRIntLit>(value);
                            SLANG_ASSERT(intLit);
                            emitOperand((SpvWord)intLit->getValue());
                            auto caseLabel = switchInst->getCaseLabel(c);
                            emitOperand(caseLabel ? getID(ensureInst(caseLabel)) : mergeBlockID);
                        }
                    });
                break;
            }
        case kIROp_Unreachable:
            result = emitOpUnreachable(parent, inst);
            break;
        case kIROp_conditionalBranch:
            SLANG_UNEXPECTED("Unstructured branching is not supported by SPIRV.");
            break;
        case kIROp_MakeVector:
            result = emitConstruct(parent, inst);
            break;
        case kIROp_MakeVectorFromScalar:
            {
                const auto scalar = inst->getOperand(0);
                const auto vecTy = as<IRVectorType>(inst->getDataType());
                SLANG_ASSERT(vecTy);
                const auto numElems = as<IRIntLit>(vecTy->getElementCount());
                SLANG_ASSERT(numElems);
                result = emitSplat(parent, inst, scalar, numElems->getValue());
            }
            break;
        case kIROp_MakeCoopVector:
            result = emitConstruct(parent, inst);
            break;
        case kIROp_MakeArray:
            result = emitConstruct(parent, inst);
            break;
        case kIROp_Select:
            result = emitInst(
                parent,
                inst,
                SpvOpSelect,
                inst->getFullType(),
                kResultID,
                OperandsOf(inst));
            break;
        case kIROp_DebugLine:
            result = emitDebugLine(parent, as<IRDebugLine>(inst));
            break;
        case kIROp_DebugVar:
            result = emitDebugVarDeclaration(parent, as<IRDebugVar>(inst));
            break;
        case kIROp_DebugValue:
            result = emitDebugValue(parent, as<IRDebugValue>(inst));
            break;
        case kIROp_GetStringHash:
            result = emitGetStringHash(inst);
            break;
        case kIROp_undefined:
            result = emitOpUndef(parent, inst, inst->getDataType());
            break;
        case kIROp_SPIRVAsm:
            result = emitSPIRVAsm(parent, as<IRSPIRVAsm>(inst));
            break;
        case kIROp_ImageLoad:
            result = emitImageLoad(parent, as<IRImageLoad>(inst));
            break;
        case kIROp_ImageStore:
            result = emitImageStore(parent, as<IRImageStore>(inst));
            break;
        case kIROp_ImageSubscript:
            result = emitImageSubscript(parent, as<IRImageSubscript>(inst));
            break;
        case kIROp_AtomicInc:
            {
                IRBuilder builder{inst};
                const auto memoryScope =
                    emitIntConstant(IRIntegerValue{SpvScopeDevice}, builder.getUIntType());
                const auto memorySemantics =
                    emitMemorySemanticMask(inst->getOperand(1), inst->getOperand(0));
                result = emitOpAtomicIIncrement(
                    parent,
                    inst,
                    inst->getFullType(),
                    inst->getOperand(0),
                    memoryScope,
                    memorySemantics);
                ensureAtomicCapability(inst, SpvOpAtomicIIncrement);
            }
            break;
        case kIROp_AtomicDec:
            {
                IRBuilder builder{inst};
                const auto memoryScope =
                    emitIntConstant(IRIntegerValue{SpvScopeDevice}, builder.getUIntType());
                const auto memorySemantics =
                    emitMemorySemanticMask(inst->getOperand(1), inst->getOperand(0));
                result = emitOpAtomicIDecrement(
                    parent,
                    inst,
                    inst->getFullType(),
                    inst->getOperand(0),
                    memoryScope,
                    memorySemantics);
                ensureAtomicCapability(inst, SpvOpAtomicIDecrement);
            }
            break;
        case kIROp_AtomicLoad:
            {
                IRBuilder builder{inst};
                if (isAtomicableAddressSpace(inst->getOperand(0)->getDataType()))
                {
                    const auto memoryScope =
                        emitIntConstant(IRIntegerValue{SpvScopeDevice}, builder.getUIntType());
                    const auto memorySemantics =
                        emitMemorySemanticMask(inst->getOperand(1), inst->getOperand(0));
                    result = emitOpAtomicLoad(
                        parent,
                        inst,
                        inst->getFullType(),
                        inst->getOperand(0),
                        memoryScope,
                        memorySemantics);
                    ensureAtomicCapability(inst, SpvOpAtomicLoad);
                }
                else
                {
                    result = emitOpLoad(parent, inst, inst->getFullType(), inst->getOperand(0));
                }
            }
            break;
        case kIROp_AtomicStore:
            {
                IRBuilder builder{inst};
                if (isAtomicableAddressSpace(inst->getOperand(0)->getDataType()))
                {
                    const auto memoryScope =
                        emitIntConstant(IRIntegerValue{SpvScopeDevice}, builder.getUIntType());
                    const auto memorySemantics =
                        emitMemorySemanticMask(inst->getOperand(2), inst->getOperand(0));
                    result = emitOpAtomicStore(
                        parent,
                        inst,
                        inst->getOperand(0),
                        memoryScope,
                        memorySemantics,
                        inst->getOperand(1));
                    ensureAtomicCapability(inst, SpvOpAtomicStore);
                }
                else
                {
                    result = emitOpStore(parent, inst, inst->getOperand(0), inst->getOperand(1));
                }
            }
            break;
        case kIROp_AtomicExchange:
            {
                IRBuilder builder{inst};
                if (isAtomicableAddressSpace(inst->getOperand(0)->getDataType()))
                {
                    const auto memoryScope =
                        emitIntConstant(IRIntegerValue{SpvScopeDevice}, builder.getUIntType());
                    const auto memorySemantics =
                        emitMemorySemanticMask(inst->getOperand(2), inst->getOperand(0));
                    result = emitOpAtomicExchange(
                        parent,
                        inst,
                        inst->getFullType(),
                        inst->getOperand(0),
                        memoryScope,
                        memorySemantics,
                        inst->getOperand(1));
                    ensureAtomicCapability(inst, SpvOpAtomicExchange);
                }
                else
                {
                    result = emitOpStore(parent, inst, inst->getOperand(0), inst->getOperand(1));
                }
            }
            break;
        case kIROp_AtomicCompareExchange:
            {
                IRBuilder builder{inst};
                const auto memoryScope =
                    emitIntConstant(IRIntegerValue{SpvScopeDevice}, builder.getUIntType());
                const auto memorySemanticsEqual =
                    emitMemorySemanticMask(inst->getOperand(3), inst->getOperand(0));
                const auto memorySemanticsUnequal =
                    emitMemorySemanticMask(inst->getOperand(4), inst->getOperand(0));
                result = emitOpAtomicCompareExchange(
                    parent,
                    inst,
                    inst->getFullType(),
                    inst->getOperand(0),
                    memoryScope,
                    memorySemanticsEqual,
                    memorySemanticsUnequal,
                    inst->getOperand(2),
                    inst->getOperand(1));
                ensureAtomicCapability(inst, SpvOpAtomicCompareExchange);
            }
            break;
        case kIROp_AtomicAdd:
        case kIROp_AtomicSub:
        case kIROp_AtomicMax:
        case kIROp_AtomicMin:
        case kIROp_AtomicAnd:
        case kIROp_AtomicOr:
        case kIROp_AtomicXor:
            {
                IRBuilder builder{inst};
                const auto memoryScope =
                    emitIntConstant(IRIntegerValue{SpvScopeDevice}, builder.getUIntType());
                const auto memorySemantics =
                    emitMemorySemanticMask(inst->getOperand(2), inst->getOperand(0));
                bool negateOperand = false;
                auto spvOp = getSpvAtomicOp(inst, negateOperand);
                auto operand = inst->getOperand(1);
                if (negateOperand)
                {
                    builder.setInsertBefore(inst);
                    auto negatedOperand = builder.emitNeg(inst->getDataType(), operand);
                    operand = negatedOperand;
                }
                result = emitOpAtomicOp(
                    parent,
                    inst,
                    spvOp,
                    inst->getFullType(),
                    inst->getOperand(0),
                    memoryScope,
                    memorySemantics,
                    operand);
                ensureAtomicCapability(inst, spvOp);
            }
            break;
        case kIROp_ControlBarrier:
            {
                IRBuilder builder{inst};
                const auto executionScope =
                    emitIntConstant(IRIntegerValue{SpvScopeWorkgroup}, builder.getUIntType());
                const auto memoryScope =
                    emitIntConstant(IRIntegerValue{SpvScopeInvocation}, builder.getUIntType());
                const auto memorySemantics = emitIntConstant(
                    IRIntegerValue{SpvMemorySemanticsMaskNone},
                    builder.getUIntType());
                emitInst(
                    parent,
                    inst,
                    SpvOpControlBarrier,
                    executionScope,
                    memoryScope,
                    memorySemantics);
            }
            break;
        case kIROp_Printf:
            {
                List<IRInst*> operands;
                operands.add(inst->getOperand(0));
                if (inst->getOperandCount() == 2)
                {
                    auto operand = inst->getOperand(1);
                    if (auto makeStruct = as<IRMakeStruct>(operand))
                    {
                        // Flatten the tuple resulting from the variadic pack.
                        for (UInt bb = 0; bb < makeStruct->getOperandCount(); ++bb)
                        {
                            operands.add(makeStruct->getOperand(bb));
                        }
                    }
                }
                ensureExtensionDeclaration(toSlice("SPV_KHR_non_semantic_info"));
                result = emitInst(
                    parent,
                    inst,
                    SpvOpExtInst,
                    inst->getFullType(),
                    kResultID,
                    getNonSemanticDebugPrintfExtInst(),
                    SpvLiteralInteger::from32(1),
                    operands.getArrayView());
            }
            break;
        }
        if (result)
            emitDecorations(inst, getID(result));
        return result;
    }

    SpvInst* emitImageLoad(SpvInstParent* parent, IRImageLoad* load)
    {
        if (load->hasAuxCoord1())
        {
            return emitInst(
                parent,
                load,
                SpvOpImageRead,
                load->getDataType(),
                kResultID,
                load->getImage(),
                load->getCoord(),
                SpvImageOperandsSampleMask,
                load->getAuxCoord1());
        }
        else
        {
            return emitInst(
                parent,
                load,
                SpvOpImageRead,
                load->getDataType(),
                kResultID,
                load->getImage(),
                load->getCoord());
        }
    }

    SpvInst* emitImageStore(SpvInstParent* parent, IRImageStore* store)
    {
        if (store->hasAuxCoord1())
        {
            return emitInst(
                parent,
                store,
                SpvOpImageWrite,
                store->getImage(),
                store->getCoord(),
                store->getValue(),
                SpvImageOperandsSampleMask,
                store->getAuxCoord1());
        }
        else
        {
            return emitInst(
                parent,
                store,
                SpvOpImageWrite,
                store->getImage(),
                store->getCoord(),
                store->getValue());
        }
    }

    SpvInst* emitImageSubscript(SpvInstParent* parent, IRImageSubscript* subscript)
    {
        IRBuilder builder(subscript);
        builder.setInsertBefore(subscript);
        return emitInst(
            parent,
            subscript,
            SpvOpImageTexelPointer,
            subscript->getDataType(),
            kResultID,
            subscript->getImage(),
            subscript->getCoord(),
            builder.getIntValue(builder.getIntType(), 0));
    }

    SpvInst* emitGetStringHash(IRInst* inst)
    {
        auto getStringHashInst = as<IRGetStringHash>(inst);
        auto stringLit = getStringHashInst->getStringLit();

        if (stringLit)
        {
            auto slice = stringLit->getStringSlice();
            return emitIntConstant(
                getStableHashCode32(slice.begin(), slice.getLength()).hash,
                inst->getDataType());
        }
        else
        {
            // Couldn't handle
            String e = "Unhandled local inst in spirv-emit:\n" +
                       dumpIRToString(inst, {IRDumpOptions::Mode::Detailed, 0});
            SLANG_UNIMPLEMENTED_X(e.getBuffer());
        }
    }

    SpvInst* emitLit(IRInst* inst)
    {
        switch (inst->getOp())
        {
        case kIROp_IntLit:
            {
                auto value = as<IRIntLit>(inst)->getValue();
                return emitIntConstant(value, inst->getDataType(), inst);
            }
        case kIROp_FloatLit:
            {
                const auto value = as<IRConstant>(inst)->value.floatVal;
                const auto type = inst->getDataType();
                return emitFloatConstant(value, type, inst);
            }
        case kIROp_BoolLit:
            {
                SpvInst* spvInst = nullptr;
                if (cast<IRBoolLit>(inst)->getValue())
                {
                    spvInst = emitOpConstantTrue(inst, inst->getDataType());
                }
                else
                {
                    spvInst = emitOpConstantFalse(inst, inst->getDataType());
                }
                m_mapIRInstToSpvInst[inst] = spvInst;
                return spvInst;
            }
        case kIROp_StringLit:
            {
                auto value = as<IRStringLit>(inst)->getStringSlice();
                return emitInst(
                    getSection(SpvLogicalSectionID::DebugStringsAndSource),
                    inst,
                    SpvOpString,
                    kResultID,
                    SpvLiteralBits::fromUnownedStringSlice(value));
            }
        default:
            return nullptr;
        }
    }

    // Both "local" and "global" instructions can have decorations.
    // When we decide to emit an instruction, we typically also want
    // to emit any decoratons that were attached to it that have
    // a SPIR-V equivalent.

    /// Emit appropriate SPIR-V decorations for the given IR `irInst`.
    ///
    /// The given `dstID` should be the `<id>` of the SPIR-V instruction being decorated,
    /// and should correspond to `irInst`.
    ///
    void emitDecorations(IRInst* irInst, SpvWord dstID)
    {
        for (auto decoration : irInst->getDecorations())
        {
            emitDecoration(dstID, decoration);
        }
    }

    SpvExecutionMode getDepthOutputExecutionMode(IRInst* builtinVar)
    {
        SpvExecutionMode result = SpvExecutionModeMax;
        bool isDepthVar = false;

        // If the builtin is an output, we need to emit the DepthReplacing execution mode
        // required by the SPIRV spec even if we are not writing to it.
        bool isOutputVar = false;

        if (auto layout = getVarLayout(builtinVar))
        {
            if (auto systemValueAttr = layout->findAttr<IRSystemValueSemanticAttr>())
            {
                String semanticName = systemValueAttr->getName();
                semanticName = semanticName.toLower();
                if (semanticName == "sv_position")
                {
                    auto importDecor = builtinVar->findDecoration<IRImportDecoration>();
                    if (importDecor->getMangledName() == "gl_FragCoord")
                    {
                        // SV_POSITION is not necessarily a depth output,
                        // if it is not written to, we don't need to emit
                        // the DepthReplacing execution mode.
                        isDepthVar = true;
                        result = SpvExecutionModeDepthReplacing;
                    }
                }
                else if (semanticName == "sv_depth")
                {
                    isDepthVar = true;
                    isOutputVar = true;
                    result = SpvExecutionModeDepthReplacing;
                }
                else if (semanticName == "sv_depthgreaterequal")
                {
                    isDepthVar = true;
                    isOutputVar = true;
                    result = SpvExecutionModeDepthGreater;
                }
                else if (semanticName == "sv_depthlessequal")
                {
                    isDepthVar = true;
                    isOutputVar = true;
                    result = SpvExecutionModeDepthLess;
                }
            }
        }
        if (!isDepthVar)
            return SpvExecutionModeMax;
        if (isOutputVar)
            return result;

        // If the builtin can be both input and output,
        // we need to check if we actually write to it.
        for (auto use = builtinVar->firstUse; use; use = use->nextUse)
        {
            auto user = use->getUser();
            if (user->getOp() == kIROp_Load)
                continue;
            if (as<IRDecoration>(user))
                continue;
            switch (user->getOp())
            {
            case kIROp_SwizzledStore:
            case kIROp_Store:
            case kIROp_Call:
                return result;
            }
        }
        return SpvExecutionModeMax;
    }

    void maybeEmitEntryPointDepthReplacingExecutionMode(
        IRFunc* entryPoint,
        const List<IRInst*>& referencedBuiltinIRVars)
    {
        // Check if the entrypoint uses any depth output builtin variables,
        // if so, we need to emit a DepthReplacing execution mode for the
        // fragment entrypoint.
        SpvExecutionMode mode = SpvExecutionModeMax;
        for (auto globalInst : referencedBuiltinIRVars)
        {
            auto thisMode = getDepthOutputExecutionMode(globalInst);
            if (mode == SpvExecutionModeMax)
                mode = thisMode;
            else if (mode != thisMode)
            {
                mode = SpvExecutionModeDepthReplacing;
                break;
            }
        }
        if (mode == SpvExecutionModeMax)
            return;

        requireSPIRVExecutionMode(nullptr, getIRInstSpvID(entryPoint), mode);
    }

    // Make user type name conform to `SPV_GOOGLE_user_type` spec.
    // https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/GOOGLE/SPV_GOOGLE_user_type.asciidoc
    String legalizeUserTypeName(UnownedStringSlice typeName)
    {
        String result = typeName;
        auto index = typeName.indexOf('<');
        if (index == -1)
            index = typeName.getLength();
        StringBuilder sb;
        sb << String(typeName.head(index)).toLower();
        if (index != typeName.getLength())
        {
            sb << ":";
            sb << typeName.tail(index);
        }
        return sb.produceString();
    }

    /// Emit an appropriate SPIR-V decoration for the given IR `decoration`, if necessary and
    /// possible.
    ///
    /// The given `dstID` should be the `<id>` of the SPIR-V instruction being decorated,
    /// and should correspond to the parent of `decoration` in the Slang IR.
    ///
    void emitDecoration(SpvWord dstID, IRDecoration* decoration)
    {
        // Unlike in the Slang IR, decorations in SPIR-V are not children
        // of the instruction they decorate, and instead are free-standing
        // instructions at global scope, which reference their target
        // instruction by its `<id>`.
        //
        // The `IRDecoration` hierarchy in Slang also maps to several
        // different categories of instruction in SPIR-V, only a subset
        // of which are officialy called "decorations."
        //
        // We will continue to use the Slang terminology here, since
        // this code path is a catch-all for stuff that only needs to
        // be emitted if the owning instruction gets emitted.
        bool isRayTracingObject = false;

        switch (decoration->getOp())
        {
        default:
            break;

        // [3.32.2. Debug Instructions]
        //
        // > OpName
        //
        case kIROp_NameHintDecoration:
            {
                auto section = getSection(SpvLogicalSectionID::DebugNames);
                auto nameHint = cast<IRNameHintDecoration>(decoration);
                // We can't associate this spirv instruction with our
                // irInstruction, our instruction may be a hint on several
                // values, however this decoration is specific to a single
                // dstID.
                emitOpName(section, nullptr, dstID, nameHint->getName());
            }
            break;

        // [3.32.5. Mode-Setting Instructions]
        //
        // > OpEntryPoint
        // > Declare an entry point, its execution model, and its interface.
        //
        case kIROp_EntryPointDecoration:
            {
                auto section = getSection(SpvLogicalSectionID::EntryPoints);
                auto entryPointDecor = cast<IREntryPointDecoration>(decoration);
                auto entryPoint = as<IRFunc>(decoration->getParent());
                auto spvStage = mapStageToExecutionModel(entryPointDecor->getProfile().getStage());
                auto name = entryPointDecor->getName()->getStringSlice();
                List<SpvInst*> params;
                HashSet<SpvInst*> paramsSet;
                List<IRInst*> referencedBuiltinIRVars;
                // `interface` part: reference all global variables that are used by this
                // entrypoint.
                for (auto globalInst : m_irModule->getModuleInst()->getChildren())
                {
                    switch (globalInst->getOp())
                    {
                    case kIROp_GlobalVar:
                    case kIROp_GlobalParam:
                    case kIROp_SPIRVAsmOperandBuiltinVar:
                        {
                            SpvInst* spvGlobalInst;
                            if (m_mapIRInstToSpvInst.tryGetValue(globalInst, spvGlobalInst))
                            {
                                // Is this globalInst referenced by this entry point?
                                auto refSet = m_referencingEntryPoints.tryGetValue(globalInst);
                                if (refSet && refSet->contains(entryPoint))
                                {
                                    if (!isSpirv14OrLater())
                                    {
                                        // Prior to SPIRV 1.4, we can only reference In and Out
                                        // variables in the interface part.
                                        if (auto ptrType =
                                                as<IRPtrTypeBase>(globalInst->getDataType()))
                                        {
                                            auto addrSpace = ptrType->getAddressSpace();
                                            if (addrSpace != AddressSpace::Input &&
                                                addrSpace != AddressSpace::Output &&
                                                addrSpace != AddressSpace::BuiltinInput &&
                                                addrSpace != AddressSpace::BuiltinOutput)
                                                continue;
                                        }
                                    }
                                    referencedBuiltinIRVars.add(globalInst);
                                    // Don't add duplicate vars to the interface list.
                                    bool paramAdded = paramsSet.add(spvGlobalInst);
                                    if (!paramAdded)
                                        continue;

                                    // Don't add a global param to the interface if it is a
                                    // specialization constant.
                                    switch (spvGlobalInst->opcode)
                                    {
                                    case SpvOpSpecConstant:
                                    case SpvOpSpecConstantFalse:
                                    case SpvOpSpecConstantTrue:
                                    case SpvOpSpecConstantComposite:
                                        break;
                                    default:
                                        params.add(spvGlobalInst);
                                        break;
                                    }
                                }
                            }
                            break;
                        }
                    default:
                        break;
                    }
                }
                emitOpEntryPoint(section, decoration, spvStage, dstID, name, params);

                // Stage specific execution mode and capability declarations.
                switch (entryPointDecor->getProfile().getStage())
                {
                case Stage::Fragment:
                    // OpExecutionMode %main OriginUpperLeft
                    requireSPIRVExecutionMode(
                        nullptr,
                        getIRInstSpvID(entryPoint),
                        SpvExecutionModeOriginUpperLeft);
                    maybeEmitEntryPointDepthReplacingExecutionMode(
                        entryPoint,
                        referencedBuiltinIRVars);
                    for (auto decor : entryPoint->getDecorations())
                    {
                        switch (decor->getOp())
                        {
                        case kIROp_EarlyDepthStencilDecoration:
                            requireSPIRVExecutionMode(
                                nullptr,
                                getIRInstSpvID(entryPoint),
                                SpvExecutionModeEarlyFragmentTests);
                            break;
                        default:
                            break;
                        }
                    }
                    break;
                case Stage::Geometry:
                    requireSPIRVCapability(SpvCapabilityGeometry);
                    break;
                case Stage::Miss:
                case Stage::AnyHit:
                case Stage::ClosestHit:
                case Stage::Intersection:
                case Stage::RayGeneration:
                case Stage::Callable:
                    requireSPIRVCapability(SpvCapabilityRayTracingKHR);
                    ensureExtensionDeclaration(UnownedStringSlice("SPV_KHR_ray_tracing"));
                    break;
                case Stage::Mesh:
                case Stage::Amplification:
                    requireSPIRVCapability(SpvCapabilityMeshShadingEXT);
                    ensureExtensionDeclaration(UnownedStringSlice("SPV_EXT_mesh_shader"));
                    break;
                case Stage::Hull:
                case Stage::Domain:
                    {
                        requireSPIRVCapability(SpvCapabilityTessellation);

                        SpvExecutionMode mode = SpvExecutionModeSpacingEqual;
                        if (auto partitioningDecor =
                                entryPoint->findDecoration<IRPartitioningDecoration>())
                        {
                            auto arg = partitioningDecor->getPartitioning()->getStringSlice();
                            if (arg.caseInsensitiveEquals(toSlice("integer")))
                                mode = SpvExecutionModeSpacingEqual;
                            else if (arg.caseInsensitiveEquals(toSlice("fractional_even")))
                                mode = SpvExecutionModeSpacingFractionalEven;
                            else if (arg.caseInsensitiveEquals(toSlice("fractional_odd")))
                                mode = SpvExecutionModeSpacingFractionalOdd;
                            else
                                m_sink->diagnose(
                                    partitioningDecor,
                                    Diagnostics::unknownTessPartitioning,
                                    arg);
                        }
                        requireSPIRVExecutionMode(nullptr, getIRInstSpvID(entryPoint), mode);
                    }
                    break;
                default:
                    break;
                }
            }
            break;
        // > OpExecutionMode

        // [3.6. Execution Mode]: LocalSize
        case kIROp_NumThreadsDecoration:
            {
                auto numThreads = cast<IRNumThreadsDecoration>(decoration);
                if (numThreads->getXSpecConst() || numThreads->getYSpecConst() ||
                    numThreads->getZSpecConst())
                {
                    // If any of the dimensions needs an ID, we need to emit
                    // all dimensions as an ID due to how LocalSizeId works.
                    int32_t ids[3];
                    for (int i = 0; i < 3; ++i)
                        ids[i] = ensureInst(numThreads->getOperand(i))->id;

                    // LocalSizeId is supported from SPIR-V 1.2 onwards without
                    // any extra capabilities.
                    requireSPIRVExecutionMode(
                        decoration,
                        dstID,
                        SpvExecutionModeLocalSizeId,
                        SpvLiteralInteger::from32(int32_t(ids[0])),
                        SpvLiteralInteger::from32(int32_t(ids[1])),
                        SpvLiteralInteger::from32(int32_t(ids[2])));
                }
                else
                {
                    requireSPIRVExecutionMode(
                        decoration,
                        dstID,
                        SpvExecutionModeLocalSize,
                        SpvLiteralInteger::from32(int32_t(numThreads->getX()->getValue())),
                        SpvLiteralInteger::from32(int32_t(numThreads->getY()->getValue())),
                        SpvLiteralInteger::from32(int32_t(numThreads->getZ()->getValue())));
                }
            }
            break;
        case kIROp_MaxVertexCountDecoration:
            // Don't do anything here, instead wait until we see OutputTopologyDecoration
            // and emit them together to ensure MaxVertexCount always appears before
            // OutputTopology, which seemed to be required by SPIRV.
            break;
        case kIROp_InstanceDecoration:
            {
                auto decor = as<IRInstanceDecoration>(decoration);
                auto count = int32_t(getIntVal(decor->getCount()));
                requireSPIRVExecutionMode(
                    decoration,
                    dstID,
                    SpvExecutionModeInvocations,
                    SpvLiteralInteger::from32(count));
            }
            break;
        case kIROp_TriangleInputPrimitiveTypeDecoration:
        case kIROp_LineInputPrimitiveTypeDecoration:
        case kIROp_LineAdjInputPrimitiveTypeDecoration:
        case kIROp_PointInputPrimitiveTypeDecoration:
        case kIROp_TriangleAdjInputPrimitiveTypeDecoration:
            // Defer this until we see kIROp_StreamOutputTypeDecoration because the driver wants
            // to see them before the output.
            break;
        case kIROp_StreamOutputTypeDecoration:
            {
                for (auto inputDecor : decoration->getParent()->getDecorations())
                {
                    switch (inputDecor->getOp())
                    {
                    case kIROp_TriangleInputPrimitiveTypeDecoration:
                        requireSPIRVExecutionMode(inputDecor, dstID, SpvExecutionModeTriangles);
                        break;
                    case kIROp_LineInputPrimitiveTypeDecoration:
                        requireSPIRVExecutionMode(inputDecor, dstID, SpvExecutionModeInputLines);
                        break;
                    case kIROp_LineAdjInputPrimitiveTypeDecoration:
                        requireSPIRVExecutionMode(
                            inputDecor,
                            dstID,
                            SpvExecutionModeInputLinesAdjacency);
                        break;
                    case kIROp_PointInputPrimitiveTypeDecoration:
                        requireSPIRVExecutionMode(inputDecor, dstID, SpvExecutionModeInputPoints);
                        break;
                    case kIROp_TriangleAdjInputPrimitiveTypeDecoration:
                        requireSPIRVExecutionMode(
                            inputDecor,
                            dstID,
                            SpvExecutionModeInputTrianglesAdjacency);
                        break;
                    }
                }
                // SPIRV requires MaxVertexCount decoration to appear before
                // OutputTopologyDecoration, so we emit them here.
                if (auto maxVertexCount =
                        decoration->getParent()->findDecoration<IRMaxVertexCountDecoration>())
                {
                    requireSPIRVExecutionMode(
                        maxVertexCount,
                        dstID,
                        SpvExecutionModeOutputVertices,
                        SpvLiteralInteger::from32(int32_t(getIntVal(maxVertexCount->getCount()))));
                }

                auto decor = as<IRStreamOutputTypeDecoration>(decoration);
                IRType* type = decor->getStreamType();

                switch (type->getOp())
                {
                case kIROp_HLSLPointStreamType:
                    requireSPIRVExecutionMode(decoration, dstID, SpvExecutionModeOutputPoints);
                    break;
                case kIROp_HLSLLineStreamType:
                    requireSPIRVExecutionMode(decoration, dstID, SpvExecutionModeOutputLineStrip);
                    break;
                case kIROp_HLSLTriangleStreamType:
                    requireSPIRVExecutionMode(
                        decoration,
                        dstID,
                        SpvExecutionModeOutputTriangleStrip);
                    break;
                default:
                    SLANG_ASSERT(!"Unknown stream out type");
                }
            }
            break;
        case kIROp_MaximallyReconvergesDecoration:
            ensureExtensionDeclaration(UnownedStringSlice("SPV_KHR_maximal_reconvergence"));
            requireSPIRVExecutionMode(nullptr, dstID, SpvExecutionModeMaximallyReconvergesKHR);
            break;
        case kIROp_QuadDerivativesDecoration:
            ensureExtensionDeclaration(UnownedStringSlice("SPV_KHR_quad_control"));
            requireSPIRVCapability(SpvCapabilityQuadControlKHR);
            requireSPIRVExecutionMode(nullptr, dstID, SpvExecutionModeQuadDerivativesKHR);
            break;
        case kIROp_RequireFullQuadsDecoration:
            ensureExtensionDeclaration(UnownedStringSlice("SPV_KHR_quad_control"));
            requireSPIRVCapability(SpvCapabilityQuadControlKHR);
            requireSPIRVExecutionMode(nullptr, dstID, SpvExecutionModeRequireFullQuadsKHR);
            break;
        case kIROp_SPIRVBufferBlockDecoration:
            {
                emitOpDecorate(
                    getSection(SpvLogicalSectionID::Annotations),
                    decoration,
                    dstID,
                    SpvDecorationBufferBlock);
            }
            break;
        case kIROp_SPIRVBlockDecoration:
            {
                emitOpDecorate(
                    getSection(SpvLogicalSectionID::Annotations),
                    decoration,
                    dstID,
                    SpvDecorationBlock);
            }
            break;

        case kIROp_SPIRVNonUniformResourceDecoration:
            {
                ensureExtensionDeclarationBeforeSpv15(toSlice("SPV_EXT_descriptor_indexing"));

                requireSPIRVCapability(SpvCapabilityShaderNonUniform);
                emitOpDecorate(
                    getSection(SpvLogicalSectionID::Annotations),
                    decoration,
                    dstID,
                    SpvDecorationNonUniform);
            }
            break;

        case kIROp_OutputTopologyDecoration:
            {
                auto entryPoint = decoration->getParent();
                IREntryPointDecoration* entryPointDecor =
                    entryPoint ? entryPoint->findDecoration<IREntryPointDecoration>() : nullptr;

                const auto o = cast<IROutputTopologyDecoration>(decoration);
                const auto topologyType = OutputTopologyType(o->getTopologyType());

                SpvExecutionMode m = SpvExecutionModeMax;
                if (entryPointDecor)
                {
                    switch (entryPointDecor->getProfile().getStage())
                    {
                    case Stage::Domain:
                    case Stage::Hull:
                        if (topologyType == OutputTopologyType::TriangleCW)
                            m = SpvExecutionModeVertexOrderCw;
                        else if (topologyType == OutputTopologyType::TriangleCCW)
                            m = SpvExecutionModeVertexOrderCcw;
                        break;
                    }
                }
                if (m == SpvExecutionModeMax)
                {
                    if (topologyType == OutputTopologyType::Triangle)
                        m = SpvExecutionModeOutputTrianglesEXT;
                    else if (topologyType == OutputTopologyType::Line)
                        m = SpvExecutionModeOutputLinesEXT;
                    else if (topologyType == OutputTopologyType::Point)
                        m = SpvExecutionModeOutputPoints;
                }

                SLANG_ASSERT(m != SpvExecutionModeMax);
                requireSPIRVExecutionMode(decoration, dstID, m);
            }
            break;

        case kIROp_VerticesDecoration:
            {
                const auto c = cast<IRVerticesDecoration>(decoration);
                requireSPIRVExecutionMode(
                    decoration,
                    dstID,
                    SpvExecutionModeOutputVertices,
                    SpvLiteralInteger::from32(int32_t(c->getMaxSize()->getValue())));
            }
            break;

        case kIROp_PrimitivesDecoration:
            {
                const auto c = cast<IRPrimitivesDecoration>(decoration);
                requireSPIRVExecutionMode(
                    decoration,
                    dstID,
                    SpvExecutionModeOutputPrimitivesEXT,
                    SpvLiteralInteger::from32(int32_t(c->getMaxSize()->getValue())));
            }
            break;

        case kIROp_VulkanHitAttributesDecoration:
        case kIROp_VulkanCallablePayloadDecoration:
        case kIROp_VulkanCallablePayloadInDecoration:
            ensureExtensionDeclaration(UnownedStringSlice("SPV_KHR_ray_tracing"));
            requireSPIRVCapability(SpvCapabilityRayTracingKHR);
            isRayTracingObject = true;
            break;
        case kIROp_VulkanHitObjectAttributesDecoration:
            // needed since GLSL will not set optypes accordingly, but will keep the decoration
            ensureExtensionDeclaration(UnownedStringSlice("SPV_NV_shader_invocation_reorder"));
            requireSPIRVCapability(SpvCapabilityShaderInvocationReorderNV);
            isRayTracingObject = true;
            break;
        case kIROp_VulkanRayPayloadDecoration:
        case kIROp_VulkanRayPayloadInDecoration:
            // needed since GLSL will not set optypes accordingly, but will keep the decoration
            ensureExtensionDeclaration(UnownedStringSlice("SPV_KHR_ray_tracing"));
            requireSPIRVCapability(SpvCapabilityRayTracingKHR);
            isRayTracingObject = true;
            break;
        case kIROp_InterpolationModeDecoration:
            _maybeEmitInterpolationModifierDecoration(
                (IRInterpolationMode)getIntVal(decoration->getOperand(0)),
                dstID);
            break;
        case kIROp_PerVertexDecoration:
            ensureExtensionDeclaration(UnownedStringSlice("SPV_KHR_fragment_shader_barycentric"));
            requireSPIRVCapability(SpvCapabilityFragmentBarycentricKHR);
            emitOpDecorate(
                getSection(SpvLogicalSectionID::Annotations),
                decoration,
                dstID,
                SpvDecorationPerVertexKHR);
            break;
        case kIROp_OutputControlPointsDecoration:
            requireSPIRVExecutionMode(
                decoration,
                dstID,
                SpvExecutionModeOutputVertices,
                SpvLiteralInteger::from32(int32_t(getIntVal(decoration->getOperand(0)))));
            break;
        case kIROp_DomainDecoration:
            {
                auto domain = cast<IRDomainDecoration>(decoration);
                SpvExecutionMode mode = SpvExecutionModeMax;
                auto domainName = as<IRStringLit>(domain->getDomain());
                if (!domainName)
                    break;
                auto domainStr = domainName->getStringSlice();
                if (domainStr.startsWithCaseInsensitive(toSlice("tri")))
                    mode = SpvExecutionModeTriangles;
                else if (domainStr.caseInsensitiveEquals(toSlice("quad")))
                    mode = SpvExecutionModeQuads;
                else if (domainStr.caseInsensitiveEquals(toSlice("isoline")))
                    mode = SpvExecutionModeIsolines;
                if (mode != SpvExecutionModeMax)
                    requireSPIRVExecutionMode(decoration, dstID, mode);
            }
            break;
        case kIROp_MemoryQualifierSetDecoration:
            {
                auto collection = as<IRMemoryQualifierSetDecoration>(decoration);
                IRIntegerValue flags = collection->getMemoryQualifierBit();
                if (flags & MemoryQualifierSetModifier::Flags::kCoherent)
                {
                    emitOpDecorate(
                        getSection(SpvLogicalSectionID::Annotations),
                        nullptr,
                        dstID,
                        SpvDecorationCoherent);
                }
                if (flags & MemoryQualifierSetModifier::Flags::kVolatile)
                {
                    emitOpDecorate(
                        getSection(SpvLogicalSectionID::Annotations),
                        nullptr,
                        dstID,
                        SpvDecorationVolatile);
                }
                if (flags & MemoryQualifierSetModifier::Flags::kRestrict)
                {
                    emitOpDecorate(
                        getSection(SpvLogicalSectionID::Annotations),
                        nullptr,
                        dstID,
                        SpvDecorationRestrict);
                }
                if (flags & MemoryQualifierSetModifier::Flags::kReadOnly)
                {
                    emitOpDecorate(
                        getSection(SpvLogicalSectionID::Annotations),
                        nullptr,
                        dstID,
                        SpvDecorationNonWritable);
                }
                if (flags & MemoryQualifierSetModifier::Flags::kWriteOnly)
                {
                    emitOpDecorate(
                        getSection(SpvLogicalSectionID::Annotations),
                        nullptr,
                        dstID,
                        SpvDecorationNonReadable);
                }
                break;
            }
        case kIROp_DownstreamModuleExportDecoration:
            {
                requireSPIRVCapability(SpvCapabilityLinkage);
                auto name =
                    decoration->getParent()->findDecoration<IRExportDecoration>()->getMangledName();
                emitInst(
                    getSection(SpvLogicalSectionID::Annotations),
                    decoration,
                    SpvOpDecorate,
                    dstID,
                    SpvDecorationLinkageAttributes,
                    name,
                    SpvLinkageTypeExport);
                break;
            }
        case kIROp_DownstreamModuleImportDecoration:
            {
                requireSPIRVCapability(SpvCapabilityLinkage);
                auto name =
                    decoration->getParent()->findDecoration<IRExportDecoration>()->getMangledName();
                emitInst(
                    getSection(SpvLogicalSectionID::Annotations),
                    decoration,
                    SpvOpDecorate,
                    dstID,
                    SpvDecorationLinkageAttributes,
                    name,
                    SpvLinkageTypeImport);
                break;
            }
            // ...
        }

        if (isRayTracingObject)
        {
            if (decoration->getOperandCount() > 0)
            {
                // if not greater than 0, this is not a layout decoration (no val)
                emitOpDecorateLocation(
                    getSection(SpvLogicalSectionID::Annotations),
                    decoration,
                    dstID,
                    SpvLiteralInteger::from32(int32_t(getIntVal(decoration->getOperand(0)))));
            }
        }

        if (shouldEmitSPIRVReflectionInfo())
        {
            switch (decoration->getOp())
            {
            default:
                break;
            case kIROp_SemanticDecoration:
                {
                    ensureExtensionDeclarationBeforeSpv14(
                        toSlice("SPV_GOOGLE_hlsl_functionality1"));
                    emitOpDecorateString(
                        getSection(SpvLogicalSectionID::Annotations),
                        decoration,
                        dstID,
                        SpvDecorationUserSemantic,
                        cast<IRSemanticDecoration>(decoration)->getSemanticName());
                }
                break;
            case kIROp_UserTypeNameDecoration:
                {
                    ensureExtensionDeclaration(toSlice("SPV_GOOGLE_user_type"));
                    ensureExtensionDeclarationBeforeSpv14(
                        toSlice("SPV_GOOGLE_hlsl_functionality1"));
                    emitOpDecorateString(
                        getSection(SpvLogicalSectionID::Annotations),
                        decoration,
                        dstID,
                        SpvDecorationUserTypeGOOGLE,
                        legalizeUserTypeName(cast<IRUserTypeNameDecoration>(decoration)
                                                 ->getUserTypeName()
                                                 ->getStringSlice())
                            .getUnownedSlice());
                }
                break;
            case kIROp_CounterBufferDecoration:
                {
                    emitOpDecorateCounterBuffer(
                        getSection(SpvLogicalSectionID::Annotations),
                        decoration,
                        dstID,
                        as<IRCounterBufferDecoration>(decoration)->getCounterBuffer());
                }
                break;
            }
        }
    }

    bool isPhysicalCompositeType(IRType* type)
    {
        for (auto decor : type->getDecorations())
        {
            switch (decor->getOp())
            {
            case kIROp_PhysicalTypeDecoration:
            case kIROp_SPIRVBlockDecoration:
            case kIROp_SPIRVBufferBlockDecoration:
                return true;
            }
        }
        return false;
    }

    void emitLayoutDecorations(IRStructType* structType, SpvWord spvStructID)
    {
        /*****
         * SPIRV Spec:
         * Each structure-type member must have an Offset decoration.
         *
         * Each array type must have an ArrayStride decoration, unless it is an
         * array that contains a structure decorated with Block or BufferBlock, in
         * which case it must not have an ArrayStride decoration.
         *
         * Each structure-type member that is a matrix or array-of-matrices must be
         * decorated with a MatrixStride Decoration, and one of the RowMajor or
         * ColMajor decorations.
         *
         * The ArrayStride, MatrixStride, and Offset decorations must be large
         * enough to hold the size of the objects they affect (that is, specifying
         * overlap is invalid). Each ArrayStride and MatrixStride must be greater
         * than zero, and it is invalid for two members of a given structure to be
         * assigned the same Offset.
         *
         *****/
        auto layout = structType->findDecoration<IRSizeAndAlignmentDecoration>();
        IRTypeLayoutRuleName layoutRuleName = IRTypeLayoutRuleName::Natural;
        if (layout)
        {
            layoutRuleName = layout->getLayoutName();
        }
        int32_t id = 0;
        bool isPhysicalType = isPhysicalCompositeType(structType);
        for (auto field : structType->getFields())
        {
            for (auto decor : field->getKey()->getDecorations())
            {
                if (auto fieldNameDecor = as<IRNameHintDecoration>(decor))
                {
                    emitOpMemberName(
                        getSection(SpvLogicalSectionID::DebugNames),
                        nullptr,
                        spvStructID,
                        id,
                        fieldNameDecor->getName());
                }
                else if (auto collection = as<IRMemoryQualifierSetDecoration>(decor))
                {
                    IRIntegerValue flags = collection->getMemoryQualifierBit();
                    if (flags & MemoryQualifierSetModifier::Flags::kCoherent)
                    {
                        emitOpMemberDecorate(
                            getSection(SpvLogicalSectionID::Annotations),
                            nullptr,
                            spvStructID,
                            SpvLiteralInteger::from32(id),
                            SpvDecorationCoherent);
                    }
                    if (flags & MemoryQualifierSetModifier::Flags::kVolatile)
                    {
                        emitOpMemberDecorate(
                            getSection(SpvLogicalSectionID::Annotations),
                            nullptr,
                            spvStructID,
                            SpvLiteralInteger::from32(id),
                            SpvDecorationVolatile);
                    }
                    if (flags & MemoryQualifierSetModifier::Flags::kRestrict)
                    {
                        emitOpMemberDecorate(
                            getSection(SpvLogicalSectionID::Annotations),
                            nullptr,
                            spvStructID,
                            SpvLiteralInteger::from32(id),
                            SpvDecorationRestrict);
                    }
                    if (flags & MemoryQualifierSetModifier::Flags::kReadOnly)
                    {
                        emitOpMemberDecorate(
                            getSection(SpvLogicalSectionID::Annotations),
                            nullptr,
                            spvStructID,
                            SpvLiteralInteger::from32(id),
                            SpvDecorationNonWritable);
                    }
                    if (flags & MemoryQualifierSetModifier::Flags::kWriteOnly)
                    {
                        emitOpMemberDecorate(
                            getSection(SpvLogicalSectionID::Annotations),
                            nullptr,
                            spvStructID,
                            SpvLiteralInteger::from32(id),
                            SpvDecorationNonReadable);
                    }
                }
                else if (
                    auto semanticDecor = field->getKey()->findDecoration<IRSemanticDecoration>())
                {
                    if (shouldEmitSPIRVReflectionInfo())
                    {
                        ensureExtensionDeclarationBeforeSpv14(
                            toSlice("SPV_GOOGLE_hlsl_functionality1"));
                        emitOpMemberDecorateString(
                            getSection(SpvLogicalSectionID::Annotations),
                            nullptr,
                            spvStructID,
                            SpvLiteralInteger::from32(id),
                            SpvDecorationUserSemantic,
                            semanticDecor->getSemanticName());
                    }
                }
            }

            if (!isPhysicalType)
                continue;

            // Emit explicit struct field layout decorations if the struct is physical.
            IRIntegerValue offset = 0;
            if (auto offsetDecor = field->getKey()->findDecoration<IRPackOffsetDecoration>())
            {
                offset = (getIntVal(offsetDecor->getRegisterOffset()) * 4 +
                          getIntVal(offsetDecor->getComponentOffset())) *
                         4;
            }
            else
            {
                getOffset(
                    m_targetProgram->getOptionSet(),
                    IRTypeLayoutRules::get(layoutRuleName),
                    field,
                    &offset);
            }
            emitOpMemberDecorateOffset(
                getSection(SpvLogicalSectionID::Annotations),
                nullptr,
                spvStructID,
                SpvLiteralInteger::from32(id),
                SpvLiteralInteger::from32(int32_t(offset)));
            auto matrixType = as<IRMatrixType>(field->getFieldType());
            auto arrayType = as<IRArrayTypeBase>(field->getFieldType());
            if (!matrixType && arrayType)
            {
                matrixType = as<IRMatrixType>(arrayType->getElementType());
            }
            if (matrixType)
            {
                // SPIRV sepc on MatrixStride:
                // Applies only to a member of a structure type. Only valid on a
                // matrix or array whose most basic element is a matrix. Matrix
                // Stride is an unsigned 32 - bit integer specifying the stride
                // of the rows in a RowMajor - decorated matrix or columns in a
                // ColMajor - decorated matrix.
                IRIntegerValue matrixStride = 0;
                auto rule = IRTypeLayoutRules::get(layoutRuleName);
                IRSizeAndAlignment elementSizeAlignment;
                getSizeAndAlignment(
                    m_targetProgram->getOptionSet(),
                    rule,
                    matrixType->getElementType(),
                    &elementSizeAlignment);
                IRIntegerValue matrixMinorVectorCount = 0;
                // Reminder: the meaning of row/column major layout
                // in our semantics is the *opposite* of what GLSL/SPIRV
                // calls them, because what they call "columns"
                // are what we call "rows."
                //
                if (getIntVal(matrixType->getLayout()) == SLANG_MATRIX_LAYOUT_COLUMN_MAJOR)
                {
                    emitOpMemberDecorate(
                        getSection(SpvLogicalSectionID::Annotations),
                        nullptr,
                        spvStructID,
                        SpvLiteralInteger::from32(id),
                        SpvDecorationRowMajor);
                    matrixMinorVectorCount = getIntVal(matrixType->getRowCount());
                }
                else
                {
                    emitOpMemberDecorate(
                        getSection(SpvLogicalSectionID::Annotations),
                        nullptr,
                        spvStructID,
                        SpvLiteralInteger::from32(id),
                        SpvDecorationColMajor);
                    matrixMinorVectorCount = getIntVal(matrixType->getColumnCount());
                }

                // We need the size of our vector. To get the stride we need to know how 'big'
                // each vector element is inside an array, due to this we align our vector
                // as if a composite.
                auto vectorSize =
                    rule->getVectorSizeAndAlignment(elementSizeAlignment, matrixMinorVectorCount);
                vectorSize = rule->alignCompositeElement(vectorSize);
                matrixStride = vectorSize.getStride();
                emitOpMemberDecorateMatrixStride(
                    getSection(SpvLogicalSectionID::Annotations),
                    nullptr,
                    spvStructID,
                    SpvLiteralInteger::from32(id),
                    SpvLiteralInteger::from32((int32_t)matrixStride));
            }
            id++;
        }
    }

    /// Map a Slang `Stage` to a corresponding SPIR-V execution model
    SpvExecutionModel mapStageToExecutionModel(Stage stage)
    {
        switch (stage)
        {
        default:
            SLANG_UNEXPECTED("unhandled stage");
            UNREACHABLE_RETURN((SpvExecutionModel)0);

#define CASE(STAGE, MODEL) \
    case Stage::STAGE:     \
        return SpvExecutionModel##MODEL

            CASE(Vertex, Vertex);
            CASE(Hull, TessellationControl);
            CASE(Domain, TessellationEvaluation);
            CASE(Geometry, Geometry);
            CASE(Fragment, Fragment);
            CASE(Compute, GLCompute);
            CASE(Mesh, MeshEXT);
            CASE(Amplification, TaskEXT);
            CASE(ClosestHit, ClosestHitKHR);
            CASE(AnyHit, AnyHitKHR);
            CASE(Callable, CallableKHR);
            CASE(Miss, MissKHR);
            CASE(Intersection, IntersectionKHR);
            CASE(RayGeneration, RayGenerationKHR);
            // TODO: Extended execution models for ray tracing, etc.

#undef CASE
        }
    }

    struct BuiltinSpvVarKey
    {
        SpvBuiltIn builtinName;
        SpvStorageClass storageClass = SpvStorageClassInput;
        bool flat = false;
        BuiltinSpvVarKey() = default;
        BuiltinSpvVarKey(SpvBuiltIn builtin, SpvStorageClass storageClass, bool isFlat)
            : builtinName(builtin), storageClass(storageClass), flat(isFlat)
        {
        }
        bool operator==(const BuiltinSpvVarKey& other) const
        {
            return builtinName == other.builtinName && storageClass == other.storageClass &&
                   flat == other.flat;
        }
        HashCode getHashCode() const
        {
            return combineHash(
                Slang::getHashCode(builtinName),
                Slang::getHashCode(storageClass),
                Slang::getHashCode(flat));
        }
    };
    Dictionary<BuiltinSpvVarKey, SpvInst*> m_builtinGlobalVars;


    bool isInstUsedInStage(IRInst* inst, Stage s)
    {
        auto* referencingEntryPoints = m_referencingEntryPoints.tryGetValue(inst);
        if (!referencingEntryPoints)
            return false;
        for (auto entryPoint : *referencingEntryPoints)
        {
            if (auto entryPointDecor = entryPoint->findDecoration<IREntryPointDecoration>())
            {
                if (entryPointDecor->getProfile().getStage() == s)
                    return true;
            }
        }
        return false;
    }

    bool needFlatDecorationForBuiltinVar(IRInst* irInst)
    {
        if (!irInst)
            return false;
        if (irInst->getOp() != kIROp_GlobalVar && irInst->getOp() != kIROp_GlobalParam)
            return false;
        auto ptrType = as<IRPtrType>(irInst->getDataType());
        if (!ptrType)
            return false;
        auto addrSpace = ptrType->getAddressSpace();
        if (addrSpace == AddressSpace::Input || addrSpace == AddressSpace::BuiltinInput)
        {
            if (isIntegralScalarOrCompositeType(ptrType->getValueType()))
            {
                if (isInstUsedInStage(irInst, Stage::Fragment))
                    return true;
            }
        }
        return false;
    }

    SpvInst* getBuiltinGlobalVar(IRType* type, SpvBuiltIn builtinVal, IRInst* irInst)
    {
        SpvInst* result = nullptr;
        auto ptrType = as<IRPtrTypeBase>(type);
        SLANG_ASSERT(ptrType && "`getBuiltinGlobalVar`: `type` must be ptr type.");
        auto storageClass = addressSpaceToStorageClass(ptrType->getAddressSpace());
        bool isFlat = needFlatDecorationForBuiltinVar(irInst);
        auto key = BuiltinSpvVarKey(builtinVal, storageClass, isFlat);
        if (m_builtinGlobalVars.tryGetValue(key, result))
        {
            return result;
        }
        IRBuilder builder(m_irModule);
        builder.setInsertBefore(type);
        auto varInst = emitOpVariable(
            getSection(SpvLogicalSectionID::GlobalVariables),
            nullptr,
            type,
            addressSpaceToStorageClass(ptrType->getAddressSpace()));
        emitOpDecorateBuiltIn(
            getSection(SpvLogicalSectionID::Annotations),
            nullptr,
            varInst,
            builtinVal);
        switch (builtinVal)
        {
        case SpvBuiltInTessLevelInner:
        case SpvBuiltInTessLevelOuter:
            emitOpDecorate(
                getSection(SpvLogicalSectionID::Annotations),
                nullptr,
                varInst,
                SpvDecorationPatch);
            break;
        }
        m_builtinGlobalVars[key] = varInst;

        if (isFlat)
        {
            _maybeEmitInterpolationModifierDecoration(
                IRInterpolationMode::NoInterpolation,
                getID(varInst));
        }

        return varInst;
    }

    SpvInst* maybeEmitSystemVal(IRInst* inst)
    {
        IRBuilder builder(m_irModule);
        builder.setInsertBefore(inst);
        if (auto builtinVarDecor = inst->findDecoration<IRTargetBuiltinVarDecoration>())
        {
            switch (builtinVarDecor->getBuiltinVarName())
            {
            case IRTargetBuiltinVarName::SpvInstanceIndex:
                return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInInstanceIndex, inst);
            case IRTargetBuiltinVarName::SpvBaseInstance:
                requireSPIRVCapability(SpvCapabilityDrawParameters);
                return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInBaseInstance, inst);
            }
        }
        if (auto layout = getVarLayout(inst))
        {
            if (auto systemValueAttr = layout->findAttr<IRSystemValueSemanticAttr>())
            {
                String semanticName = systemValueAttr->getName();
                semanticName = semanticName.toLower();
                if (semanticName == "sv_position")
                {
                    auto importDecor = inst->findDecoration<IRImportDecoration>();
                    if (importDecor->getMangledName() == "gl_FragCoord")
                        return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInFragCoord, inst);
                    else
                        return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInPosition, inst);
                }
                else if (semanticName == "sv_target")
                {
                    // Note: we do *not* need to generate some kind of `gl_`
                    // builtin for fragment-shader outputs: they are just
                    // ordinary `out` variables, with ordinary `location`s,
                    // as far as GLSL is concerned.
                    return nullptr;
                }
                else if (semanticName == "sv_clipdistance")
                {
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInClipDistance, inst);
                }
                else if (semanticName == "sv_culldistance")
                {
                    requireSPIRVCapability(SpvCapabilityCullDistance);
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInCullDistance, inst);
                }
                else if (semanticName == "sv_coverage")
                {
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInSampleMask, inst);
                }
                else if (semanticName == "sv_innercoverage")
                {
                    requireSPIRVCapability(SpvCapabilityFragmentFullyCoveredEXT);
                    ensureExtensionDeclaration(
                        UnownedStringSlice("SPV_EXT_fragment_fully_covered"));
                    return getBuiltinGlobalVar(
                        inst->getFullType(),
                        SpvBuiltInFullyCoveredEXT,
                        inst);
                }
                else if (semanticName == "sv_depth")
                {
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInFragDepth, inst);
                }
                else if (semanticName == "sv_depthgreaterequal")
                {
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInFragDepth, inst);
                }
                else if (semanticName == "sv_depthlessequal")
                {
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInFragDepth, inst);
                }
                else if (semanticName == "sv_dispatchthreadid")
                {
                    return getBuiltinGlobalVar(
                        inst->getFullType(),
                        SpvBuiltInGlobalInvocationId,
                        inst);
                }
                else if (semanticName == "sv_domainlocation")
                {
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInTessCoord, inst);
                }
                else if (semanticName == "sv_groupid")
                {
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInWorkgroupId, inst);
                }
                else if (semanticName == "sv_groupindex")
                {
                    return getBuiltinGlobalVar(
                        inst->getFullType(),
                        SpvBuiltInLocalInvocationIndex,
                        inst);
                }
                else if (semanticName == "sv_groupthreadid")
                {
                    return getBuiltinGlobalVar(
                        inst->getFullType(),
                        SpvBuiltInLocalInvocationId,
                        inst);
                }
                else if (semanticName == "sv_gsinstanceid")
                {
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInInvocationId, inst);
                }
                else if (semanticName == "sv_instanceid")
                {
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInInstanceIndex, inst);
                }
                else if (semanticName == "sv_baseinstanceid")
                {
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInBaseInstance, inst);
                }
                else if (semanticName == "sv_isfrontface")
                {
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInFrontFacing, inst);
                }
                else if (semanticName == "sv_outputcontrolpointid")
                {
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInInvocationId, inst);
                }
                else if (semanticName == "sv_pointsize")
                {
                    // float in hlsl & glsl
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInPointSize, inst);
                }
                else if (semanticName == "sv_pointcoord")
                {
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInPointCoord, inst);
                }
                else if (semanticName == "sv_drawindex")
                {
                    requireSPIRVCapability(SpvCapabilityDrawParameters);
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInDrawIndex, inst);
                }
                else if (semanticName == "sv_primitiveid")
                {
                    auto entryPoints = m_referencingEntryPoints.tryGetValue(inst);
                    // SPIRV requires `Geometry` capability being declared for a fragment
                    // shader, if that shader uses sv_primitiveid.
                    // We will check if this builtin is used by non-ray-tracing, non-geometry or
                    // non-tessellation shader stages, and if so include a declaration of
                    // Geometry capability.
                    bool needGeometryCapability = true;
                    if (entryPoints)
                    {
                        for (auto entryPoint : *entryPoints)
                        {
                            if (auto entryPointDecor =
                                    entryPoint->findDecoration<IREntryPointDecoration>())
                            {
                                switch (entryPointDecor->getProfile().getStage())
                                {
                                case Stage::Mesh:
                                case Stage::Geometry:
                                case Stage::Intersection:
                                case Stage::Amplification:
                                case Stage::AnyHit:
                                case Stage::ClosestHit:
                                case Stage::Hull:
                                case Stage::Domain:
                                    needGeometryCapability = false;
                                    break;
                                }
                            }
                        }
                    }
                    if (needGeometryCapability)
                        requireSPIRVCapability(SpvCapabilityGeometry);
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInPrimitiveId, inst);
                }
                else if (semanticName == "sv_rendertargetarrayindex")
                {
                    if (isSpirv14OrLater())
                        requireSPIRVCapability(SpvCapabilityShaderLayer);
                    else
                        requireSPIRVCapability(SpvCapabilityGeometry);
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInLayer, inst);
                }
                else if (semanticName == "sv_sampleindex")
                {
                    requireSPIRVCapability(SpvCapabilitySampleRateShading);
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInSampleId, inst);
                }
                else if (semanticName == "sv_stencilref")
                {
                    requireSPIRVCapability(SpvCapabilityStencilExportEXT);
                    ensureExtensionDeclaration(UnownedStringSlice("SPV_EXT_shader_stencil_export"));
                    return getBuiltinGlobalVar(
                        inst->getFullType(),
                        SpvBuiltInFragStencilRefEXT,
                        inst);
                }
                else if (semanticName == "sv_tessfactor")
                {
                    requireSPIRVCapability(SpvCapabilityTessellation);
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInTessLevelOuter, inst);
                }
                else if (semanticName == "sv_insidetessfactor")
                {
                    requireSPIRVCapability(SpvCapabilityTessellation);
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInTessLevelInner, inst);
                }
                else if (semanticName == "sv_vertexid")
                {
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInVertexIndex, inst);
                }
                else if (semanticName == "sv_viewid")
                {
                    requireSPIRVCapability(SpvCapabilityMultiView);
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInViewIndex, inst);
                }
                else if (semanticName == "sv_viewportarrayindex")
                {
                    requireSPIRVCapability(SpvCapabilityShaderViewportIndex);
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInViewportIndex, inst);
                }
                else if (semanticName == "nv_x_right")
                {
                    SLANG_UNIMPLEMENTED_X("spirv emit for nv_x_right");
                }
                else if (semanticName == "nv_viewport_mask")
                {
                    requireSPIRVCapability(SpvCapabilityPerViewAttributesNV);
                    ensureExtensionDeclaration(UnownedStringSlice("SPV_NV_mesh_shader"));
                    return getBuiltinGlobalVar(
                        inst->getFullType(),
                        SpvBuiltInViewportMaskPerViewNV,
                        inst);
                }
                else if (semanticName == "sv_barycentrics")
                {
                    requireSPIRVCapability(SpvCapabilityFragmentBarycentricKHR);
                    ensureExtensionDeclaration(
                        UnownedStringSlice("SPV_KHR_fragment_shader_barycentric"));
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInBaryCoordKHR, inst);

                    // TODO: There is also the `gl_BaryCoordNoPerspNV` builtin, which
                    // we ought to use if the `noperspective` modifier has been
                    // applied to this varying input.
                }
                else if (semanticName == "sv_cullprimitive")
                {
                    requireSPIRVCapability(SpvCapabilityMeshShadingEXT);
                    ensureExtensionDeclaration(UnownedStringSlice("SPV_EXT_mesh_shader"));
                    return getBuiltinGlobalVar(
                        inst->getFullType(),
                        SpvBuiltInCullPrimitiveEXT,
                        inst);
                }
                else if (semanticName == "sv_shadingrate")
                {
                    requireSPIRVCapability(SpvCapabilityFragmentShadingRateKHR);
                    ensureExtensionDeclaration(UnownedStringSlice("SPV_KHR_fragment_shading_rate"));
                    auto importDecor = inst->findDecoration<IRImportDecoration>();
                    if (importDecor &&
                        importDecor->getMangledName() == "gl_PrimitiveShadingRateEXT")
                        return getBuiltinGlobalVar(
                            inst->getFullType(),
                            SpvBuiltInPrimitiveShadingRateKHR,
                            inst);
                    else
                        return getBuiltinGlobalVar(
                            inst->getFullType(),
                            SpvBuiltInShadingRateKHR,
                            inst);
                }
                else if (semanticName == "sv_startvertexlocation")
                {
                    requireSPIRVCapability(SpvCapabilityDrawParameters);
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInBaseVertex, inst);
                }
                else if (semanticName == "sv_startinstancelocation")
                {
                    requireSPIRVCapability(SpvCapabilityDrawParameters);
                    return getBuiltinGlobalVar(inst->getFullType(), SpvBuiltInBaseInstance, inst);
                }
                SLANG_UNREACHABLE("Unimplemented system value in spirv emit.");
            }
        }

        //
        // These are system-value variables which require redeclaration in
        // GLSL, SPIR-V makes no such distinction so we can use similar logic
        // to above.
        //
        if (const auto linkageDecoration = inst->findDecoration<IRLinkageDecoration>())
        {
            const auto name = linkageDecoration->getMangledName();
            if (name == "gl_PrimitiveTriangleIndicesEXT")
                return getBuiltinGlobalVar(
                    inst->getFullType(),
                    SpvBuiltInPrimitiveTriangleIndicesEXT,
                    inst);
            if (name == "gl_PrimitiveLineIndicesEXT")
                return getBuiltinGlobalVar(
                    inst->getFullType(),
                    SpvBuiltInPrimitiveLineIndicesEXT,
                    inst);
            if (name == "gl_PrimitivePointIndicesEXT")
                return getBuiltinGlobalVar(
                    inst->getFullType(),
                    SpvBuiltInPrimitivePointIndicesEXT,
                    inst);
        }

        return nullptr;
    }

    void maybeEmitPointerDecoration(SpvInst* varInst, IRType* type, bool isVar, IROp op)
    {
        auto ptrType = as<IRPtrType>(unwrapArray(type));
        if (!ptrType)
            return;
        if (addressSpaceToStorageClass(ptrType->getAddressSpace()) ==
            SpvStorageClassPhysicalStorageBuffer)
        {
            // If inst has a pointer type with PhysicalStorageBuffer address space,
            // emit AliasedPointer decoration.
            emitOpDecorate(
                getSection(SpvLogicalSectionID::Annotations),
                nullptr,
                varInst,
                (isVar ? SpvDecorationAliasedPointer : SpvDecorationAliased));
        }
        else
        {
            // If the pointee type is a pointer with StorageBuffer address space,
            // we also want to emit AliasedPointer decoration.
            ptrType = as<IRPtrType>(unwrapArray(ptrType->getValueType()));
            if (!ptrType)
                return;
            if (addressSpaceToStorageClass(ptrType->getAddressSpace()) ==
                SpvStorageClassPhysicalStorageBuffer)
            {
                emitOpDecorate(
                    getSection(SpvLogicalSectionID::Annotations),
                    nullptr,
                    varInst,
                    (op == kIROp_GlobalVar || op == kIROp_Var || op == kIROp_DebugVar
                         ? SpvDecorationAliasedPointer
                         : SpvDecorationAliased));
            }
        }
    }

    void maybeEmitPointerDecoration(SpvInst* varInst, IRInst* inst)
    {
        maybeEmitPointerDecoration(varInst, inst->getDataType(), as<IRVar>(inst), inst->getOp());
    }

    void maybeEmitWriteOnlyImageDecoration(SpvInst* varInst, IRInst* inst)
    {
        auto ptrType = as<IRPtrTypeBase>(inst->getDataType());
        if (!ptrType)
            return;
        auto textureType = as<IRTextureType>(ptrType->getValueType());
        if (!textureType)
            return;

        if (textureType->getAccess() == SLANG_RESOURCE_ACCESS_WRITE)
        {
            emitOpDecorate(
                getSection(SpvLogicalSectionID::Annotations),
                nullptr,
                getID(varInst),
                SpvDecorationNonReadable);
        }
    }

    SpvInst* emitParam(SpvInstParent* parent, IRInst* inst)
    {
        auto paramSpvInst = emitOpFunctionParameter(parent, inst, inst->getFullType());
        maybeEmitName(paramSpvInst, inst);
        maybeEmitPointerDecoration(paramSpvInst, inst);
        return paramSpvInst;
    }

    SpvInst* emitVar(SpvInstParent* parent, IRInst* inst)
    {
        auto ptrType = as<IRPtrTypeBase>(inst->getDataType());
        SLANG_ASSERT(ptrType);
        SpvStorageClass storageClass = getSpvStorageClass(ptrType);

        auto varSpvInst = emitOpVariable(parent, inst, inst->getFullType(), storageClass);
        maybeEmitName(varSpvInst, inst);
        maybeEmitPointerDecoration(varSpvInst, inst);
        return varSpvInst;
    }

    /// Cached `IRParam` indices in an `IRBlock`. For use in `getParamIndexInBlock`.
    struct BlockParamIndexInfo : public RefObject
    {
        Dictionary<IRParam*, int> mapParamToIndex;
    };
    Dictionary<IRBlock*, RefPtr<BlockParamIndexInfo>> m_mapIRBlockToParamIndexInfo;

    /// Returns the index of an `IRParam` inside a `IRBlock`.
    /// The results are cached in `m_mapIRBlockToParamIndexInfo` to avoid linear search.
    int getParamIndexInBlock(IRBlock* block, IRParam* paramInst)
    {
        RefPtr<BlockParamIndexInfo> info;
        int result = -1;
        if (m_mapIRBlockToParamIndexInfo.tryGetValue(block, info))
        {
            info->mapParamToIndex.tryGetValue(paramInst, result);
            SLANG_ASSERT(result != -1);
            return result;
        }
        info = new BlockParamIndexInfo();
        int paramIndex = 0;
        for (auto param : block->getParams())
        {
            info->mapParamToIndex[param] = paramIndex;
            if (param == paramInst)
                result = paramIndex;
            paramIndex++;
        }
        m_mapIRBlockToParamIndexInfo[block] = info;
        SLANG_ASSERT(result != -1);
        return result;
    }

    bool isGlobalValueInst(IRInst* inst)
    {
        if (as<IRConstant>(inst))
            return true;
        switch (inst->getOp())
        {
        case kIROp_Func:
        case kIROp_GlobalParam:
        case kIROp_GlobalVar:
            return true;
        default:
            return false;
        }
    }

    void emitLoopHeaderBlock(IRLoop* loopInst, SpvInst* loopHeaderBlock)
    {
        bool hasBackJump = false;
        for (auto use = loopInst->getTargetBlock()->firstUse; use; use = use->nextUse)
        {
            if (use->getUser() == loopInst)
                continue;
            hasBackJump = true;
            break;
        }
        if (!hasBackJump)
        {
            // If the loop does not have a back jump, it is used as a breakable region.
            // SPIRV does not allow loops without a back jump, so we are going to emit
            // a switch instead.
            IRBuilder builder(loopInst);
            builder.setInsertBefore(loopInst);
            emitOpSelectionMerge(
                loopHeaderBlock,
                nullptr,
                getIRInstSpvID(loopInst->getBreakBlock()),
                SpvSelectionControlMaskNone);
            emitInst(
                loopHeaderBlock,
                nullptr,
                SpvOpSwitch,
                emitIntConstant(0, builder.getIntType()),
                getIRInstSpvID(loopInst->getTargetBlock()));
            return;
        }

        SpvLoopControlMask loopControl = SpvLoopControlMaskNone;
        if (auto loopControlDecoration = loopInst->findDecoration<IRLoopControlDecoration>())
        {
            switch (loopControlDecoration->getMode())
            {
            case IRLoopControl::kIRLoopControl_Unroll:
                loopControl = SpvLoopControlUnrollMask;
                break;
            case IRLoopControl::kIRLoopControl_Loop:
                loopControl = SpvLoopControlDontUnrollMask;
                break;
            default:
                break;
            }
        }
        emitOpLoopMerge(
            loopHeaderBlock,
            nullptr,
            getIRInstSpvID(loopInst->getBreakBlock()),
            getIRInstSpvID(loopInst->getContinueBlock()),
            loopControl);
        emitOpBranch(loopHeaderBlock, nullptr, loopInst->getTargetBlock());
    }

    SpvInst* emitPhi(SpvInstParent* parent, IRParam* inst)
    {
        requireVariableBufferCapabilityIfNeeded(inst->getDataType());

        // An `IRParam` in an ordinary `IRBlock` represents a phi value.
        // We can translate them directly to SPIRV's `Phi` instruction.
        // In order to do that, we need to figure out the source values
        // of this `IRParam`, which can be done by looking at the users
        // of current `IRBlock`.

        // First, we find the index of this param.
        IRBlock* block = as<IRBlock>(inst->getParent());
        // Special case: if block is a loop's target block, emit phis into the header block instead.
        IRInst* loopInst = nullptr;
        if (isLoopTargetBlock(block, loopInst))
        {
            SpvInst* loopSpvBlockInst = nullptr;
            m_mapIRInstToSpvInst.tryGetValue(loopInst, loopSpvBlockInst);
            SLANG_ASSERT(loopSpvBlockInst);
            parent = loopSpvBlockInst;
        }

        SLANG_ASSERT(block);
        int paramIndex = getParamIndexInBlock(block, inst);

        // Emit a Phi instruction.
        auto phiSpvInst = emitInstCustomOperandFunc(
            parent,
            inst,
            SpvOpPhi,
            [&]()
            {
                emitOperand(inst->getFullType());
                emitOperand(kResultID);
                // Find phi arguments from incoming branch instructions that target `block`.
                for (auto use = block->firstUse; use; use = use->nextUse)
                {
                    auto branchInst = as<IRUnconditionalBranch>(use->getUser());
                    if (!branchInst)
                        continue;
                    if (branchInst->getTargetBlock() != inst->getParent())
                        continue;

                    UInt argStartIndex = 0;
                    switch (branchInst->getOp())
                    {
                    case kIROp_unconditionalBranch:
                        argStartIndex = 1;
                        break;
                    case kIROp_loop:
                        argStartIndex = 3;
                        break;
                    default:
                        // A phi argument can only come from an unconditional branch inst.
                        // Other uses are not relavent so we should skip.
                        continue;
                    }
                    SLANG_ASSERT(argStartIndex + paramIndex < branchInst->getOperandCount());
                    auto valueInst = branchInst->getOperand(argStartIndex + paramIndex);
                    if (isGlobalValueInst(valueInst))
                        ensureInst(valueInst);
                    emitOperand(getIRInstSpvID(valueInst));
                    auto sourceBlock = as<IRBlock>(branchInst->getParent());
                    SLANG_ASSERT(sourceBlock);
                    emitOperand(getIRInstSpvID(sourceBlock));
                }
            });

        maybeEmitName(phiSpvInst, inst);
        return phiSpvInst;
    }

    SpvInst* emitCall(SpvInstParent* parent, IRCall* inst)
    {
        auto funcValue = inst->getCallee();

        // Does this function declare any requirements.
        handleRequiredCapabilities(funcValue);
        requireVariableBufferCapabilityIfNeeded(inst->getDataType());

        // We want to detect any call to an intrinsic operation, and inline
        // the SPIRV snippet directly at the call site.
        if (auto targetIntrinsic = Slang::findBestTargetIntrinsicDecoration(
                funcValue,
                m_targetRequest->getTargetCaps()))
        {
            return emitIntrinsicCallExpr(parent, static_cast<IRCall*>(inst), targetIntrinsic);
        }
        else if (auto spvOpDecor = funcValue->findDecorationImpl(kIROp_SPIRVOpDecoration))
        {
            SpvOp op = (SpvOp)getIntVal(spvOpDecor->getOperand(0));
            List<IRInst*> args;
            for (UInt i = 0; i < inst->getArgCount(); i++)
                args.add(inst->getArg(i));
            return emitInst(parent, inst, op, inst->getFullType(), kResultID, args);
        }
        else
        {
            return emitOpFunctionCall(
                parent,
                inst,
                inst->getFullType(),
                funcValue,
                inst->getArgsList());
        }
    }

    SpvInst* emitIntrinsicCallExpr(
        SpvInstParent* parent,
        IRCall* inst,
        IRTargetIntrinsicDecoration* intrinsic)
    {
        SpvSnippet* snippet = getParsedSpvSnippet(intrinsic);
        SLANG_ASSERT(snippet);
        SpvSnippetEmitContext context;
        context.irResultType = inst->getDataType();
        context.resultType = ensureInst(inst->getFullType());
        context.isResultTypeFloat = isFloatType(inst->getDataType());
        context.isResultTypeSigned = isSignedType((IRType*)inst->getDataType());
        for (SlangUInt i = 0; i < inst->getArgCount(); i++)
        {
            auto argInst = ensureInst(inst->getArg(i));
            if (argInst)
            {
                context.argumentIds.add(getID(argInst));
            }
            else
            {
                context.argumentIds.add(0xFFFFFFFF);
            }
        }
        // A SPIRV snippet may refer to the result type of this inst with a
        // different storage-class qualifier. We need to pre-create these
        // storage-class-qualified result pointer types so they can be used
        // during inlining of the snippet.
        {
            IRBuilder builder(m_irModule);
            builder.setInsertBefore(inst);
            if (snippet->usedPtrResultTypeStorageClasses.getCount())
                SLANG_UNIMPLEMENTED_X("specifying storage classes in __target_intrinsic modifiers");
        }
        return emitSpvSnippet(parent, inst, context, snippet);
    }

    Dictionary<SpvSnippet::ASMConstant, SpvInst*> m_spvSnippetConstantInsts;

    // Emit SPV Inst that represents a constant defined in a SpvSnippet.
    SpvInst* maybeEmitSpvConstant(SpvSnippet::ASMConstant constant)
    {
        SpvInst* result = nullptr;
        if (m_spvSnippetConstantInsts.tryGetValue(constant, result))
            return result;

        IRBuilder builder(m_irModule);
        builder.setInsertInto(m_irModule->getModuleInst());
        switch (constant.type)
        {
        case SpvSnippet::ASMType::Float:
            result = emitFloatConstant(constant.floatValues[0], builder.getType(kIROp_FloatType));
            break;
        case SpvSnippet::ASMType::Float2:
            {
                auto floatType = builder.getType(kIROp_FloatType);
                auto element1 = emitFloatConstant(constant.floatValues[0], floatType);
                auto element2 = emitFloatConstant(constant.floatValues[1], floatType);
                result = emitOpConstantComposite(
                    nullptr,
                    builder.getVectorType(floatType, builder.getIntValue(builder.getIntType(), 2)),
                    makeArray(element1, element2));
            }
            break;
        case SpvSnippet::ASMType::Int:
            result = emitIntConstant((IRIntegerValue)constant.intValues[0], builder.getIntType());
            break;
        case SpvSnippet::ASMType::UInt16:
            result = emitIntConstant(
                (IRIntegerValue)constant.intValues[0],
                builder.getType(kIROp_UInt16Type));
            break;
        case SpvSnippet::ASMType::UInt2:
            {
                auto uintType = builder.getType(kIROp_UIntType);
                auto element1 = emitIntConstant((IRIntegerValue)constant.intValues[0], uintType);
                auto element2 = emitIntConstant((IRIntegerValue)constant.intValues[1], uintType);
                result = emitOpConstantComposite(
                    nullptr,
                    builder.getVectorType(uintType, builder.getIntValue(builder.getIntType(), 2)),
                    makeArray(element1, element2));
            }
            break;
        }
        m_spvSnippetConstantInsts[constant] = result;
        return result;
    }

    // Emit SPV Inst that represents a type defined in a SpvSnippet.
    void emitSpvSnippetASMTypeOperand(SpvSnippet::ASMType type)
    {
        IRBuilder builder(m_irModule);
        builder.setInsertInto(m_irModule->getModuleInst());
        IRType* irType = nullptr;
        switch (type)
        {
        case SpvSnippet::ASMType::Float:
            irType = builder.getType(kIROp_FloatType);
            break;
        case SpvSnippet::ASMType::Half:
            irType = builder.getType(kIROp_HalfType);
            break;
        case SpvSnippet::ASMType::Int:
            irType = builder.getIntType();
            break;
        case SpvSnippet::ASMType::UInt:
            irType = builder.getUIntType();
            break;
        case SpvSnippet::ASMType::UInt16:
            irType = builder.getType(kIROp_UInt16Type);
            break;
        case SpvSnippet::ASMType::Float2:
            irType = builder.getVectorType(
                builder.getType(kIROp_FloatType),
                builder.getIntValue(builder.getIntType(), 2));
            break;
        case SpvSnippet::ASMType::UInt2:
            irType = builder.getVectorType(
                builder.getType(kIROp_UIntType),
                builder.getIntValue(builder.getIntType(), 2));
            break;
        default:
            SLANG_UNEXPECTED("unhandled case in emitSpvSnippetASMTypeOperand");
        }
        emitOperand(irType);
    }

    SpvInst* emitSpvSnippet(
        SpvInstParent* parent,
        IRCall* inst,
        const SpvSnippetEmitContext& context,
        SpvSnippet* snippet)
    {
        ShortList<SpvInst*> emittedInsts;
        for (Index i = 0; i < snippet->instructions.getCount(); i++)
        {
            auto& spvSnippetInst = snippet->instructions[i];
            InstConstructScope scopeInst(this, (SpvOp)spvSnippetInst.opCode, nullptr);
            SpvInst* spvInst = scopeInst;
            for (auto operand : spvSnippetInst.operands)
            {
                switch (operand.type)
                {
                case SpvSnippet::ASMOperandType::SpvWord:
                    emitOperand(operand.content);
                    break;
                case SpvSnippet::ASMOperandType::ObjectReference:
                    SLANG_ASSERT(operand.content < (SpvWord)context.argumentIds.getCount());
                    emitOperand(context.argumentIds[operand.content]);
                    break;
                case SpvSnippet::ASMOperandType::ResultId:
                    emitOperand(kResultID);
                    break;
                case SpvSnippet::ASMOperandType::ResultTypeId:
                    {
                        emitOperand(context.resultType);
                    }
                    break;
                case SpvSnippet::ASMOperandType::InstReference:
                    SLANG_ASSERT(operand.content < (SpvWord)emittedInsts.getCount());
                    emitOperand(emittedInsts[operand.content]);
                    break;
                case SpvSnippet::ASMOperandType::GLSL450ExtInstSet:
                    emitOperand(getGLSL450ExtInst());
                    break;
                case SpvSnippet::ASMOperandType::FloatIntegerSelection:
                    if (context.isResultTypeFloat)
                    {
                        emitOperand(operand.content);
                    }
                    else
                    {
                        emitOperand(operand.content2);
                    }
                    break;
                case SpvSnippet::ASMOperandType::FloatUnsignedSignedSelection:
                    if (context.isResultTypeFloat)
                    {
                        emitOperand(operand.content);
                    }
                    else
                    {
                        if (context.isResultTypeSigned)
                        {
                            emitOperand(operand.content3);
                        }
                        else
                        {
                            emitOperand(operand.content2);
                        }
                    }
                    break;
                case SpvSnippet::ASMOperandType::TypeReference:
                    {
                        emitSpvSnippetASMTypeOperand((SpvSnippet::ASMType)operand.content);
                    }
                    break;
                case SpvSnippet::ASMOperandType::ConstantReference:
                    {
                        auto constant = snippet->constants[operand.content];
                        if (constant.type == SpvSnippet::ASMType::FloatOrDouble)
                        {
                            switch (extractBaseType(context.irResultType))
                            {
                            case BaseType::Float:
                                constant.type = SpvSnippet::ASMType::Float;
                                break;
                            case BaseType::Double:
                                constant.type = SpvSnippet::ASMType::Double;
                                break;
                            default:
                                break;
                            }
                        }
                        SpvInst* spvConstant = maybeEmitSpvConstant(constant);
                        emitOperand(spvConstant);
                    }
                    break;
                }
            }
            parent->addInst(spvInst);
            emittedInsts.add(spvInst);
        }
        auto resultInst = emittedInsts.getLast();
        registerInst(inst, resultInst);
        return resultInst;
    }

    struct StructTypeInfo : public RefObject
    {
        Dictionary<IRStructKey*, Index> structFieldIndices;
    };

    Dictionary<IRStructType*, RefPtr<StructTypeInfo>> m_structTypeInfos;

    RefPtr<StructTypeInfo> createStructTypeInfo(IRStructType* structType)
    {
        RefPtr<StructTypeInfo> typeInfo = new StructTypeInfo();
        Index index = 0;
        for (auto field : structType->getFields())
        {
            typeInfo->structFieldIndices[field->getKey()] = index;
            index++;
        }
        return typeInfo;
    }
    Index getStructFieldId(IRStructType* structType, IRStructKey* structFieldKey)
    {
        RefPtr<StructTypeInfo> info;
        if (!m_structTypeInfos.tryGetValue(structType, info))
        {
            info = createStructTypeInfo(structType);
            m_structTypeInfos[structType] = info;
        }
        Index fieldIndex = -1;
        info->structFieldIndices.tryGetValue(structFieldKey, fieldIndex);
        SLANG_ASSERT(fieldIndex != -1);
        return fieldIndex;
    }

    SpvInst* emitFieldAddress(SpvInstParent* parent, IRFieldAddress* fieldAddress)
    {
        IRBuilder builder(m_irModule);
        builder.setInsertBefore(fieldAddress);

        auto base = fieldAddress->getBase();
        SpvWord baseId = 0;
        IRStructType* baseStructType = nullptr;

        if (auto ptrLikeType = as<IRPointerLikeType>(base->getDataType()))
        {
            baseStructType = as<IRStructType>(ptrLikeType->getElementType());
            baseId = getID(ensureInst(base));
        }
        else if (auto ptrType = as<IRPtrTypeBase>(base->getDataType()))
        {
            baseStructType = as<IRStructType>(ptrType->getValueType());
            baseId = getID(ensureInst(base));
        }
        else
        {
            baseStructType = as<IRStructType>(base->getDataType());

            auto structPtrType = builder.getPtrType(baseStructType);
            auto varInst = emitOpVariable(parent, nullptr, structPtrType, SpvStorageClassFunction);
            emitOpStore(parent, nullptr, varInst, base);
            baseId = getID(varInst);
        }
        SLANG_ASSERT(baseStructType && "field_address requires base to be a struct.");
        auto fieldId = emitIntConstant(
            getStructFieldId(baseStructType, as<IRStructKey>(fieldAddress->getField())),
            builder.getIntType());
        SLANG_ASSERT(as<IRPtrTypeBase>(fieldAddress->getFullType()));
        return emitOpAccessChain(
            parent,
            fieldAddress,
            fieldAddress->getFullType(),
            baseId,
            makeArray(fieldId));
    }

    SpvInst* emitFieldExtract(SpvInstParent* parent, IRFieldExtract* inst)
    {
        IRBuilder builder(m_irModule);
        builder.setInsertBefore(inst);

        IRStructType* baseStructType = as<IRStructType>(inst->getBase()->getDataType());
        SLANG_ASSERT(baseStructType && "field_extract requires base to be a struct.");
        auto fieldId = static_cast<SpvWord>(
            getStructFieldId(baseStructType, as<IRStructKey>(inst->getField())));

        return emitOpCompositeExtract(
            parent,
            inst,
            inst->getDataType(),
            inst->getBase(),
            makeArray(SpvLiteralInteger::from32(fieldId)));
    }

    SpvInst* emitGetOffsetPtr(SpvInstParent* parent, IRInst* inst)
    {
        return emitOpPtrAccessChain(
            parent,
            inst,
            inst->getDataType(),
            inst->getOperand(0),
            inst->getOperand(1));
    }

    SpvInst* emitGetElementPtr(SpvInstParent* parent, IRGetElementPtr* inst)
    {
        IRBuilder builder(m_irModule);
        auto base = inst->getBase();
        const SpvWord baseId = getID(ensureInst(base));

        // We might replace resultType with a different storage class equivalent
        auto resultType = as<IRPtrTypeBase>(inst->getDataType());
        SLANG_ASSERT(resultType);

        if (const auto basePtrType = as<IRPtrTypeBase>(base->getDataType()))
        {
            // If the base pointer has a specific address space and the
            // expected result type doesn't, then make sure they match.
            // It's invalid spir-v if they don't match
            resultType = getPtrTypeWithAddressSpace(
                cast<IRPtrTypeBase>(inst->getDataType()),
                basePtrType->getAddressSpace());
        }
        else
        {
            SLANG_ASSERT(
                as<IRPointerLikeType>(base->getDataType()) ||
                !"invalid IR: base of getElementPtr must be a pointer.");
        }
        return emitOpAccessChain(
            parent,
            inst,
            inst->getFullType(),
            baseId,
            makeArray(inst->getIndex()));
    }

    SpvInst* emitGetElement(SpvInstParent* parent, IRGetElement* inst)
    {
        requireVariableBufferCapabilityIfNeeded(inst->getDataType());

        // Note: SPIRV only supports the case where `index` is constant.
        auto base = inst->getBase();
        const auto baseTy = base->getDataType();
        SLANG_ASSERT(
            as<IRPointerLikeType>(baseTy) || as<IRArrayType>(baseTy) || as<IRVectorType>(baseTy) ||
            as<IRCoopVectorType>(baseTy) || as<IRMatrixType>(baseTy) ||
            as<IRCoopMatrixType>(baseTy));

        IRBuilder builder(m_irModule);
        builder.setInsertBefore(inst);
        auto indexOperand = inst->getIndex();
        if (auto globalValueRef = as<IRGlobalValueRef>(indexOperand))
            indexOperand = globalValueRef->getValue();
        if (auto index = as<IRIntLit>(indexOperand))
        {
            return emitOpCompositeExtract(
                parent,
                inst,
                inst->getFullType(),
                inst->getBase(),
                makeArray(SpvLiteralInteger::from32((int32_t)index->getValue())));
        }
        else
        {
            SLANG_ASSERT(as<IRVectorType>(baseTy) || as<IRCoopVectorType>(baseTy));
            // SPIRV Only allows dynamic element extract on vector types.
            return emitOpVectorExtractDynamic(
                parent,
                inst,
                inst->getFullType(),
                inst->getBase(),
                inst->getIndex());
        }
    }

    SpvInst* emitLoad(SpvInstParent* parent, IRLoad* inst)
    {
        requireVariableBufferCapabilityIfNeeded(inst->getDataType());

        auto ptrType = as<IRPtrTypeBase>(inst->getPtr()->getDataType());
        if (ptrType && addressSpaceToStorageClass(ptrType->getAddressSpace()) ==
                           SpvStorageClassPhysicalStorageBuffer)
        {
            IRSizeAndAlignment sizeAndAlignment;
            if (auto alignedAttr = inst->findAttr<IRAlignedAttr>())
            {
                sizeAndAlignment.alignment = (int)getIntVal(alignedAttr->getAlignment());
            }
            else
            {
                getNaturalSizeAndAlignment(
                    m_targetProgram->getOptionSet(),
                    ptrType->getValueType(),
                    &sizeAndAlignment);
            }
            return emitOpLoadAligned(
                parent,
                inst,
                inst->getDataType(),
                inst->getPtr(),
                SpvLiteralInteger::from32(sizeAndAlignment.alignment));
        }
        else
        {
            return emitOpLoad(parent, inst, inst->getDataType(), inst->getPtr());
        }
    }

    SpvInst* emitStore(SpvInstParent* parent, IRStore* inst)
    {
        auto ptrType = as<IRPtrTypeBase>(inst->getPtr()->getDataType());
        if (ptrType && addressSpaceToStorageClass(ptrType->getAddressSpace()) ==
                           SpvStorageClassPhysicalStorageBuffer)
        {
            IRSizeAndAlignment sizeAndAlignment;
            if (auto alignedAttr = inst->findAttr<IRAlignedAttr>())
            {
                sizeAndAlignment.alignment = (int)getIntVal(alignedAttr->getAlignment());
            }
            else
            {
                getNaturalSizeAndAlignment(
                    m_targetProgram->getOptionSet(),
                    ptrType->getValueType(),
                    &sizeAndAlignment);
            }
            return emitOpStoreAligned(
                parent,
                inst,
                inst->getPtr(),
                inst->getVal(),
                SpvLiteralInteger::from32(sizeAndAlignment.alignment));
        }
        else
        {
            return emitOpStore(parent, inst, inst->getPtr(), inst->getVal());
        }
    }

    SpvInst* emitSwizzledStore(SpvInstParent* parent, IRSwizzledStore* inst)
    {
        auto sourceVectorType = as<IRVectorType>(inst->getSource()->getDataType());
        SLANG_ASSERT(sourceVectorType);
        auto sourceElementType = sourceVectorType->getElementType();
        SLANG_ASSERT(
            getIntVal(sourceVectorType->getElementCount()) ==
            (IRIntegerValue)inst->getElementCount());
        SpvInst* result = nullptr;
        IRBuilder builder(inst);
        builder.setInsertBefore(inst);
        auto destPtrType = as<IRPtrTypeBase>(inst->getDest()->getDataType());
        auto addrSpace = AddressSpace::Function;
        if (destPtrType->hasAddressSpace())
            addrSpace = destPtrType->getAddressSpace();
        auto ptrElementType = builder.getPtrType(kIROp_PtrType, sourceElementType, addrSpace);
        for (UInt i = 0; i < inst->getElementCount(); i++)
        {
            auto index = inst->getElementIndex(i);
            auto addr = emitOpAccessChain(
                parent,
                nullptr,
                ptrElementType,
                inst->getDest(),
                makeArray(index));
            auto val = emitOpCompositeExtract(
                parent,
                nullptr,
                sourceElementType,
                inst->getSource(),
                makeArray(SpvLiteralInteger::from32((int32_t)i)));
            result =
                emitOpStore(parent, (i == inst->getElementCount() - 1 ? inst : nullptr), addr, val);
        }
        return result;
    }

    SpvInst* emitSwizzleSet(SpvInstParent* parent, IRSwizzleSet* inst)
    {
        if (inst->getElementCount() == 1)
        {
            auto index = inst->getElementIndex(0);
            if (auto intLit = as<IRIntLit>(index))
                return emitOpCompositeInsert(
                    parent,
                    inst,
                    inst->getFullType(),
                    inst->getSource(),
                    inst->getBase(),
                    makeArray(SpvLiteralInteger::from32((uint32_t)intLit->value.intVal)));
        }
        auto resultVectorType = as<IRVectorType>(inst->getDataType());
        List<SpvLiteralInteger> shuffleIndices;
        shuffleIndices.setCount((Index)getIntVal(resultVectorType->getElementCount()));
        for (Index i = 0; i < shuffleIndices.getCount(); i++)
            shuffleIndices[i] = SpvLiteralInteger::from32((int32_t)i);

        for (UInt i = 0; i < inst->getElementCount(); i++)
        {
            auto destIndex = (int32_t)getIntVal(inst->getElementIndex(i));
            SLANG_ASSERT(destIndex < shuffleIndices.getCount());
            shuffleIndices[destIndex] =
                SpvLiteralInteger::from32((int32_t)(i + shuffleIndices.getCount()));
        }
        auto source = inst->getSource();
        if (!as<IRVectorType>(source->getDataType()))
        {
            IRBuilder builder(inst);
            builder.setInsertBefore(inst);
            source = builder.emitMakeVectorFromScalar(resultVectorType, source);
        }
        return emitOpVectorShuffle(
            parent,
            inst,
            inst->getFullType(),
            inst->getBase(),
            inst->getSource(),
            shuffleIndices.getArrayView());
    }

    IRPtrTypeBase* getPtrTypeWithAddressSpace(
        IRPtrTypeBase* ptrTypeWithNoAddressSpace,
        AddressSpace addressSpace)
    {
        // If it's already ok, return as is
        if (ptrTypeWithNoAddressSpace->getAddressSpace() == addressSpace)
            return ptrTypeWithNoAddressSpace;

        // It has an address space, but it doesn't match then fail, this
        // indicates a problem with whatever's creating these types
        SLANG_ASSERT(!ptrTypeWithNoAddressSpace->hasAddressSpace());

        IRBuilder builder(ptrTypeWithNoAddressSpace);
        return builder.getPtrType(
            ptrTypeWithNoAddressSpace->getOp(),
            ptrTypeWithNoAddressSpace->getValueType(),
            addressSpace);
    }

    SpvInst* emitStructuredBufferGetElementPtr(SpvInstParent* parent, IRInst* inst)
    {
        //"%addr = OpAccessChain resultType*StorageBuffer resultId _0 const(int, 0) _1;"
        IRBuilder builder(inst);
        auto addressSpace =
            isSpirv14OrLater() ? AddressSpace::StorageBuffer : AddressSpace::Uniform;
        return emitOpAccessChain(
            parent,
            inst,
            // Make sure the resulting pointer has the correct storage class
            getPtrTypeWithAddressSpace(cast<IRPtrTypeBase>(inst->getDataType()), addressSpace),
            inst->getOperand(0),
            makeArray(emitIntConstant(0, builder.getIntType()), ensureInst(inst->getOperand(1))));
    }

    SpvInst* emitStructuredBufferGetDimensions(SpvInstParent* parent, IRInst* inst)
    {
        IRBuilder builder(inst);
        auto arrayLength = emitInst(
            parent,
            nullptr,
            SpvOpArrayLength,
            builder.getUIntType(),
            kResultID,
            inst->getOperand(0),
            SpvLiteralInteger::from32(0));
        auto elementType = as<IRPtrType>(inst->getOperand(0)->getDataType())->getValueType();
        IRIntegerValue stride = 0;
        if (auto sizeDecor = elementType->findDecoration<IRSizeAndAlignmentDecoration>())
        {
            stride = align(sizeDecor->getSize(), (int)sizeDecor->getAlignment());
        }
        auto strideOperand = emitIntConstant(stride, builder.getUIntType());
        auto result =
            emitOpCompositeConstruct(parent, inst, inst->getDataType(), arrayLength, strideOperand);
        return result;
    }

    SpvInst* emitGetBufferPtr(SpvInstParent* parent, IRInst* inst)
    {
        IRBuilder builder(inst);
        auto addressSpace =
            isSpirv14OrLater() ? AddressSpace::StorageBuffer : AddressSpace::Uniform;
        // The buffer is a global parameter, so it's a pointer
        IRPtrTypeBase* bufPtrType = cast<IRPtrTypeBase>(inst->getOperand(0)->getDataType());
        // It's lowered to a struct type..
        IRStructType* bufType = cast<IRStructType>(bufPtrType->getValueType());
        // containing an unsized array, specifically one with an explicit
        // stride, which is not expressible in spirv_asm blocks
        IRArrayTypeBase* arrayType =
            cast<IRArrayTypeBase>(bufType->getFields().getFirst()->getFieldType());
        return emitOpAccessChain(
            parent,
            inst,
            builder.getPtrType(arrayType, addressSpace),
            inst->getOperand(0),
            makeArray(emitIntConstant(0, builder.getIntType())));
    }

    SpvInst* emitSwizzle(SpvInstParent* parent, IRSwizzle* inst)
    {
        if (inst->getElementCount() == 1)
        {
            const auto index = as<IRIntLit>(inst->getElementIndex(0))->getValue();
            return emitOpCompositeExtract(
                parent,
                inst,
                inst->getDataType(),
                inst->getBase(),
                makeArray(SpvLiteralInteger::from32(int32_t(index))));
        }
        else
        {
            return emitInstCustomOperandFunc(
                parent,
                inst,
                SpvOpVectorShuffle,
                [&]()
                {
                    emitOperand(inst->getDataType());
                    emitOperand(kResultID);
                    emitOperand(inst->getBase());
                    emitOperand(inst->getBase());
                    for (UInt i = 0; i < inst->getElementCount(); i++)
                    {
                        auto index = as<IRIntLit>(inst->getElementIndex(i));
                        emitOperand((SpvWord)index->getValue());
                    }
                });
        }
    }

    SpvInst* emitIntCast(SpvInstParent* parent, IRIntCast* inst)
    {
        const auto fromTypeV = inst->getOperand(0)->getDataType();
        const auto toTypeV = inst->getDataType();
        SLANG_ASSERT(!as<IRVectorType>(fromTypeV) == !as<IRVectorType>(toTypeV));
        const auto fromType = getVectorOrCoopMatrixElementType(fromTypeV);
        const auto toType = getVectorOrCoopMatrixElementType(toTypeV);

        if (as<IRBoolType>(fromType))
        {
            // Cast from bool to int.
            IRBuilder builder(inst);
            builder.setInsertBefore(inst);
            auto zero = builder.getIntValue(toType, 0);
            auto one = builder.getIntValue(toType, 1);
            if (auto vecType = as<IRVectorType>(toTypeV))
            {
                auto zeroV =
                    emitSplat(parent, nullptr, zero, getIntVal(vecType->getElementCount()));
                auto oneV = emitSplat(parent, nullptr, one, getIntVal(vecType->getElementCount()));
                return emitInst(
                    parent,
                    inst,
                    SpvOpSelect,
                    inst->getFullType(),
                    kResultID,
                    inst->getOperand(0),
                    oneV,
                    zeroV);
            }
            return emitInst(
                parent,
                inst,
                SpvOpSelect,
                inst->getFullType(),
                kResultID,
                inst->getOperand(0),
                one,
                zero);
        }
        else if (as<IRBoolType>(toType))
        {
            // Cast from int to bool.
            IRBuilder builder(inst);
            builder.setInsertBefore(inst);
            auto zero = builder.getIntValue(fromType, 0);
            if (auto vecType = as<IRVectorType>(toTypeV))
            {
                auto zeroV =
                    emitSplat(parent, nullptr, zero, getIntVal(vecType->getElementCount()));
                return emitOpINotEqual(
                    parent,
                    inst,
                    inst->getFullType(),
                    inst->getOperand(0),
                    zeroV);
            }
            else
            {
                return emitOpINotEqual(
                    parent,
                    inst,
                    inst->getFullType(),
                    inst->getOperand(0),
                    zero);
            }
        }

        SLANG_ASSERT(isIntegralType(fromType));
        SLANG_ASSERT(isIntegralType(toType));

        const auto fromInfo = getIntTypeInfo(fromType);
        const auto toInfo = getIntTypeInfo(toType);

        if (fromInfo == toInfo)
            return emitOpCopyObject(parent, inst, toTypeV, inst->getOperand(0));
        else if (fromInfo.width == toInfo.width)
            return emitOpBitcast(parent, inst, toTypeV, inst->getOperand(0));
        else if (!fromInfo.isSigned && !toInfo.isSigned)
            // unsigned to unsigned, don't sign extend
            return emitOpUConvert(parent, inst, toTypeV, inst->getOperand(0));
        else if (toInfo.isSigned)
            // unsigned to signed, sign extend
            return emitOpSConvert(parent, inst, toTypeV, inst->getOperand(0));
        else if (fromInfo.isSigned)
            // signed to unsigned, sign extend
            return emitOpSConvert(parent, inst, toTypeV, inst->getOperand(0));
        else if (fromInfo.isSigned && toInfo.isSigned)
            // signed to signed, sign extend
            return emitOpSConvert(parent, inst, toTypeV, inst->getOperand(0));

        SLANG_UNREACHABLE(__func__);
    }

    SpvInst* emitFloatCastForMatrix(
        SpvInstParent* parent,
        IRFloatCast* inst,
        IRMatrixType* fromTypeM,
        IRMatrixType* toTypeM)
    {
        // Because there is no spirv instruction to convert matrix to matrix, we need to convert it
        // row by row.
        auto rowCount = getIntVal(fromTypeM->getRowCount());
        auto colCount = getIntVal(fromTypeM->getColumnCount());

        IRBuilder builder(m_irModule);
        // Get from and to type of the row vector
        auto fromTypeV = builder.getVectorType(fromTypeM->getElementType(), colCount);
        auto toVectorV = builder.getVectorType(toTypeM->getElementType(), colCount);

        List<SpvInst*> rowVectorsConverted;
        // convert each row vector to toType.
        for (uint32_t i = 0; i < rowCount; i++)
        {
            auto rowVector = emitOpCompositeExtract(
                parent,
                nullptr,
                fromTypeV,
                inst->getOperand(0),
                makeArray(SpvLiteralInteger::from32(i)));

            auto rowVectorConverted = emitOpFConvert(parent, nullptr, toVectorV, rowVector);
            rowVectorsConverted.add(rowVectorConverted);
        }

        // construct a matrix from the converted row vectors.
        return emitCompositeConstruct(parent, inst, toTypeM, rowVectorsConverted);
    }

    SpvInst* emitFloatCast(SpvInstParent* parent, IRFloatCast* inst)
    {
        const auto fromTypeV = inst->getOperand(0)->getDataType();
        const auto toTypeV = inst->getDataType();

        IRType* fromType = nullptr;
        IRType* toType = nullptr;

        bool isMatrixCast = false;
        if (as<IRVectorType>(fromTypeV) || as<IRVectorType>(toTypeV) ||
            as<IRCoopVectorType>(fromTypeV) || as<IRCoopVectorType>(toTypeV) ||
            // Cooperative matrices behave like vectors where arithmetic operations can be performed
            // directly without having to loop through the matrix and performing operations on the
            // vectors.
            as<IRCoopMatrixType>(fromTypeV) || as<IRCoopMatrixType>(toTypeV))
        {
            fromType = getVectorOrCoopMatrixElementType(fromTypeV);
            toType = getVectorOrCoopMatrixElementType(toTypeV);
        }
        else if (as<IRMatrixType>(fromTypeV) || as<IRMatrixType>(toTypeV))
        {
            fromType = getMatrixElementType(fromTypeV);
            toType = getMatrixElementType(toTypeV);
            isMatrixCast = true;
        }
        else
        {
            fromType = fromTypeV;
            toType = toTypeV;
        }

        // We'd better give some diagnostics to at least point out which line in the shader is
        // wrong, so it can help the user or developers to locate the issue easier.
        if (!isFloatingType(fromType))
        {
            m_sink->diagnose(inst, Diagnostics::internalCompilerError);
        }

        if (!isFloatingType(toType))
        {
            m_sink->diagnose(inst, Diagnostics::internalCompilerError);
        }

        SLANG_ASSERT(isFloatingType(fromType));
        SLANG_ASSERT(isFloatingType(toType));
        SLANG_ASSERT(!isTypeEqual(fromType, toType));

        if (isMatrixCast)
        {
            return emitFloatCastForMatrix(
                parent,
                inst,
                as<IRMatrixType>(fromTypeV),
                as<IRMatrixType>(toTypeV));
        }

        return emitOpFConvert(parent, inst, toTypeV, inst->getOperand(0));
    }

    SpvInst* emitIntToFloatCast(SpvInstParent* parent, IRCastIntToFloat* inst)
    {
        const auto fromTypeV = inst->getOperand(0)->getDataType();
        const auto toTypeV = inst->getDataType();
        SLANG_ASSERT(!as<IRVectorType>(fromTypeV) == !as<IRVectorType>(toTypeV));
        const auto fromType = getVectorOrCoopMatrixElementType(fromTypeV);
        const auto toType = getVectorOrCoopMatrixElementType(toTypeV);

        SLANG_ASSERT(isFloatingType(toType));

        if (isIntegralType(fromType))
        {
            const auto fromInfo = getIntTypeInfo(fromType);

            return fromInfo.isSigned
                       ? emitOpConvertSToF(parent, inst, toTypeV, inst->getOperand(0))
                       : emitOpConvertUToF(parent, inst, toTypeV, inst->getOperand(0));
        }
        else if (as<IRBoolType>(fromType))
        {
            IRBuilder builder(inst);
            builder.setInsertBefore(inst);
            auto one = builder.getFloatValue(toType, 1.0f);
            auto zero = builder.getFloatValue(toType, 0.0f);
            if (as<IRVectorType>(toTypeV))
            {
                one = builder.emitMakeVectorFromScalar(toTypeV, one);
                zero = builder.emitMakeVectorFromScalar(toTypeV, zero);
            }
            return emitInst(
                parent,
                inst,
                SpvOpSelect,
                inst->getFullType(),
                kResultID,
                inst->getOperand(0),
                one,
                zero);
        }
        else
        {
            SLANG_UNREACHABLE("unknown from type");
        }
    }

    SpvInst* emitFloatToIntCast(SpvInstParent* parent, IRCastFloatToInt* inst)
    {
        const auto fromTypeV = inst->getOperand(0)->getDataType();
        const auto toTypeV = inst->getDataType();
        SLANG_ASSERT(!as<IRVectorType>(fromTypeV) == !as<IRVectorType>(toTypeV));
        const auto fromType = getVectorOrCoopMatrixElementType(fromTypeV);
        const auto toType = getVectorOrCoopMatrixElementType(toTypeV);
        SLANG_ASSERT(isFloatingType(fromType));

        if (as<IRBoolType>(toType))
        {
            // Float to bool cast.
            IRBuilder builder(inst);
            builder.setInsertBefore(inst);
            auto zero = builder.getIntValue(fromType, 0);
            if (auto vecType = as<IRVectorType>(toTypeV))
            {
                auto zeroV =
                    emitSplat(parent, nullptr, zero, getIntVal(vecType->getElementCount()));
                return emitInst(
                    parent,
                    inst,
                    SpvOpFUnordNotEqual,
                    inst->getFullType(),
                    kResultID,
                    inst->getOperand(0),
                    zeroV);
            }
            else
            {
                return emitInst(
                    parent,
                    inst,
                    SpvOpFUnordNotEqual,
                    inst->getFullType(),
                    kResultID,
                    inst->getOperand(0),
                    zero);
            }
        }

        SLANG_ASSERT(isIntegralType(toType));

        const auto toInfo = getIntTypeInfo(toType);

        return toInfo.isSigned ? emitOpConvertFToS(parent, inst, toTypeV, inst->getOperand(0))
                               : emitOpConvertFToU(parent, inst, toTypeV, inst->getOperand(0));
    }

    SpvInst* emitCastPtrToInt(SpvInstParent* parent, IRInst* inst)
    {
        return emitInst(
            parent,
            inst,
            SpvOpConvertPtrToU,
            inst->getFullType(),
            kResultID,
            inst->getOperand(0));
    }

    SpvInst* emitCastPtrToBool(SpvInstParent* parent, IRInst* inst)
    {
        IRBuilder builder(inst);
        auto uintVal = emitInst(
            parent,
            nullptr,
            SpvOpConvertPtrToU,
            builder.getUInt64Type(),
            kResultID,
            inst->getOperand(0));
        return emitOpINotEqual(
            parent,
            inst,
            kResultID,
            uintVal,
            builder.getIntValue(builder.getUInt64Type(), 0));
    }

    SpvInst* emitCastIntToPtr(SpvInstParent* parent, IRInst* inst)
    {
        return emitInst(
            parent,
            inst,
            SpvOpConvertUToPtr,
            inst->getFullType(),
            kResultID,
            inst->getOperand(0));
    }

    SpvInst* emitBitfieldExtract(SpvInstParent* parent, IRInst* inst)
    {
        auto dataType = inst->getDataType();
        IRVectorType* vectorType = as<IRVectorType>(dataType);
        Slang::IRType* elementType = dataType;
        if (vectorType)
            elementType = vectorType->getElementType();

        const IntInfo i = getIntTypeInfo(elementType);

        // NM: technically, using bitfield intrinsics for anything non-32-bit goes against
        // VK specification: VUID-StandaloneSpirv-Base-04781. However, it works on at least
        // NVIDIA HW.
        SpvOp opcode = i.isSigned ? SpvOpBitFieldSExtract : SpvOpBitFieldUExtract;
        return emitInst(
            parent,
            inst,
            opcode,
            inst->getFullType(),
            kResultID,
            inst->getOperand(0),
            inst->getOperand(1),
            inst->getOperand(2));
    }

    SpvInst* emitBitfieldInsert(SpvInstParent* parent, IRInst* inst)
    {
        auto dataType = inst->getDataType();
        IRVectorType* vectorType = as<IRVectorType>(dataType);
        Slang::IRType* elementType = dataType;
        if (vectorType)
            elementType = vectorType->getElementType();

        const IntInfo i = getIntTypeInfo(elementType);

        if (i.width == 64)
            requireSPIRVCapability(SpvCapabilityInt64);
        if (i.width == 16)
            requireSPIRVCapability(SpvCapabilityInt16);
        if (i.width == 8)
            requireSPIRVCapability(SpvCapabilityInt8);

        return emitInst(
            parent,
            inst,
            SpvOpBitFieldInsert,
            inst->getFullType(),
            kResultID,
            inst->getOperand(0),
            inst->getOperand(1),
            inst->getOperand(2),
            inst->getOperand(3));
    }

    template<typename T, typename Ts>
    SpvInst* emitCompositeConstruct(
        SpvInstParent* parent,
        IRInst* inst,
        const T& idResultType,
        const Ts& constituents)
    {
        if (parent == getSection(SpvLogicalSectionID::ConstantsAndTypes))
            return emitOpConstantComposite(parent, inst, idResultType, constituents);
        return emitOpCompositeConstruct(parent, inst, idResultType, constituents);
    }

    SpvInst* emitCompositeConstruct(SpvInstParent* parent, IRInst* inst)
    {
        if (parent == getSection(SpvLogicalSectionID::ConstantsAndTypes))
            return emitOpConstantComposite(parent, inst, inst->getDataType(), OperandsOf(inst));
        return emitOpCompositeConstruct(parent, inst, inst->getDataType(), OperandsOf(inst));
    }

    SpvInst* emitMakeArrayFromElement(SpvInstParent* parent, IRInst* inst)
    {
        List<IRInst*> elements;
        auto arrayType = as<IRArrayType>(inst->getDataType());
        auto elementCount = getIntVal(arrayType->getElementCount());
        for (IRIntegerValue i = 0; i < elementCount; i++)
        {
            elements.add(inst->getOperand(0));
        }
        return emitCompositeConstruct(parent, inst, inst->getDataType(), elements);
    }

    SpvInst* emitMakeMatrixFromScalar(SpvInstParent* parent, IRInst* inst)
    {
        List<SpvInst*> rowVectors;
        auto matrixType = as<IRMatrixType>(inst->getDataType());
        auto rowCount = getIntVal(matrixType->getRowCount());
        auto colCount = getIntVal(matrixType->getColumnCount());
        IRBuilder builder(inst);
        builder.setInsertBefore(inst);
        auto rowVectorType = builder.getVectorType(matrixType->getElementType(), colCount);
        List<IRInst*> colElements;
        for (IRIntegerValue i = 0; i < colCount; i++)
        {
            colElements.add(inst->getOperand(0));
        }
        auto rowVector = emitCompositeConstruct(parent, nullptr, rowVectorType, colElements);
        for (IRIntegerValue i = 0; i < rowCount; i++)
        {
            rowVectors.add(rowVector);
        }
        return emitCompositeConstruct(parent, inst, inst->getDataType(), rowVectors);
    }

    SpvInst* emitMakeMatrix(SpvInstParent* parent, IRInst* inst)
    {
        // If operands are already row vectors, use CompositeConstruct directly.
        if (as<IRVectorType>(inst->getOperand(0)->getDataType()))
        {
            return emitCompositeConstruct(parent, inst);
        }
        // Otherwise, operands are raw elements, we need to construct row vectors first,
        // then construct matrix from row vectors.
        List<SpvInst*> rowVectors;
        auto matrixType = as<IRMatrixType>(inst->getDataType());
        auto rowCount = getIntVal(matrixType->getRowCount());
        auto colCount = getIntVal(matrixType->getColumnCount());
        IRBuilder builder(inst);
        builder.setInsertBefore(inst);
        auto rowVectorType = builder.getVectorType(matrixType->getElementType(), colCount);
        List<IRInst*> colElements;
        UInt index = 0;
        for (IRIntegerValue j = 0; j < rowCount; j++)
        {
            colElements.clear();
            for (IRIntegerValue i = 0; i < colCount; i++)
            {
                colElements.add(inst->getOperand(index));
                index++;
            }
            auto rowVector = emitCompositeConstruct(parent, nullptr, rowVectorType, colElements);
            rowVectors.add(rowVector);
        }
        return emitCompositeConstruct(parent, inst, inst->getDataType(), rowVectors);
    }

    SpvInst* emitConstruct(SpvInstParent* parent, IRInst* inst)
    {
        if (as<IRBasicType>(inst->getDataType()))
        {
            if (inst->getOperandCount() == 1)
            {
                if (inst->getDataType() == inst->getOperand(0)->getDataType())
                    return emitOpCopyObject(parent, inst, inst->getFullType(), inst->getOperand(0));
                else
                    return emitOpBitcast(parent, inst, inst->getFullType(), inst->getOperand(0));
            }
            else
            {
                SLANG_ASSERT(!"spirv emit: unsupported Construct inst.");
                return nullptr;
            }
        }
        else
        {
            return emitCompositeConstruct(parent, inst);
        }
    }

    SpvInst* emitSplat(SpvInstParent* parent, IRInst* inst, IRInst* scalar, IRIntegerValue numElems)
    {
        const auto scalarTy = as<IRBasicType>(scalar->getDataType());
        SLANG_ASSERT(scalarTy);
        const auto spvVecTy = ensureVectorType(scalarTy->getBaseType(), numElems, nullptr);
        return emitCompositeConstruct(
            parent,
            inst,
            spvVecTy,
            List<IRInst*>::makeRepeated(scalar, Index(numElems)));
    }

    bool isSignedType(IRType* type)
    {
        switch (type->getOp())
        {
        case kIROp_FloatType:
        case kIROp_DoubleType:
            return true;
        case kIROp_IntType:
        case kIROp_Int16Type:
        case kIROp_Int64Type:
        case kIROp_Int8Type:
            return true;
        case kIROp_VectorType:
            return isSignedType(as<IRVectorType>(type)->getElementType());
        case kIROp_MatrixType:
            return isSignedType(as<IRMatrixType>(type)->getElementType());
        default:
            return false;
        }
    }

    bool isFloatType(IRInst* type)
    {
        switch (type->getOp())
        {
        case kIROp_FloatType:
        case kIROp_DoubleType:
        case kIROp_HalfType:
            return true;
        case kIROp_VectorType:
            return isFloatType(as<IRVectorType>(type)->getElementType());
        case kIROp_MatrixType:
            return isFloatType(as<IRMatrixType>(type)->getElementType());
        default:
            return false;
        }
    }

    SpvInst* emitVectorOrScalarArithmetic(
        SpvInstParent* parent,
        IRInst* instToRegister,
        IRInst* type,
        IROp op,
        UInt operandCount,
        ArrayView<IRInst*> operands)
    {
        IRType* elementType = getVectorOrCoopMatrixElementType(operands[0]->getDataType());
        IRBasicType* basicType = as<IRBasicType>(elementType);
        bool isFloatingPoint = false;
        bool isBool = false;
        switch (basicType->getBaseType())
        {
        case BaseType::Float:
        case BaseType::Double:
        case BaseType::Half:
            isFloatingPoint = true;
            break;
        case BaseType::Bool:
            isBool = true;
            break;
        default:
            break;
        }
        SpvOp opCode = SpvOpUndef;
        bool isSigned = isSignedType(basicType);
        switch (op)
        {
        case kIROp_Add:
            opCode = isFloatingPoint ? SpvOpFAdd : SpvOpIAdd;
            break;
        case kIROp_Sub:
            opCode = isFloatingPoint ? SpvOpFSub : SpvOpISub;
            break;
        case kIROp_Mul:
            opCode = isFloatingPoint ? SpvOpFMul : SpvOpIMul;
            break;
        case kIROp_Div:
            opCode = isFloatingPoint ? SpvOpFDiv : isSigned ? SpvOpSDiv : SpvOpUDiv;
            break;
        case kIROp_IRem:
            opCode = isSigned ? SpvOpSRem : SpvOpUMod;
            break;
        case kIROp_FRem:
            opCode = SpvOpFRem;
            break;
        case kIROp_Less:
            opCode = isFloatingPoint ? SpvOpFOrdLessThan
                     : isSigned      ? SpvOpSLessThan
                                     : SpvOpULessThan;
            break;
        case kIROp_Leq:
            opCode = isFloatingPoint ? SpvOpFOrdLessThanEqual
                     : isSigned      ? SpvOpSLessThanEqual
                                     : SpvOpULessThanEqual;
            break;
        case kIROp_Eql:
            opCode = isFloatingPoint ? SpvOpFOrdEqual : isBool ? SpvOpLogicalEqual : SpvOpIEqual;
            break;
        case kIROp_Neq:
            opCode = isFloatingPoint ? SpvOpFUnordNotEqual
                     : isBool        ? SpvOpLogicalNotEqual
                                     : SpvOpINotEqual;
            break;
        case kIROp_Geq:
            opCode = isFloatingPoint ? SpvOpFOrdGreaterThanEqual
                     : isSigned      ? SpvOpSGreaterThanEqual
                                     : SpvOpUGreaterThanEqual;
            break;
        case kIROp_Greater:
            opCode = isFloatingPoint ? SpvOpFOrdGreaterThan
                     : isSigned      ? SpvOpSGreaterThan
                                     : SpvOpUGreaterThan;
            break;
        case kIROp_Neg:
            opCode = isFloatingPoint ? SpvOpFNegate : SpvOpSNegate;
            break;
        case kIROp_And:
            opCode = SpvOpLogicalAnd;
            break;
        case kIROp_Or:
            opCode = SpvOpLogicalOr;
            break;
        case kIROp_Not:
            opCode = SpvOpLogicalNot;
            break;
        case kIROp_BitAnd:
            if (isBool)
                opCode = SpvOpLogicalAnd;
            else
                opCode = SpvOpBitwiseAnd;
            break;
        case kIROp_BitOr:
            if (isBool)
                opCode = SpvOpLogicalOr;
            else
                opCode = SpvOpBitwiseOr;
            break;
        case kIROp_BitXor:
            if (isBool)
                opCode = SpvOpLogicalNotEqual;
            else
                opCode = SpvOpBitwiseXor;
            break;
        case kIROp_BitNot:
            if (isBool)
                opCode = SpvOpLogicalNot;
            else
                opCode = SpvOpNot;
            break;
        case kIROp_Rsh:
            opCode = isSigned ? SpvOpShiftRightArithmetic : SpvOpShiftRightLogical;
            break;
        case kIROp_Lsh:
            opCode = SpvOpShiftLeftLogical;
            break;
        default:
            SLANG_ASSERT(!"unknown arithmetic opcode");
            break;
        }
        if (operandCount == 1)
        {
            return emitInst(parent, instToRegister, opCode, type, kResultID, operands);
        }
        else if (operandCount == 2)
        {
            auto l = operands[0];
            const auto lVec = as<IRVectorType>(l->getDataType());
            auto r = operands[1];
            const auto rVec = as<IRVectorType>(r->getDataType());
            if (op == kIROp_Mul && isFloatingPoint)
            {
                if (lVec && !rVec)
                {
                    return emitInst(
                        parent,
                        instToRegister,
                        SpvOpVectorTimesScalar,
                        type,
                        kResultID,
                        operands);
                }
                else if (!lVec && rVec)
                {
                    IRInst* newOperands[2] = {operands[1], operands[0]};
                    return emitInst(
                        parent,
                        instToRegister,
                        SpvOpVectorTimesScalar,
                        type,
                        kResultID,
                        ArrayView<IRInst*>(newOperands, 2));
                }
            }
            const auto go = [&](const auto l, const auto r)
            { return emitInst(parent, instToRegister, opCode, type, kResultID, l, r); };
            if (lVec && !rVec)
            {
                const auto len = as<IRIntLit>(lVec->getElementCount());
                SLANG_ASSERT(len);
                return go(l, emitSplat(parent, nullptr, r, len->getValue()));
            }
            else if (!lVec && rVec)
            {
                const auto len = as<IRIntLit>(rVec->getElementCount());
                SLANG_ASSERT(len);
                return go(emitSplat(parent, nullptr, l, len->getValue()), r);
            }
            return go(l, r);
        }
        SLANG_UNREACHABLE("Arithmetic op with 0 or more than 2 operands");
    }


    SpvInst* emitArithmetic(SpvInstParent* parent, IRInst* inst)
    {
        if (const auto matrixType = as<IRMatrixType>(inst->getDataType()))
        {
            auto rowCount = getIntVal(matrixType->getRowCount());
            auto colCount = getIntVal(matrixType->getColumnCount());
            IRBuilder builder(inst);
            builder.setInsertBefore(inst);
            auto rowVectorType = builder.getVectorType(matrixType->getElementType(), colCount);
            List<SpvInst*> rows;
            for (IRIntegerValue i = 0; i < rowCount; i++)
            {
                List<IRInst*> operands;
                for (UInt j = 0; j < inst->getOperandCount(); j++)
                {
                    auto originalOperand = inst->getOperand(j);
                    if (as<IRMatrixType>(originalOperand->getDataType()))
                    {
                        auto operand = builder.emitElementExtract(originalOperand, i);
                        emitLocalInst(parent, operand);
                        operands.add(operand);
                    }
                    else
                    {
                        operands.add(originalOperand);
                    }
                }
                rows.add(emitVectorOrScalarArithmetic(
                    parent,
                    nullptr,
                    rowVectorType,
                    inst->getOp(),
                    inst->getOperandCount(),
                    operands.getArrayView()));
            }
            return emitCompositeConstruct(parent, inst, inst->getDataType(), rows);
        }

        Array<IRInst*, 4> operands;
        for (UInt i = 0; i < inst->getOperandCount(); i++)
            operands.add(inst->getOperand(i));
        return emitVectorOrScalarArithmetic(
            parent,
            inst,
            inst->getDataType(),
            inst->getOp(),
            inst->getOperandCount(),
            operands.getView());
    }

    SpvInst* emitDebugLine(SpvInstParent* parent, IRDebugLine* debugLine)
    {
        auto scope = findDebugScope(debugLine);
        if (!scope)
            return nullptr;
        return emitOpDebugLine(
            parent,
            debugLine,
            debugLine->getFullType(),
            getNonSemanticDebugInfoExtInst(),
            debugLine->getSource(),
            debugLine->getLineStart(),
            debugLine->getLineEnd(),
            debugLine->getColStart(),
            debugLine->getColEnd());
    }

    SpvInst* getDwarfExpr()
    {
        if (m_nullDwarfExpr)
            return m_nullDwarfExpr;
        m_nullDwarfExpr = emitOpDebugExpression(
            getSection(SpvLogicalSectionID::ConstantsAndTypes),
            nullptr,
            m_voidType,
            getNonSemanticDebugInfoExtInst(),
            List<SpvInst*>());
        return m_nullDwarfExpr;
    }

    bool translateIRAccessChain(
        IRBuilder& builder,
        IRInst* baseType,
        const List<IRInst*>& irAccessChain,
        List<SpvInst*>& spvAccessChain)
    {
        auto type = baseType;
        for (Index i = 0; i < irAccessChain.getCount(); i++)
        {
            auto element = irAccessChain[i];
            if (element->getOp() == kIROp_StructKey)
            {
                auto key = as<IRStructKey>(element);
                auto structType = as<IRStructType>(type);
                if (!structType)
                    return false;
                UInt fieldIndex = 0;
                for (auto field : structType->getFields())
                {
                    if (field->getKey() == key)
                    {
                        type = unwrapAttributedType(field->getFieldType());
                        break;
                    }
                    fieldIndex++;
                }
                spvAccessChain.add(emitIntConstant(fieldIndex, builder.getIntType()));
            }
            else
            {
                if (auto arrayType = as<IRArrayTypeBase>(type))
                    type = arrayType->getElementType();
                else if (auto vectorType = as<IRVectorType>(type))
                    type = vectorType->getElementType();
                else if (auto matrixType = as<IRMatrixType>(type))
                    type = builder.getVectorType(
                        matrixType->getElementType(),
                        matrixType->getColumnCount());
                else
                    return false;
                spvAccessChain.add(ensureInst(element));
            }
        }
        return true;
    }

    SpvInst* emitDebugValue(SpvInstParent* parent, IRDebugValue* debugValue)
    {
        // We are asked to update the value for a debug variable.
        // A debug variable is already emited as a OpDebugVariable +
        // OpVariable + OpDebugDeclare. We only need to store the new value
        // into the associated OpVariable. The `debugValue->getDebugVar()` inst
        // already maps to the `OpVariable` SpvInst, so we just need to emit
        // code for a store into the subset of the OpVariable with the correct
        // accesschain defined in the debug value inst.
        //
        IRBuilder builder(debugValue);
        builder.setInsertBefore(debugValue);

        // First we need to check if the debug variable has a backing ordinary
        // variable. If it doesn't, we can't emit a store.
        //
        List<IRInst*> irAccessChain;
        auto rootVar = getRootAddr(debugValue->getDebugVar(), irAccessChain);
        SpvInst* spvDebugVar = nullptr;
        if (!m_mapIRInstToSpvInst.tryGetValue(rootVar, spvDebugVar))
            return nullptr;
        if (!spvDebugVar)
            return nullptr;
        if (spvDebugVar->opcode != SpvOpVariable)
        {
            // If the root variable can't be represented by a normal variable,
            // try to emit a DebugValue if possible. Usually this means that the variable
            // represents a shader resource.
            //
            // SPIR-V requires the access chain specified in a DebugValue operation to
            // be fully static. We will skip emitting the debug inst if the access chain
            // isn't static.
            //
            auto type = unwrapAttributedType(debugValue->getDebugVar()->getDataType());
            List<SpvInst*> accessChain;
            bool isConstAccessChain =
                translateIRAccessChain(builder, type, irAccessChain, accessChain);

            if (isConstAccessChain)
            {
                return emitOpDebugValue(
                    parent,
                    debugValue,
                    m_voidType,
                    getNonSemanticDebugInfoExtInst(),
                    rootVar,
                    debugValue->getValue(),
                    getDwarfExpr(),
                    accessChain);
            }

            // Fallback to not emit anything for now.
            return nullptr;
        }

        // The ordinary case is the debug variable has a backing ordinary variable.
        // We can simply emit a store into the backing variable for the DebugValue operation.
        //
        builder.setInsertBefore(debugValue);
        return emitOpStore(parent, debugValue, debugValue->getDebugVar(), debugValue->getValue());
    }

    IRInst* getName(IRInst* inst)
    {
        IRInst* nameOperand = nullptr;
        for (auto decor : inst->getDecorations())
        {
            if (auto nameHint = as<IRNameHintDecoration>(decor))
                return nameHint->getNameOperand();
            if (auto linkage = as<IRLinkageDecoration>(decor))
                nameOperand = linkage->getMangledNameOperand();
        }
        if (nameOperand)
            return nameOperand;

        IRBuilder builder(inst);
        return builder.getStringValue(toSlice("unamed"));
    }

    Dictionary<IRType*, SpvInst*> m_mapTypeToDebugType;
    HashSet<IRType*> m_emittingTypes; // Types that are being emitted.
    Dictionary<IRType*, SpvInst*> m_mapForwardRefsToDebugType;
    static constexpr const int kUnknownPhysicalLayout = 1 << 17;
    static constexpr const int kDebugTypeAtomicQualifier = 3;

    SpvInst* emitDebugTypeImpl(IRType* type)
    {
        auto scope = findDebugScope(type);
        if (!scope)
            return ensureInst(m_voidType);

        IRBuilder builder(type);
        if (const auto funcType = as<IRFuncType>(type))
        {
            List<SpvInst*> argTypes;
            return emitOpDebugTypeFunction(
                getSection(SpvLogicalSectionID::ConstantsAndTypes),
                nullptr,
                m_voidType,
                getNonSemanticDebugInfoExtInst(),
                builder.getIntValue(builder.getUIntType(), 0),
                ensureInst(m_voidType),
                argTypes);
        }

        auto name = getName(type);

        if (auto structType = as<IRStructType>(type))
        {
            auto loc = structType->findDecoration<IRDebugLocationDecoration>();
            IRInst* source = loc ? loc->getSource() : m_defaultDebugSource;
            IRInst* line = loc ? loc->getLine() : builder.getIntValue(builder.getUIntType(), 0);
            IRInst* col = loc ? loc->getCol() : line;
            if (!name)
            {
                static uint32_t uid = 0;
                uid++;
                name = builder.getStringValue(
                    (String("unnamed_type_") + String(uid)).getUnownedSlice());
            }
            IRSizeAndAlignment structSizeAlignment;
            getNaturalSizeAndAlignment(m_targetProgram->getOptionSet(), type, &structSizeAlignment);

            List<SpvInst*> members;
            for (auto field : structType->getFields())
            {
                IRIntegerValue offset = 0;
                IRSizeAndAlignment sizeAlignment;
                getNaturalOffset(m_targetProgram->getOptionSet(), field, &offset);

                auto fieldType = field->getFieldType();
                getNaturalSizeAndAlignment(
                    m_targetProgram->getOptionSet(),
                    fieldType,
                    &sizeAlignment);

                SpvInst* forwardRef = nullptr;
                SpvInst* spvFieldType = nullptr;
                if (auto fieldPtrType = as<IRPtrTypeBase>(fieldType))
                {
                    auto fieldPtrBaseType = fieldPtrType->getValueType();
                    if (as<IRStructType>(fieldPtrBaseType) &&
                        m_emittingTypes.contains(fieldPtrBaseType))
                    {
                        forwardRef = emitDebugForwardRefs(fieldPtrBaseType);

                        SpvStorageClass storageClass = SpvStorageClassFunction;
                        if (fieldPtrType->hasAddressSpace())
                            storageClass =
                                addressSpaceToStorageClass(fieldPtrType->getAddressSpace());

                        spvFieldType = emitOpDebugTypePointer(
                            getSection(SpvLogicalSectionID::ConstantsAndTypes),
                            nullptr,
                            m_voidType,
                            getNonSemanticDebugInfoExtInst(),
                            forwardRef,
                            builder.getIntValue(builder.getUIntType(), storageClass),
                            builder.getIntValue(builder.getUIntType(), kUnknownPhysicalLayout));
                    }
                }

                if (spvFieldType == nullptr)
                {
                    spvFieldType = emitDebugType(fieldType);
                }

                auto memberType = emitOpDebugTypeMember(
                    getSection(SpvLogicalSectionID::ConstantsAndTypes),
                    nullptr,
                    m_voidType,
                    getNonSemanticDebugInfoExtInst(),
                    getName(field->getKey()),
                    spvFieldType,
                    source,
                    line,
                    col,
                    builder.getIntValue(builder.getUIntType(), offset * 8),
                    builder.getIntValue(builder.getUIntType(), sizeAlignment.size * 8),
                    builder.getIntValue(builder.getUIntType(), 0));
                members.add(memberType);
            }

            SpvInst* forwardRef = nullptr;
            // If `type` has been declared with a forward reference, fill in the forward reference.
            if (m_mapForwardRefsToDebugType.tryGetValue(type, forwardRef))
            {
                // "OpExtInstWithForwardRefsKHR" requires "forward declared ID" at the end.
                auto tmp = m_memoryArena.allocateArray<SpvWord>(
                    forwardRef->operandWordsCount + members.getCount());
                memcpy(
                    tmp,
                    forwardRef->operandWords,
                    forwardRef->operandWordsCount * sizeof(SpvWord));
                for (Index i = 0; i < members.getCount(); i++)
                    tmp[forwardRef->operandWordsCount + i] = getID(members[i]);
                forwardRef->operandWords = tmp;
                forwardRef->operandWordsCount += (uint32_t)members.getCount();
            }

            return emitOpDebugTypeComposite(
                getSection(SpvLogicalSectionID::ConstantsAndTypes),
                nullptr,
                m_voidType,
                getNonSemanticDebugInfoExtInst(),
                name,
                builder.getIntValue(builder.getUIntType(), 1), // struct
                source,
                line,
                col,
                scope,
                name,
                builder.getIntValue(builder.getUIntType(), structSizeAlignment.size * 8),
                builder.getIntValue(builder.getUIntType(), kUnknownPhysicalLayout),
                members);
        }

        if (auto arrayType = as<IRArrayTypeBase>(type))
        {
            auto sizedArrayType = as<IRArrayType>(arrayType);
            return emitOpDebugTypeArray(
                getSection(SpvLogicalSectionID::ConstantsAndTypes),
                nullptr,
                m_voidType,
                getNonSemanticDebugInfoExtInst(),
                emitDebugType(arrayType->getElementType()),
                sizedArrayType ? builder.getIntValue(
                                     builder.getUIntType(),
                                     getIntVal(sizedArrayType->getElementCount()))
                               : builder.getIntValue(builder.getUIntType(), 0));
        }
        else if (auto vectorType = as<IRVectorType>(type))
        {
            auto elementType = emitDebugType(vectorType->getElementType());
            return emitOpDebugTypeVector(
                getSection(SpvLogicalSectionID::ConstantsAndTypes),
                nullptr,
                m_voidType,
                getNonSemanticDebugInfoExtInst(),
                elementType,
                builder.getIntValue(
                    builder.getUIntType(),
                    getIntVal(vectorType->getElementCount())));
        }
        else if (auto matrixType = as<IRMatrixType>(type))
        {
            IRInst* count = nullptr;
            bool isColumnMajor = false;
            IRType* innerVectorType = nullptr;
            if (getIntVal(matrixType->getLayout()) == kMatrixLayoutMode_ColumnMajor)
            {
                innerVectorType =
                    builder.getVectorType(matrixType->getElementType(), matrixType->getRowCount());
                isColumnMajor = true;
                count = matrixType->getColumnCount();
            }
            else
            {
                innerVectorType = builder.getVectorType(
                    matrixType->getElementType(),
                    matrixType->getColumnCount());
                count = matrixType->getRowCount();
            }
            auto elementType = emitDebugType(innerVectorType);
            return emitOpDebugTypeMatrix(
                getSection(SpvLogicalSectionID::ConstantsAndTypes),
                nullptr,
                m_voidType,
                getNonSemanticDebugInfoExtInst(),
                elementType,
                builder.getIntValue(builder.getUIntType(), getIntVal(count)),
                builder.getBoolValue(isColumnMajor));
        }
        else if (auto basicType = as<IRBasicType>(type))
        {
            IRSizeAndAlignment sizeAlignment;
            getNaturalSizeAndAlignment(m_targetProgram->getOptionSet(), basicType, &sizeAlignment);
            int spvEncoding = 0;
            StringBuilder sbName;
            getTypeNameHint(sbName, basicType);
            switch (type->getOp())
            {
            case kIROp_IntType:
            case kIROp_Int16Type:
            case kIROp_Int64Type:
            case kIROp_Int8Type:
            case kIROp_IntPtrType:
                spvEncoding = 4; // Signed
                break;
            case kIROp_UIntType:
            case kIROp_UInt16Type:
            case kIROp_UInt64Type:
            case kIROp_UInt8Type:
            case kIROp_UIntPtrType:
                spvEncoding = 6; // Unsigned
                break;
            case kIROp_FloatType:
            case kIROp_DoubleType:
            case kIROp_HalfType:
                spvEncoding = 3; // Float
                break;
            case kIROp_BoolType:
                spvEncoding = 2; // boolean
                break;
            default:
                spvEncoding = 0; // Unspecified.
                break;
            }
            return emitOpDebugTypeBasic(
                getSection(SpvLogicalSectionID::ConstantsAndTypes),
                nullptr,
                m_voidType,
                getNonSemanticDebugInfoExtInst(),
                builder.getStringValue(sbName.getUnownedSlice()),
                builder.getIntValue(builder.getUIntType(), sizeAlignment.size * 8),
                builder.getIntValue(builder.getUIntType(), spvEncoding),
                builder.getIntValue(builder.getUIntType(), kUnknownPhysicalLayout));
        }
        else if (auto ptrType = as<IRPtrTypeBase>(type))
        {
            IRType* baseType = ptrType->getValueType();
            // Emit DebugTypePointer for pointer types.
            SpvInst* debugBaseType = emitDebugType(baseType);
            SpvStorageClass storageClass = SpvStorageClassFunction;
            if (ptrType->hasAddressSpace())
                storageClass = addressSpaceToStorageClass(ptrType->getAddressSpace());

            return emitOpDebugTypePointer(
                getSection(SpvLogicalSectionID::ConstantsAndTypes),
                nullptr,
                m_voidType,
                getNonSemanticDebugInfoExtInst(),
                debugBaseType,
                builder.getIntValue(builder.getUIntType(), storageClass),
                builder.getIntValue(builder.getUIntType(), kUnknownPhysicalLayout));
        }
        else if (auto atomicType = as<IRAtomicType>(type))
        {
            auto baseType = atomicType->getElementType();
            auto debugBaseType = emitDebugType(baseType);

            return emitOpDebugTypeQualifier(
                getSection(SpvLogicalSectionID::ConstantsAndTypes),
                nullptr,
                m_voidType,
                getNonSemanticDebugInfoExtInst(),
                debugBaseType,
                builder.getIntValue(builder.getUIntType(), kDebugTypeAtomicQualifier));
        }
        return ensureInst(m_voidType);
    }

    SpvInst* emitDebugType(IRType* type)
    {
        if (auto debugType = m_mapTypeToDebugType.tryGetValue(type))
            return *debugType;
        bool isStruct = type->getOp() == kIROp_StructType;
        if (isStruct)
            m_emittingTypes.add(type);
        auto result = emitDebugTypeImpl(type);
        if (isStruct)
            m_emittingTypes.remove(type);
        m_mapTypeToDebugType[type] = result;
        return result;
    }

    SpvInst* emitDebugForwardRefsImpl(IRType* type)
    {
        auto scope = findDebugScope(type);
        if (!scope)
            return ensureInst(m_voidType);

        auto name = getName(type);
        IRBuilder builder(type);

        if (auto structType = as<IRStructType>(type))
        {
            auto loc = structType->findDecoration<IRDebugLocationDecoration>();
            IRInst* source = loc ? loc->getSource() : m_defaultDebugSource;
            IRInst* line = loc ? loc->getLine() : builder.getIntValue(builder.getUIntType(), 0);
            IRInst* col = loc ? loc->getCol() : line;
            if (!name)
            {
                static uint32_t uid = 0;
                uid++;
                name = builder.getStringValue(
                    (String("unnamed_forward_type_") + String(uid)).getUnownedSlice());
            }
            IRSizeAndAlignment structSizeAlignment;
            getNaturalSizeAndAlignment(m_targetProgram->getOptionSet(), type, &structSizeAlignment);

            ensureExtensionDeclaration(UnownedStringSlice("SPV_KHR_relaxed_extended_instruction"));
            return emitOpDebugForwardRefsComposite(
                getSection(SpvLogicalSectionID::ConstantsAndTypes),
                nullptr,
                m_voidType,
                getNonSemanticDebugInfoExtInst(),
                name,
                builder.getIntValue(builder.getUIntType(), 1), // struct
                source,
                line,
                col,
                scope,
                name,
                builder.getIntValue(builder.getUIntType(), structSizeAlignment.size * 8),
                builder.getIntValue(builder.getUIntType(), kUnknownPhysicalLayout));
        }
        SLANG_UNIMPLEMENTED_X("Not implemented forward pointer debug type");
    }

    SpvInst* emitDebugForwardRefs(IRType* type)
    {
        if (auto debugType = m_mapForwardRefsToDebugType.tryGetValue(type))
            return *debugType;
        auto result = emitDebugForwardRefsImpl(type);
        m_mapForwardRefsToDebugType[type] = result;
        return result;
    }

    SpvInst* emitDebugFunction(SpvInst* firstBlock, SpvInst* spvFunc, IRFunc* function)
    {
        auto scope = findDebugScope(function);
        if (!scope)
            return nullptr;
        auto name = getName(function);
        if (!name)
            return nullptr;
        auto debugLoc = function->findDecoration<IRDebugLocationDecoration>();
        if (!debugLoc)
            return nullptr;
        auto debugType = emitDebugType(function->getDataType());
        IRBuilder builder(function);
        auto debugFunc = emitOpDebugFunction(
            getSection(SpvLogicalSectionID::ConstantsAndTypes),
            nullptr,
            m_voidType,
            getNonSemanticDebugInfoExtInst(),
            name,
            debugType,
            debugLoc->getSource(),
            debugLoc->getLine(),
            debugLoc->getCol(),
            scope,
            name,
            builder.getIntValue(builder.getUIntType(), 0),
            debugLoc->getLine());
        registerDebugInst(function, debugFunc);

        emitOpDebugFunctionDefinition(
            firstBlock,
            nullptr,
            m_voidType,
            getNonSemanticDebugInfoExtInst(),
            debugFunc,
            spvFunc);

        return debugFunc;
    }

    SpvInst* emitSPIRVAsm(SpvInstParent* parent, IRSPIRVAsm* inst)
    {
        SpvInst* last = nullptr;

        // This keeps track of the named IDs used in the asm block
        Dictionary<UnownedStringSlice, SpvWord> idMap;

        for (const auto spvInst : inst->getInsts())
        {
            const bool isLast = spvInst == inst->getLastChild();

            const auto parentForOpCode =
                [this](SpvOp opcode, SpvInstParent* defaultParent) -> SpvInstParent*
            {
                const auto info = m_grammarInfo->opInfos.lookup(opcode);
                SLANG_ASSERT(info.has_value());
                switch (info->class_)
                {
                case SPIRVCoreGrammarInfo::OpInfo::TypeDeclaration:
                case SPIRVCoreGrammarInfo::OpInfo::ConstantCreation:
                    return getSection(SpvLogicalSectionID::ConstantsAndTypes);
                // Don't add this case, it's not correct as not all "Debug"
                // instructions belong in this block
                // case SPIRVCoreGrammarInfo::OpInfo::Debug:
                //     return getSection(SpvLogicalSectionID::DebugNames);
                default:
                    switch (opcode)
                    {
                    case SpvOpName:
                        return getSection(SpvLogicalSectionID::DebugNames);
                    case SpvOpCapability:
                        return getSection(SpvLogicalSectionID::Capabilities);
                    case SpvOpExtension:
                        return getSection(SpvLogicalSectionID::Extensions);
                    case SpvOpExecutionMode:
                    case SpvOpExecutionModeId:
                        return getSection(SpvLogicalSectionID::ExecutionModes);
                    case SpvOpDecorate:
                    case SpvOpDecorateId:
                    case SpvOpDecorateString:
                    case SpvOpMemberDecorate:
                    case SpvOpMemberDecorateString:
                        return getSection(SpvLogicalSectionID::Annotations);
                    case SpvOpTypeNodePayloadArrayAMDX:
                        return getSection(SpvLogicalSectionID::ConstantsAndTypes);
                    default:
                        return defaultParent;
                    }
                }
            };

            const auto emitSpvAsmOperand = [&](IRSPIRVAsmOperand* operand)
            {
                switch (operand->getOp())
                {
                case kIROp_SPIRVAsmOperandEnum:
                case kIROp_SPIRVAsmOperandLiteral:
                    {
                        const auto v = as<IRConstant>(operand->getValue());
                        SLANG_ASSERT(v);
                        if (operand->getOperandCount() >= 2)

                        {
                            const auto constantType = cast<IRType>(operand->getOperand(1));
                            SpvInst* constant;
                            switch (v->getOp())
                            {
                            case kIROp_IntLit:
                                {
                                    // TODO: range checking
                                    const auto i = cast<IRIntLit>(v)->getValue();
                                    constant = emitIntConstant(i, constantType);
                                    break;
                                }
                            case kIROp_StringLit:
                                SLANG_UNIMPLEMENTED_X("String constants in SPIR-V emit");
                            default:
                                SLANG_UNREACHABLE("Unhandled case in emitSPIRVAsm");
                            }
                            emitOperand(constant);
                        }
                        else
                        {
                            switch (v->getOp())
                            {
                            case kIROp_StringLit:
                                emitOperand(
                                    SpvLiteralBits::fromUnownedStringSlice(v->getStringSlice()));
                                break;
                            case kIROp_IntLit:
                                {
                                    // TODO: range checking
                                    const auto i = cast<IRIntLit>(v)->getValue();
                                    emitOperand(SpvLiteralInteger::from32(uint32_t(i)));
                                    break;
                                }
                            default:
                                SLANG_UNREACHABLE("Unhandled case in emitSPIRVAsm");
                            }
                        }
                        break;
                    }
                case kIROp_SPIRVAsmOperandInst:
                    {
                        const auto i = operand->getValue();
                        emitOperand(ensureInst(i));

                        break;
                    }
                case kIROp_SPIRVAsmOperandResult:
                    {
                        SLANG_ASSERT(isLast);
                        emitOperand(kResultID);
                        break;
                    }
                case kIROp_SPIRVAsmOperandId:
                    {
                        const auto idName =
                            cast<IRStringLit>(operand->getValue())->getStringSlice();
                        SpvWord id;
                        if (!idMap.tryGetValue(idName, id))
                        {
                            id = freshID();
                            idMap.set(idName, id);
                        }
                        emitOperand(id);
                        break;
                    }
                case kIROp_SPIRVAsmOperandSampledType:
                    {
                        // Make a 4 vector of the component type
                        IRBuilder builder(m_irModule);
                        const auto elementType = cast<IRType>(operand->getValue());
                        const auto sampledType = builder.getVectorType(
                            getSPIRVSampledElementType(getVectorElementType(elementType)),
                            4);
                        emitOperand(ensureInst(sampledType));
                        break;
                    }
                case kIROp_SPIRVAsmOperandImageType:
                case kIROp_SPIRVAsmOperandSampledImageType:
                    {
                        IRBuilder builder(m_irModule);
                        auto textureInst =
                            as<IRTextureTypeBase>(operand->getValue()->getDataType());
                        auto imageType = builder.getTextureType(
                            textureInst->getElementType(),
                            textureInst->getShapeInst(),
                            textureInst->getIsArrayInst(),
                            textureInst->getIsMultisampleInst(),
                            textureInst->getSampleCountInst(),
                            textureInst->getAccessInst(),
                            textureInst->getIsShadowInst(),
                            builder.getIntValue(
                                builder.getIntType(),
                                (operand->getOp() == kIROp_SPIRVAsmOperandSampledImageType ? 1
                                                                                           : 0)),
                            textureInst->getFormatInst());
                        emitOperand(ensureInst(imageType));
                        break;
                    }
                case kIROp_SPIRVAsmOperandBuiltinVar:
                    {
                        emitOperand(ensureInst(operand));
                        break;
                    }
                case kIROp_SPIRVAsmOperandGLSL450Set:
                    {
                        emitOperand(getGLSL450ExtInst());
                        break;
                    }
                case kIROp_SPIRVAsmOperandDebugPrintfSet:
                    {
                        emitOperand(getNonSemanticDebugPrintfExtInst());
                        break;
                    }
                default:
                    SLANG_UNREACHABLE("Unhandled case in emitSPIRVAsm");
                }
            };

            if (spvInst->getOpcodeOperand()->getOp() == kIROp_SPIRVAsmOperandTruncate)
            {
                const auto getSlangType = [&](IRSPIRVAsmOperand* operand) -> IRType*
                {
                    switch (operand->getOp())
                    {
                    case kIROp_SPIRVAsmOperandInst:
                        return cast<IRType>(operand->getValue());
                    case kIROp_SPIRVAsmOperandSampledType:
                        {
                            // Make a 4 vector of the component type
                            IRBuilder builder(m_irModule);
                            const auto elementType = cast<IRType>(operand->getValue());
                            return builder.getVectorType(getVectorElementType(elementType), 4);
                        }
                    case kIROp_SPIRVAsmOperandEnum:
                    case kIROp_SPIRVAsmOperandLiteral:
                    case kIROp_SPIRVAsmOperandResult:
                    case kIROp_SPIRVAsmOperandId:
                        SLANG_UNEXPECTED("truncate should have been given slang types");
                    default:
                        SLANG_UNREACHABLE("Unhandled case in emitSPIRVAsm");
                    }
                };

                SLANG_ASSERT(spvInst->getSPIRVOperands().getCount() == 4);
                const auto toType = getSlangType(spvInst->getSPIRVOperands()[0]);
                const auto toIdOperand = spvInst->getSPIRVOperands()[1];
                const auto fromType = getSlangType(spvInst->getSPIRVOperands()[2]);
                const auto fromIdOperand = spvInst->getSPIRVOperands()[3];
                auto fromElementType = getSPIRVSampledElementType(fromType);
                SpvInst* fromSpvInst = nullptr;
                // If the component types are not the same, convert them to be so.
                if (!isTypeEqual(getVectorElementType(toType), fromElementType))
                {
                    SpvOp convertOp =
                        isIntegralType(fromElementType)
                            ? (isSignedType(fromElementType) ? SpvOpSConvert : SpvOpUConvert)
                            : SpvOpFConvert;
                    auto newFromType =
                        replaceVectorElementType(fromType, getVectorElementType(toType));
                    fromSpvInst = emitInstCustomOperandFunc(
                        parent,
                        nullptr,
                        convertOp,
                        [&]()
                        {
                            emitOperand(newFromType);
                            emitOperand(kResultID), emitSpvAsmOperand(fromIdOperand);
                        });
                }

                // If we don't need truncation, but a different result ID is
                // expected, then just unify them in the idMap
                if (isTypeEqual(toType, fromType))
                {
                    // TODO: if this is the last inst, we should just remove it
                    // and rewrite the penultimate one
                    last = emitInstCustomOperandFunc(
                        parent,
                        isLast ? as<IRInst>(inst) : spvInst,
                        SpvOpCopyObject,
                        [&]()
                        {
                            emitOperand(toType);
                            emitSpvAsmOperand(toIdOperand);
                            fromSpvInst ? emitOperand(fromSpvInst)
                                        : emitSpvAsmOperand(fromIdOperand);
                        });
                }
                // Otherwise, if we are truncating to a scalar, extract the first element
                else if (!as<IRVectorType>(toType))
                {
                    last = emitInstCustomOperandFunc(
                        parent,
                        isLast ? as<IRInst>(inst) : spvInst,
                        SpvOpCompositeExtract,
                        [&]()
                        {
                            emitOperand(toType);
                            emitSpvAsmOperand(toIdOperand);
                            fromSpvInst ? emitOperand(fromSpvInst)
                                        : emitSpvAsmOperand(fromIdOperand);
                            emitOperand(SpvLiteralInteger::from32(0));
                        });
                }
                // Otherwise, if we are truncating to a 1-vector from a scalar
                else if (as<IRVectorType>(toType) && !as<IRVectorType>(fromType))
                {
                    last = emitInstCustomOperandFunc(
                        parent,
                        isLast ? as<IRInst>(inst) : spvInst,
                        SpvOpCompositeConstruct,
                        [&]()
                        {
                            emitOperand(toType);
                            emitSpvAsmOperand(toIdOperand);
                            fromSpvInst ? emitOperand(fromSpvInst)
                                        : emitSpvAsmOperand(fromIdOperand);
                        });
                }
                // Otherwise, we are truncating a vector to a smaller vector
                else
                {
                    const auto toVector = cast<IRVectorType>(unwrapAttributedType(toType));
                    const auto toVectorSize = getIntVal(toVector->getElementCount());
                    const auto fromVector = cast<IRVectorType>(unwrapAttributedType(fromType));
                    const auto fromVectorSize = getIntVal(fromVector->getElementCount());
                    if (toVectorSize > fromVectorSize)
                        m_sink->diagnose(inst, Diagnostics::spirvInvalidTruncate);
                    last = emitInstCustomOperandFunc(
                        parent,
                        isLast ? as<IRInst>(inst) : spvInst,
                        SpvOpVectorShuffle,
                        [&]()
                        {
                            emitOperand(toType);
                            emitSpvAsmOperand(toIdOperand);
                            fromSpvInst ? emitOperand(fromSpvInst)
                                        : emitSpvAsmOperand(fromIdOperand);
                            emitOperand(emitOpUndef(parent, nullptr, fromVector));
                            for (Int32 i = 0; i < toVectorSize; ++i)
                                emitOperand(SpvLiteralInteger::from32(i));
                        });
                }
            }
            else
            {
                const SpvOp opcode = SpvOp(spvInst->getOpcodeOperandWord());

                switch (opcode)
                {
                case SpvOpCapability:
                    requireSPIRVCapability(
                        (SpvCapability)getIntVal(spvInst->getOperand(1)->getOperand(0)));
                    continue;
                case SpvOpExtension:
                    ensureExtensionDeclaration(
                        as<IRStringLit>(spvInst->getOperand(1)->getOperand(0))->getStringSlice());
                    continue;
                case SpvOpExecutionMode:
                    {
                        if (auto refEntryPointSet =
                                m_referencingEntryPoints.tryGetValue(getParentFunc(inst)))
                        {
                            for (auto entryPoint : *refEntryPointSet)
                            {
                                emitInstMemoizedNoResultIDCustomOperandFunc(
                                    getSection(SpvLogicalSectionID::ExecutionModes),
                                    nullptr,
                                    SpvOpExecutionMode,
                                    [&]()
                                    {
                                        emitOperand(entryPoint);
                                        for (UInt s = 2; s < spvInst->getOperandCount(); s++)
                                            emitSpvAsmOperand(
                                                as<IRSPIRVAsmOperand>(spvInst->getOperand(s)));
                                    });
                            }
                        }
                        continue;
                    }
                default:
                    break;
                }
                const auto opParent = parentForOpCode(opcode, parent);
                const auto opInfo = m_grammarInfo->opInfos.lookup(opcode);

                // TODO: handle resultIdIndex == 1, for constants
                const bool memoize =
                    opParent == getSection(SpvLogicalSectionID::ConstantsAndTypes) && opInfo &&
                    opInfo->resultIdIndex == 0;

                // We want the "result instruction" to refer to the top level
                // block which assumes its value, the others are free to refer
                // to whatever, so just use the internal spv inst rep
                // TODO: This is not correct, because the instruction which is
                // assigned to result is not necessarily the last instruction
                const auto assignedInst = isLast ? as<IRInst>(inst) : spvInst;

                if (memoize)
                {
                    last = emitInstMemoizedCustomOperandFunc(
                        opParent,
                        assignedInst,
                        opcode,
                        kResultID,
                        [&]()
                        {
                            Index i = 0;
                            for (const auto operand : spvInst->getSPIRVOperands())
                            {
                                if (i++ != 0)
                                    emitSpvAsmOperand(operand);
                            };
                        });

                    // The result operand is the one at index 1, after the
                    // opcode itself.
                    // If this happens to be an "id" operand, then we need to
                    // correct the Id we have stored in our map with the actual
                    // memoized result. This is safe because a condition on
                    // memoized instructions is that they come before their
                    // uses.
                    const auto resOperand = cast<IRSPIRVAsmOperand>(spvInst->getOperand(1));
                    if (resOperand->getOp() == kIROp_SPIRVAsmOperandId)
                    {
                        const auto idName =
                            cast<IRStringLit>(resOperand->getValue())->getStringSlice();
                        idMap[idName] = last->id;
                    }
                }
                else
                {
                    last = emitInstCustomOperandFunc(
                        opParent,
                        assignedInst,
                        opcode,
                        [&]()
                        {
                            for (const auto operand : spvInst->getSPIRVOperands())
                                emitSpvAsmOperand(operand);
                        });
                }
            }
        }

        for (const auto& [name, id] : idMap)
            emitOpName(getSection(SpvLogicalSectionID::DebugNames), nullptr, id, name);

        return last;
    }

    OrderedHashSet<SpvCapability> m_capabilities;
    void requireSPIRVCapability(SpvCapability capability)
    {
        if (m_capabilities.add(capability))
        {
            emitOpCapability(getSection(SpvLogicalSectionID::Capabilities), nullptr, capability);
        }
    }

    List<List<SpvCapability>> m_anyCapability;
    void requireSPIRVAnyCapability(List<SpvCapability> capabilities)
    {
        if (!m_anyCapability.contains(capabilities))
        {
            m_anyCapability.add(capabilities);
        }
    }

    void emitSPIRVAnyCapabilities()
    {
        for (const auto& options : m_anyCapability)
        {
            bool found = false;
            for (SpvCapability option : options)
            {
                if (m_capabilities.contains(option))
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                requireSPIRVCapability(options[0]);
            }
        }
    }

    void requireVariableBufferCapabilityIfNeeded(IRInst* type)
    {
        if (auto ptrType = as<IRPtrTypeBase>(type))
        {
            if (ptrType->getAddressSpace() == AddressSpace::StorageBuffer)
            {
                ensureExtensionDeclaration(UnownedStringSlice("SPV_KHR_variable_pointers"));
                requireSPIRVCapability(SpvCapabilityVariablePointers);
            }
        }
    }

    // https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpExecutionMode
    Dictionary<SpvWord, OrderedHashSet<SpvExecutionMode>> m_executionModes;
    template<typename... Operands>
    void requireSPIRVExecutionMode(
        IRInst* parentInst,
        SpvWord entryPoint,
        SpvExecutionMode executionMode,
        const Operands&... ops)
    {
        if (m_executionModes[entryPoint].add(executionMode))
        {
            SpvOp execModeOp = SpvOpExecutionMode;
            if (executionMode == SpvExecutionModeLocalSizeId ||
                executionMode == SpvExecutionModeLocalSizeHintId ||
                executionMode == SpvExecutionModeSubgroupsPerWorkgroupId)
            {
                execModeOp = SpvOpExecutionModeId;
            }

            emitInst(
                getSection(SpvLogicalSectionID::ExecutionModes),
                parentInst,
                execModeOp,
                entryPoint,
                executionMode,
                ops...);
        }
    }

    // Applies execution mode to entry points that reference `childFunc`.
    template<typename... Operands>
    void requireSPIRVExecutionModeOnReferencingEntryPoints(
        IRInst* parentInst,
        IRFunc* childFunc,
        SpvExecutionMode executionMode,
        const Operands&... ops)
    {

        if (auto entryPointsUsingInst =
                getReferencingEntryPoints(m_referencingEntryPoints, childFunc))
        {
            for (IRFunc* entryPoint : *entryPointsUsingInst)
            {
                requireSPIRVExecutionMode(
                    parentInst,
                    getIRInstSpvID(entryPoint),
                    executionMode,
                    ops...);
            }
        }
    }

    SPIRVEmitContext(IRModule* module, TargetProgram* program, DiagnosticSink* sink)
        : SPIRVEmitSharedContext(module, program, sink), m_irModule(module), m_memoryArena(2048)
    {
    }
};

SlangResult emitSPIRVFromIR(
    CodeGenContext* codeGenContext,
    IRModule* irModule,
    const List<IRFunc*>& irEntryPoints,
    List<uint8_t>& spirvOut)
{
    spirvOut.clear();

    bool symbolsEmitted = false;

    auto sink = codeGenContext->getSink();

#if 0
    {
        DiagnosticSinkWriter writer(codeGenContext->getSink());
        dumpIR(
            irModule,
            { IRDumpOptions::Mode::Simplified, 0 },
            "BEFORE SPIR-V LEGALIZE",
            codeGenContext->getSourceManager(),
            &writer);
    }
#endif

    SPIRVEmitContext context(irModule, codeGenContext->getTargetProgram(), sink);
    legalizeIRForSPIRV(&context, irModule, irEntryPoints, codeGenContext);

#if 0
    {
        DiagnosticSinkWriter writer(codeGenContext->getSink());
        dumpIR(
            irModule,
            { IRDumpOptions::Mode::Simplified, 0 },
            "BEFORE SPIR-V EMIT",
            codeGenContext->getSourceManager(),
            &writer);
    }
#endif

    removeAvailableInDownstreamModuleDecorations(CodeGenTarget::SPIRV, irModule);

    auto shouldPreserveParams = codeGenContext->getTargetProgram()->getOptionSet().getBoolOption(
        CompilerOptionName::PreserveParameters);
    auto generateWholeProgram = codeGenContext->getTargetProgram()->getOptionSet().getBoolOption(
        CompilerOptionName::GenerateWholeProgram);
    for (auto inst : irModule->getGlobalInsts())
    {
        if (as<IRDebugSource>(inst))
        {
            context.ensureInst(inst);
        }
        if (shouldPreserveParams && as<IRGlobalParam>(inst))
        {
            context.ensureInst(inst);
            symbolsEmitted = true;
        }
        if (generateWholeProgram)
        {
            if (auto func = as<IRFunc>(inst))
            {
                if (func->findDecoration<IRDownstreamModuleExportDecoration>())
                {
                    context.ensureInst(inst);
                    symbolsEmitted = true;
                }
            }
        }
    }

    // Emit source language info.
    // By default we will use SpvSourceLanguageSlang.
    // However this will cause problems when using swiftshader.
    // To workaround this problem, we allow overriding this behavior with an
    // environment variable that will be set in the software testing environment.
    auto sourceLanguage = SpvSourceLanguageSlang;
    StringBuilder noSlangEnv;
    PlatformUtil::getEnvironmentVariable(
        toSlice("SLANG_USE_SPV_SOURCE_LANGUAGE_UNKNOWN"),
        noSlangEnv);
    if (noSlangEnv.produceString() == "1")
    {
        sourceLanguage = SpvSourceLanguageUnknown;
    }
    context.emitInst(
        context.getSection(SpvLogicalSectionID::DebugStringsAndSource),
        nullptr,
        SpvOpSource,
        SpvLiteralInteger::from32(
            sourceLanguage),           // language identifier, should be SpvSourceLanguageSlang.
        SpvLiteralInteger::from32(1)); // language version.

    for (auto irEntryPoint : irEntryPoints)
    {
        context.ensureInst(irEntryPoint);
        symbolsEmitted = true;
    }

    if (!symbolsEmitted)
    {
        sink->diagnose(irModule->getModuleInst(), Diagnostics::outputSpvIsEmpty);
        return SLANG_FAIL;
    }

    // Move forward delcared pointers to the end.
    do
    {
        auto fwdPointers = context.m_forwardDeclaredPointers;
        context.m_forwardDeclaredPointers.clear();

        for (auto ptrType : fwdPointers)
        {
            auto spvPtrType = ptrType.key;
            // When we emit a pointee type, we may introduce new
            // forward-declared pointer types, so we need to
            // keep iterating until we have emitted all of them.
            context.ensureInst(ptrType.value->getValueType());
            auto parent = spvPtrType->parent;
            spvPtrType->removeFromParent();
            parent->addInst(spvPtrType);
        }
    } while (context.m_forwardDeclaredPointers.getCount() != 0);

    // Emit extensions and capabilities for which there are multiple options available.
    // This is delayed to avoid emitting unnecessary extensions and capabilities if
    // one of the options is already required by some other op.
    context.emitSPIRVAnyExtension();
    context.emitSPIRVAnyCapabilities();

    context.emitFrontMatter();

    context.emitPhysicalLayout();

    spirvOut.addRange(
        (uint8_t const*)context.m_words.getBuffer(),
        context.m_words.getCount() * Index(sizeof(context.m_words[0])));

    return SLANG_OK;
}


} // namespace Slang
