// slang-ir-restructure.h
#pragma once

#include "../core/slang-basic.h"
#include "slang-ir-insts.h"

namespace Slang
{
class DiagnosticSink;
struct IRBlock;
struct IRGlobalValueWithCode;
struct IRInst;
struct IRLoop;

/// A structured control-flow region.
///
/// A `Region` is used to layer structured control flow information
/// over an existing IR control flow graph (CFG). Each `Region`
/// represents a sub-graph of the CFG such that control always
/// enters at the start of the region.
///
class Region : public RefObject
{
public:
    enum class Flavor
    {
        Simple,
        If,
        Break,
        Continue,
        Loop,
        Switch,
    };

    Flavor getFlavor() { return flavor; }

    Region* getParent() { return parent; }

    /// Is this region a descendent of `other`?
    ///
    /// For the purpose of this query, a region
    /// is a descendent of itself.
    bool isDescendentOf(Region* other);

    /// Is this region a descendent of `block`?
    ///
    /// This tests is the region is a descendent
    /// of any simple region for `block`.
    bool isDescendentOf(IRBlock* block);

protected:
    Region(Flavor flavor, Region* parent)
        : flavor(flavor), parent(parent)
    {
    }

    /// What kind of region is this?
    Flavor flavor;

    /// The parent region of this region.
    Region* parent;
};

/// Base type for regions that have a "next" region.
///
/// While we think of it as a region to execute
/// after this region, the `nextRegion` is actually
/// a *child* region, in that it can see local
/// values that were defined in this parent region
/// (and any other ancestor regions).
class SeqRegion : public Region
{
protected:
    SeqRegion(Flavor flavor, Region* parent)
        : Region(flavor, parent)
    {
    }

public:
    /// The (child) region to execute after this one.
    RefPtr<Region> nextRegion;
};

/// A simple region that encapsulates a basic block.
///
class SimpleRegion : public SeqRegion
{
public:
    SimpleRegion(Region* parent, IRBlock* block)
        : SeqRegion(Region::Flavor::Simple, parent), block(block)
    {
    }

    /// The basic block for this region.
    IRBlock* block = nullptr;

    /// The next simple region for the same block
    ///
    /// A single IR basic block may turn into multiple regions,
    /// if the restructuring pass has to duplicate it (this
    /// currently happens for the continue clause in a `for`
    /// loop if it has multiple `continue` sites.
    ///
    SimpleRegion* nextSimpleRegionForSameBlock = nullptr;
};

/// A conditional region, corresponding to an `if`
///
class IfRegion : public SeqRegion
{
public:
    IfRegion(Region* parent, IRIfElse* ifElseInst)
        : SeqRegion(Region::Flavor::If, parent), ifElseInst(ifElseInst)
    {
    }

    /// The IR `ifElse` instruction
    IRIfElse* ifElseInst;

    IRInst* getCondition() { return ifElseInst->getCondition(); }

    /// The region to execute if the `condition` is `true`
    RefPtr<Region> thenRegion;

    /// The region to execute if the `condition` is `false`
    RefPtr<Region> elseRegion;
};

/// Base type for regions that execution can `break` out of
class BreakableRegion : public SeqRegion
{
protected:
    BreakableRegion(Flavor flavor, Region* parent)
        : SeqRegion(flavor, parent)
    {
    }
};

/// A region that expresses a `break` out of nested control flow.
///
class BreakRegion : public Region
{
public:
    BreakRegion(Region* parent, BreakableRegion* outerRegion)
        : Region(Region::Flavor::Break, parent), outerRegion(outerRegion)
    {
    }

    BreakableRegion* outerRegion;
};

/// A structured loop
class LoopRegion : public BreakableRegion
{
public:
    LoopRegion(Region* parent, IRLoop* loopInst)
        : BreakableRegion(Region::Flavor::Loop, parent), loopInst(loopInst)
    {
    }

    /// The IR instruction that represents the branch into the loop.
    /// We keep this instruction around because it may have decorations
    /// that need to influence how we emit this loop.
    ///
    IRLoop* loopInst;

    /// The code inside the loop.
    ///
    /// The body region may include `break` or `continue` operations for this loop.
    RefPtr<Region> body;
};

/// A region that expresses a `continue` for a structured loop.
///
class ContinueRegion : public Region
{
public:
    ContinueRegion(Region* parent, LoopRegion* outerRegion)
        : Region(Region::Flavor::Continue, parent), outerRegion(outerRegion)
    {
    }

    LoopRegion* outerRegion;
};

/// A structured `switch` statement.
class SwitchRegion : public BreakableRegion
{
public:
    SwitchRegion(Region* parent, IRSwitch* switchInst)
        : BreakableRegion(Region::Flavor::Switch, parent), switchInst(switchInst)
    {
    }

    /// The IR `switch` instruction
    IRSwitch* switchInst;

    IRInst* getCondition() { return switchInst->getCondition(); }

    /// A collection of `case`s that share the same code.
    class Case : public RefObject
    {
    public:
        /// The various values that should branch to this case.
        ///
        /// It is possible for this list to be empty if this
        /// is the `default` case and has no explicit values
        /// that map to it.
        ///
        List<IRInst*> values;

        /// The region to execute if this case is selected.
        RefPtr<Region> body;
    };

    /// All of the cases for the `switch`.
    ///
    /// This includes any `default` cases.
    ///
    /// As an invariant, a case that "falls through" to another
    /// should immediately precede its target in this list.
    ///
    List<RefPtr<Case>> cases;

    /// The default case, if any.
    ///
    /// It is valid for this to be `null` if there is no `default` case,
    /// in which case the default behavior should be to branch to the region
    /// after the `switch`.
    ///
    /// The default case must also be present in `cases`.
    Case* defaultCase = nullptr;
};

/// Container for all of the regions in a function.
///
/// A `RegionTree` owns the `Region` objects associated with a function,
/// along with a mapping from basic blocks in the IR function to regions
/// in the tree.
///
class RegionTree : public RefObject
{
public:
    /// Type for the mapping from IR blocks to regions.
    typedef Dictionary<IRBlock*, SimpleRegion*> MapBlockToRegion;

    /// A dictionary to map from IR blocks to regions.
    MapBlockToRegion mapBlockToRegion;

    /// The root region of the region tree.
    RefPtr<Region> rootRegion;

    /// The IR function that was used to compute the region tree.
    IRGlobalValueWithCode* irCode = nullptr;
};

/// Construct structrured regions to represent the control flow in an IR function.
///
/// The resulting `RegionTree` will encode a structured (statement-like)
/// form for the control flow graph (CFG) of `code`.
/// In cases where our current restructuring approach is not powerful
/// enough to handle something in the input CFG, diagnostic messages
/// will be output to the given `sink`.
///
RefPtr<RegionTree> generateRegionTreeForFunc(IRGlobalValueWithCode* code, DiagnosticSink* sink);
} // namespace Slang
