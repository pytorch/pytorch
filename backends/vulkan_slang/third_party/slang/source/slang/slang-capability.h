// slang-capability.h
#pragma once

#include "../core/slang-dictionary.h"
#include "../core/slang-list.h"
#include "../core/slang-string.h"

#include <optional>
#include <stdint.h>

namespace Slang
{

// This file defines a system for reasoning about the "capabilities" that a
// target supports or, conversely, the capabilities that a function or other
// symbol requires.
//
// The central idea is that we can think of the each of these cases as a set,
// where the elements of the set are atomic features that are either present
// on a target or not (no in-between states). For example, an atomic feature
// might be used to represent support for double-precision floating-point
// operations. When compiling for a target, we need to know whether the
// target supports double-precision or not, and for a particular function
// it either requires double-precision math to run, or not.
//
// In this system, the atomic capabilities are represented as cases of
// the `CapabilityAtom` enumeration, which is generated from declarations
// in the `slang-capability-defs.h` file.
//
#include "slang-generated-capability-defs.h"

// Once we have a universe of suitable capability atoms, we can define
// the capabilities of a target as simply the set of all atomic capabilities
// that it supports.
//
// The situation is slightly more complicated for a function. A function
// might require a specific set of atomic feature, and that is the simple
// case. In this simple case, we know that a target can run a function
// if the features of the target are a super-set of those required by
// the function.
//
// In the more general case, we might have a function that can be used
// with multiple different combinations of features: e.g., you can use
// the function if your target supports features A and B, or if it supports
// features C and D. In our representation, that case is handled by
// assocaiting multiple distinct sets of capabilities with one declaration,
// with each set expressing one way that the declaration can be legally used.
//
// In all cases, we represent a set of capabilities with `CapabilitySet`.

struct CapabilityAtomSet : UIntSet
{
    using UIntSet::UIntSet;

    CapabilityAtomSet newSetWithoutImpliedAtoms() const;
};

struct CapabilityTargetSet;
typedef Dictionary<CapabilityAtom, CapabilityTargetSet> CapabilityTargetSets;

/// CapabilityStageSet encapsulates all capabilities of a specific shader stage for a specific
/// target. Capabilities may be disjoint, but only in rare cases:
/// {{glsl, _GLSL_130, GL_EXT_FOO1}, {glsl, _GLSL_130, _GLSL_140, _GLSL_150}}
struct CapabilityStageSet
{
    CapabilityAtom stage{};

    /// LinkedList of all disjoint sets for fast remove/add of unconstrained list positions.
    std::optional<CapabilityAtomSet> atomSet{};

    void addNewSet(CapabilityAtomSet&& setToAdd)
    {
        if (!atomSet)
            atomSet = setToAdd;
        else
            atomSet->add(setToAdd);
    }

    /// Join `this` with a compatble stage set of `CapabilityTargetSet other`.
    /// Return false when `other` is fully incompatible.
    /// incompatability is when `this->stage` is not a supported stage by `other.shaderStageSets`.
    bool tryJoin(const CapabilityTargetSet& other);
};

/// CapabilityTargetSet encapsulates all capabilities of a specific target
/// Format: {shader_stage, shader_stage_set}
typedef Dictionary<CapabilityAtom, CapabilityStageSet> CapabilityStageSets;
struct CapabilityTargetSet
{
    CapabilityAtom target{};

    CapabilityStageSets shaderStageSets{};

    /// Join a compatable target set from `this` with `CapabilityTargetSet other`.
    /// Return false when `other` is fully incompatible.
    /// incompatability is when one of 2 senarios are true:
    /// 1. `this->target` is not a supported target by `other.shaderStageSets`
    /// 2. `this` has completly disjoint shader stages from other.
    bool tryJoin(const CapabilityTargetSets& other);
    void unionWith(const CapabilityTargetSet& other);
};

struct CapabilitySet
{
public:
    /// Default-construct an empty capability set
    CapabilitySet();

    CapabilitySet(CapabilitySet const& other) = default;
    CapabilitySet& operator=(CapabilitySet const& other) = default;
    CapabilitySet(CapabilitySet&& other) = default;
    CapabilitySet& operator=(CapabilitySet&& other) = default;

    /// Construct a capability set from an explicit list of atomic capabilities
    CapabilitySet(Int atomCount, CapabilityName const* atoms);

    /// Construct a capability set from an explicit list of atomic capabilities
    explicit CapabilitySet(List<CapabilityName> const& atoms);

    /// Construct a singleton set from a single atomic capability
    explicit CapabilitySet(CapabilityName atom);

    /// Make an empty capability set
    static CapabilitySet makeEmpty();

    /// Make an invalid capability set (such that no target could ever support it)
    static CapabilitySet makeInvalid();

    /// Is this capability set empty (such that any target supports it)?
    bool isEmpty() const;

    /// Is this capability set invalid (such that no target could support it)?
    bool isInvalid() const;

    /// Is this capability set incompatible with the given `other` set.
    bool isIncompatibleWith(CapabilityAtom other) const;

    /// Is this capability set incompatible with the given `other` set.
    bool isIncompatibleWith(CapabilityName other) const;

    /// Is this capability set incompatible with the given `other` atomic capability.
    bool isIncompatibleWith(CapabilitySet const& other) const;

    enum class ImpliesReturnFlags : int
    {
        NotImplied = 0,
        Implied = 1 << 0,
    };
    /// Does this capability set imply all the capabilities in `other`?
    bool implies(CapabilitySet const& other) const;
    /// Does this capability set imply at least 1 set in other.
    ImpliesReturnFlags atLeastOneSetImpliedInOther(CapabilitySet const& other) const;

    /// Does this capability set imply the atomic capability `other`?
    bool implies(CapabilityAtom other) const;

    /// Join two capability sets to form ('this' & 'other').
    /// Destroy incompatible targets/sets apart of 'this' between ('this' & 'other').
    /// `this` may be made invalid if other is fully disjoint.
    void join(const CapabilitySet& other);

    /// Join two capability sets to form ('this' & 'other').
    /// If a target/set has an incompatible atom, do not destroy the target/set.
    void nonDestructiveJoin(const CapabilitySet& other);

    /// Add all targets/sets of 'other' into 'this'. Overlapping sets are removed.
    void unionWith(const CapabilitySet& other);

    /// Return a capability set of 'target' atoms 'this' has, but 'other' does not.
    CapabilitySet getTargetsThisHasButOtherDoesNot(const CapabilitySet& other);

    /// Return a capability set of 'stage' atoms 'this' has, but 'other' does not.
    CapabilitySet getStagesThisHasButOtherDoesNot(const CapabilitySet& other);

    /// Are these two capability sets equal?
    bool operator==(CapabilitySet const& that) const;

    void addCapability(List<List<CapabilityAtom>>& atomLists);
    /// Calculate a list of "compacted" atoms, which excludes any atoms from the expanded list that
    /// are implies by another item in the list.

    /// returns true if 'this' is a better target for 'targetCaps' than 'that'
    /// isEqual: is `this` and `that` equal
    bool isBetterForTarget(
        CapabilitySet const& that,
        CapabilitySet const& targetCaps,
        bool& isEqual) const;

    /// Find any capability sets which are in 'available' but not in 'required'. Return false if
    /// this situation occurs.
    static bool checkCapabilityRequirement(
        CapabilitySet const& available,
        CapabilitySet const& required,
        CapabilityAtomSet& outFailedAvailableSet);

    // For each element in `elementsToPermutateWith`, create and add a different conjunction
    // permutation by adding to `setToPermutate`.
    template<CapabilityName keyholeAtomToPermuteWith>
    void addPermutationsOfConjunctionForEachInContainer(
        CapabilityAtomSet& setToPermutate,
        const CapabilityAtomSet& elementsToPermutateWith,
        CapabilityAtom knownTargetAtom,
        CapabilityAtom knownStageAtom);
    // This is used for adding conjunctions directly and efficently, this is not functionally a
    // join. if `knownStage`/`knownTarget` is not CapabilityAtom::Invalid, the given atom will be
    // assumed as an assigned key atom (faster)
    inline void addConjunction(
        CapabilityAtomSet conjunction,
        CapabilityAtom knownTarget,
        CapabilityAtom knownStage);
    inline void addUnexpandedCapabilites(CapabilityName atom);

    CapabilityTargetSets& getCapabilityTargetSets() { return m_targetSets; }
    const CapabilityTargetSets& getCapabilityTargetSets() const { return m_targetSets; }

    // If this capability set uniquely implies one stage atom, return it. Otherwise returns
    // CapabilityAtom::Invalid.
    CapabilityAtom getUniquelyImpliedStageAtom() const;

    struct AtomSets
    {
        struct Iterator
        {
        private:
            const CapabilityTargetSets* context;
            CapabilityTargetSets::ConstIterator targetNode{};
            CapabilityStageSets::ConstIterator stageNode{};
            const std::optional<CapabilityAtomSet>* atomSetNode = {};

        public:
            operator bool() const { return (atomSetNode) ? atomSetNode->has_value() : false; }
            const CapabilityAtomSet& operator*() const { return *(*this->atomSetNode); }
            const CapabilityAtomSet* operator->() const { return &(*(*this->atomSetNode)); }
            bool operator==(const Iterator& other) const
            {
                return other.context == this->context && other.targetNode == this->targetNode &&
                       other.stageNode == this->stageNode;
            }
            bool operator!=(const Iterator& other) const { return !(other == *this); }

            Iterator& operator++()
            {
                for (;;)
                {
                    this->stageNode++;
                    if (this->stageNode == (*this->targetNode).second.shaderStageSets.end())
                    {
                        for (;;)
                        {
                            this->targetNode++;
                            if (this->targetNode == this->context->end())
                            {
                                this->stageNode = {};
                                this->atomSetNode = {};
                                return *this;
                            }
                            this->stageNode = (*this->targetNode).second.shaderStageSets.begin();
                            if (this->stageNode == (*this->targetNode).second.shaderStageSets.end())
                                continue;
                            break;
                        }
                    }
                    if (!(*this->stageNode).second.atomSet)
                        continue;
                    this->atomSetNode = &(*this->stageNode).second.atomSet;
                    break;
                }
                return *this;
            }
            Iterator& operator++(int) { return ++(*this); }
            Iterator begin() const
            {
                Iterator tmp(this->context);
                tmp.targetNode = this->context->begin();
                if (tmp.targetNode == this->context->end())
                    return tmp;
                tmp.stageNode = (*tmp.targetNode).second.shaderStageSets.begin();
                while (tmp.stageNode == (*tmp.targetNode).second.shaderStageSets.end())
                {
                    tmp.targetNode++;
                    if (tmp.targetNode == this->context->end())
                        return end();
                    tmp.stageNode = (*tmp.targetNode).second.shaderStageSets.begin();
                }
                tmp.atomSetNode = &(*tmp.stageNode).second.atomSet;
                if (!tmp.atomSetNode->has_value())
                    tmp++;
                return tmp;
            }
            Iterator end() const
            {
                Iterator tmp(this->context);
                tmp.targetNode = this->context->end();
                return tmp;
            }
            Iterator(const CapabilityTargetSets* mainContext) { context = mainContext; }
        };
    };
    /// Get access to the raw atomic capabilities that define this set.
    /// Get all bottom level UIntSets for each CapabilityTargetSet.
    CapabilitySet::AtomSets::Iterator getAtomSets() const;

    /// Add spirv version capabilities from 'spirv CapabilityTargetSet' as glsl_spirv version
    /// capability in 'glsl CapabilityTargetSet'
    void addSpirvVersionFromOtherAsGlslSpirvVersion(CapabilitySet& other);

    /// Gets the first valid compile-target found in the CapabilitySet
    CapabilityAtom getCompileTarget()
    {
        if (isEmpty() || isInvalid())
            return CapabilityAtom::Invalid;
        return (*m_targetSets.begin()).first;
    }

    /// Gets the first valid stage found in the CapabilitySet
    CapabilityAtom getTargetStage()
    {
        if (isEmpty() || isInvalid())
            return CapabilityAtom::Invalid;
        return (*(*m_targetSets.begin()).second.shaderStageSets.begin()).first;
    }

private:
    /// underlying data of CapabilitySet.
    CapabilityTargetSets m_targetSets{};

    void addCapability(CapabilityName name);

    bool hasSameTargets(const CapabilitySet& other) const;

    enum class ImpliesFlags
    {
        None = 0,
        OnlyRequireASingleValidImply = 1 << 0,
    };
    ImpliesReturnFlags _implies(CapabilitySet const& other, ImpliesFlags flags) const;
};

/// Returns true if atom is derived from base
bool isCapabilityDerivedFrom(CapabilityAtom atom, CapabilityAtom base);

/// Find a capability atom with the given `name`, or return CapabilityAtom::Invalid.
CapabilityName findCapabilityName(UnownedStringSlice const& name);

/// Check if 'name' is an '_Internal' or 'External' capability.
bool isInternalCapabilityName(CapabilityName name);

CapabilityAtom getLatestSpirvAtom();
CapabilityAtom getLatestMetalAtom();

/// For debug purposes ensure a casted CapabilityAtom is valid
template<typename T>
inline CapabilityAtom asAtom(T name)
{
    SLANG_ASSERT((UInt)name < (UInt)CapabilityAtom::Count);
    return CapabilityAtom(name);
}

/// Gets the capability names.
void getCapabilityNames(List<UnownedStringSlice>& ioNames);

UnownedStringSlice capabilityNameToString(CapabilityName name);

bool isDirectChildOfAbstractAtom(CapabilityAtom name);
bool isStageAtom(CapabilityName name, CapabilityName& outCanonicalStage);

/// Return true if `name` represents an atom for a target version, e.g. spirv_1_5.
bool isTargetVersionAtom(CapabilityAtom name);
bool isSpirvExtensionAtom(CapabilityAtom name);

void printDiagnosticArg(StringBuilder& sb, CapabilityAtom atom);
void printDiagnosticArg(StringBuilder& sb, CapabilityName name);

const CapabilityAtomSet& getAtomSetOfTargets();
const CapabilityAtomSet& getAtomSetOfStages();

bool hasTargetAtom(const CapabilityAtomSet& setIn, CapabilityAtom& targetAtom);

void freeCapabilityDefs();

// #define UNIT_TEST_CAPABILITIES
#ifdef UNIT_TEST_CAPABILITIES
void TEST_CapabilitySet();
#endif

} // namespace Slang
