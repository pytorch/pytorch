// slang-ir-autodiff-loop-analysis.h
#pragma once

#include "slang-ir-autodiff-region.h"
#include "slang-ir-autodiff.h"
#include "slang-ir-dominators.h"
#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{
struct SimpleRelation
{
    enum Type
    {
        Any,             // Target can be anything (all values are possible)
        IntegerRelation, // Target satisfies a simple integer equality/inequality
        BoolRelation,    // Target satisfies boolean equality
        Impossible       // Target is impossible (has no possible values)
    } type;

    enum Comparator
    {
        LessThanEqual,
        GreaterThanEqual,
        Equal,
        NotEqual
    } comparator;
    IRIntegerValue integerValue;
    bool boolValue;

    static SimpleRelation integerRelation(Comparator comparator, IRIntegerValue integerValue)
    {
        return SimpleRelation{IntegerRelation, comparator, integerValue, false};
    }

    static SimpleRelation boolRelation(bool boolValue)
    {
        return SimpleRelation{BoolRelation, Equal, 0, boolValue};
    }

    static SimpleRelation impossibleRelation()
    {
        return SimpleRelation{Impossible, Equal, 0, false};
    }

    static SimpleRelation anyRelation() { return SimpleRelation{Any, Equal, 0, false}; }

    bool operator==(const SimpleRelation& other) const
    {
        switch (type)
        {
        case Any:
            return other.type == Any;
        case IntegerRelation:
            return other.type == IntegerRelation && comparator == other.comparator &&
                   integerValue == other.integerValue;
        case BoolRelation:
            return other.type == BoolRelation && boolValue == other.boolValue;
        case Impossible:
            return other.type == Impossible;
        default:
            SLANG_UNREACHABLE("Unhandled relation type");
        }
    }

    bool operator!=(const SimpleRelation& other) const { return !(*this == other); }

    SimpleRelation negated() const
    {
        switch (type)
        {
        case Any:
            return SimpleRelation{Impossible, Equal, 0, false};
        case Impossible:
            return SimpleRelation{Any, Equal, 0, false};
        case BoolRelation:
            return SimpleRelation{BoolRelation, Equal, 0, !boolValue};
        case IntegerRelation:
            switch (comparator)
            {
            case LessThanEqual:
                return SimpleRelation{IntegerRelation, GreaterThanEqual, integerValue + 1, false};
            case GreaterThanEqual:
                return SimpleRelation{IntegerRelation, LessThanEqual, integerValue - 1, false};
            case Equal:
                return SimpleRelation{IntegerRelation, NotEqual, integerValue, false};
            case NotEqual:
                return SimpleRelation{IntegerRelation, Equal, integerValue, false};
            default:
                SLANG_UNREACHABLE("Unhandled comparator");
            }
        default:
            SLANG_UNREACHABLE("Unhandled relation type");
        }
    }

    HashCode64 getHashCode() const
    {
        HashCode64 code = Slang::getHashCode(int(type));
        switch (type)
        {
        case IntegerRelation:
            code = combineHash(code, Slang::getHashCode(comparator));
            code = combineHash(code, Slang::getHashCode(integerValue));
            break;
        case BoolRelation:
            code = combineHash(code, Slang::getHashCode(boolValue));
            break;
        case Impossible:
        case Any:
            break;
        default:
            SLANG_UNREACHABLE("Unhandled relation type");
        }
        return code;
    }
};

struct StatementSet
{
    // A conjunction of independent statements (a1 ^ a2 ^ a3 ...)
    // - One simple relation per inst.
    // - The absence of an entry implies that the inst is unconstrained.
    // - The presence of any "Impossible" relation indicates that the entire conjunction is always
    // false.
    //
    Dictionary<IRInst*, SimpleRelation> statements;

    // Disjunction of a conjunction of statements (a1 ^ a2 ^ a3 ...) with the current conjunction.
    void disjunct(StatementSet other);

    // Conjunction of a conjunction of statements (a1 ^ a2 ^ a3 ...) with the current conjunction.
    void conjunct(StatementSet other);

    // Conjunction of a single statement with the current conjunction.
    void conjunct(IRInst* inst, SimpleRelation relation);

    void set(IRInst* inst, SimpleRelation relation)
    {
        if (relation.type == SimpleRelation::Any)
        {
            if (statements.containsKey(inst))
                statements.remove(inst);
            return;
        }

        statements[inst] = relation;
    }

    bool isTriviallyFalse()
    {
        for (auto& statement : statements)
        {
            if (statement.second.type == SimpleRelation::Impossible)
                return true;
        }
        return false;
    }

    bool isTriviallyTrue() { return statements.getCount() == 0; }
};


// Utility functions.
bool isIntegerConstantValue(IRInst* inst);
bool isBoolConstantValue(IRInst* inst);
IRIntegerValue getConstantIntegerValue(IRInst* inst);
bool getConstantBoolValue(IRInst* inst);

bool doesRelationImply(SimpleRelation relationA, SimpleRelation relationB);

// Try to collect a set of implications for any insts visible in a block,
// subject to the set of predicates.
//
StatementSet collectImplications(
    RefPtr<IRDominatorTree> domTree,
    IRBlock* block,
    StatementSet Predicates);

} // namespace Slang
