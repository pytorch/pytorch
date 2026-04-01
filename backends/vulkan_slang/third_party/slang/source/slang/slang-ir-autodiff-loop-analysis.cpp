// slang-ir-autodiff-loop-analysis.cpp

#include "slang-ir-autodiff-loop-analysis.h"

namespace Slang
{

static bool isCompareCmpInst(IRInst* inst)
{
    // Switch on the opcode of the instruction
    switch (inst->getOp())
    {
    case kIROp_Less:
    case kIROp_Greater:
    case kIROp_Leq:
    case kIROp_Geq:
    case kIROp_Eql:
    case kIROp_Neq:
        return true;
    default:
        return false;
    }
}

SimpleRelation mergeEqualityWithIntegerRelation(SimpleRelation equality, SimpleRelation relation)
{
    SLANG_ASSERT(
        equality.type == SimpleRelation::IntegerRelation &&
        relation.type == SimpleRelation::IntegerRelation);
    SLANG_ASSERT(equality.comparator == SimpleRelation::Equal);

    switch (relation.comparator)
    {
    case SimpleRelation::Equal:
        if (relation.integerValue == equality.integerValue)
            return relation;
        break; // Technically we'd want to return a "set" here, but we don't have a representation
               // for that.
    case SimpleRelation::LessThanEqual:
        if (equality.integerValue <= relation.integerValue)
            return relation;
        break;
    case SimpleRelation::GreaterThanEqual:
        if (equality.integerValue >= relation.integerValue)
            return relation;
        break;
    default:
        break;
    }

    return SimpleRelation::anyRelation();
}

SimpleRelation mergeIntervals(SimpleRelation a, SimpleRelation b)
{
    SLANG_ASSERT(
        a.type == SimpleRelation::IntegerRelation && b.type == SimpleRelation::IntegerRelation);

    if (a.comparator == SimpleRelation::Equal)
    {
        return mergeEqualityWithIntegerRelation(a, b);
    }
    else if (b.comparator == SimpleRelation::Equal)
    {
        return mergeEqualityWithIntegerRelation(b, a);
    }

    // TODO: Handle other cases...
    return SimpleRelation::anyRelation();
}

// Returns the tighest "simple" relation such that (a v b -> result)
//
// Note: "simple" means that the relation is not a disjunction or conjunction of other relations.
//
SimpleRelation relationUnion(SimpleRelation a, SimpleRelation b)
{
    // Base case. The disjunction operator is idempotent.
    if (a == b)
        return a;

    // If either side is trivially true, the result is trivially true.
    if (a.type == SimpleRelation::Any || b.type == SimpleRelation::Any)
        return SimpleRelation::anyRelation();

    // If either side is trivially false, then the result is the other relation.
    if (a.type == SimpleRelation::Impossible)
        return b;

    if (b.type == SimpleRelation::Impossible)
        return a;

    // If one is the negated form of the other, there's really nothing we can prove, since
    // A OR ~A is always true.
    //
    if (a.negated() == b)
        return SimpleRelation::anyRelation();

    // Handle the case of where one is an inequality and the other is an equality.
    if (a.type == SimpleRelation::IntegerRelation && b.type == SimpleRelation::IntegerRelation)
        return mergeIntervals(a, b);

    // TODO: Here's where we can handle subset cases like (a < 10) and (a < 20) => (a < 20), etc..
    // But we don't _have_ to. The more we can prove, the more cases we can handle, but the result
    // is still correct without it.
    //

    // Default to not being able to say anything.
    return SimpleRelation::anyRelation();
}

SimpleRelation intersectEqualityWithIntegerRelation(
    SimpleRelation equality,
    SimpleRelation relation)
{
    SLANG_ASSERT(equality.type == SimpleRelation::IntegerRelation);
    SLANG_ASSERT(relation.type == SimpleRelation::IntegerRelation);
    SLANG_ASSERT(equality.comparator == SimpleRelation::Equal);

    if (relation.comparator == SimpleRelation::Equal)
    {
        if (relation.integerValue == equality.integerValue)
            return SimpleRelation::integerRelation(SimpleRelation::Equal, equality.integerValue);
        else
            return SimpleRelation::impossibleRelation();
    }
    else if (relation.comparator == SimpleRelation::LessThanEqual)
    {
        if (equality.integerValue <= relation.integerValue)
            return SimpleRelation::integerRelation(
                SimpleRelation::LessThanEqual,
                relation.integerValue);
        else
            return SimpleRelation::impossibleRelation();
    }
    else if (relation.comparator == SimpleRelation::GreaterThanEqual)
    {
        if (equality.integerValue >= relation.integerValue)
            return SimpleRelation::integerRelation(
                SimpleRelation::GreaterThanEqual,
                relation.integerValue);
        else
            return SimpleRelation::impossibleRelation();
    }
    else if (relation.comparator == SimpleRelation::NotEqual)
    {
        if (equality.integerValue != relation.integerValue)
            return SimpleRelation::integerRelation(SimpleRelation::NotEqual, relation.integerValue);
        else
            return SimpleRelation::impossibleRelation();
    }

    return SimpleRelation::anyRelation();
}

// Intersect intervals.
SimpleRelation intersectIntervals(SimpleRelation a, SimpleRelation b)
{
    SLANG_ASSERT(
        a.type == SimpleRelation::IntegerRelation && b.type == SimpleRelation::IntegerRelation);

    if (a.comparator == SimpleRelation::Equal)
    {
        return intersectEqualityWithIntegerRelation(a, b);
    }
    else if (b.comparator == SimpleRelation::Equal)
    {
        return intersectEqualityWithIntegerRelation(b, a);
    }

    // TODO: Handle other cases...

    // We'll just default to picking the first one, since (a ^ b) -> a is always true.
    return a;
}

// Returns the best "simple" relation such that (a ^ b -> result)
//
SimpleRelation relationIntersection(SimpleRelation a, SimpleRelation b)
{
    // Base case. The conjunction operator is idempotent.
    if (a == b)
        return a;

    // If one is the negated form of the other, then we can prove that the result is impossible.
    // Doesn't necessarily mean that we have an error on our hands, but it does mean that whatever
    // case we're considering can't happen, so can be ignored (unreachable)
    //
    if (a.negated() == b)
        return SimpleRelation::impossibleRelation();

    // If any of the relations is impossible, then the result is impossible.
    if (a.type == SimpleRelation::Impossible || b.type == SimpleRelation::Impossible)
        return SimpleRelation::impossibleRelation();

    // If any one of the relations is trivially true, then the result is the other relation.
    if (a.type == SimpleRelation::Any)
        return b;

    if (b.type == SimpleRelation::Any)
        return a;

    //
    // We'll handle the cases where one is an equality and the other is an inequality or equality.
    //
    // i.e. For a conjunction (a == 10) ^ (a < 20), we can use the narrower relation (a == 10).
    //
    if (a.type == SimpleRelation::IntegerRelation && b.type == SimpleRelation::IntegerRelation)
        return intersectIntervals(a, b);

    // TODO: Handle other cases...
    return SimpleRelation::anyRelation();
}

void StatementSet::disjunct(StatementSet other)
{
    // false v (a1 v a2 v a3 ...) = (a1 v a2 v a3 ...)
    if (isTriviallyFalse())
    {
        statements = other.statements;
        return;
    }

    // (a1 v a2 v a3 ...) v false = (a1 v a2 v a3 ...)
    if (other.isTriviallyFalse())
        return;

    // true v (a1 v a2 v a3 ...) = true
    if (other.isTriviallyTrue())
    {
        statements.clear();
        return;
    }

    // (a1 v a2 v a3 ...) v true = true
    if (isTriviallyTrue())
        return;

    for (auto& statement : other.statements)
    {
        // Since we hold only one statement per inst, we can perform disjunction
        // on a per-inst basis.
        // If an inst does not exist in the current set, then it's an empty statement.
        //
        if (statements.containsKey(statement.first))
        {
            auto newRelation = relationUnion(statement.second, statements[statement.first]);
            set(statement.first, newRelation);
        }
    }

    // Remove any insts that don't have a corresponding statement in the other set,
    // since this effectively means "any".
    //
    statements.removeIf([&](auto const& statement)
                        { return !other.statements.containsKey(statement.first); });
}

void StatementSet::conjunct(StatementSet other)
{
    // true ^ (a1 ^ a2 ^ a3 ...) = (a1 ^ a2 ^ a3 ...)
    if (other.isTriviallyTrue())
        return;

    // (a1 ^ a2 ^ a3 ...) ^ true = (a1 ^ a2 ^ a3 ...)
    if (isTriviallyTrue())
    {
        statements = other.statements;
        return;
    }

    // false ^ (a1 ^ a2 ^ a3 ...) = false
    if (isTriviallyFalse())
        return;

    // (a1 ^ a2 ^ a3 ...) ^ false = false
    if (other.isTriviallyFalse())
    {
        statements = other.statements;
        return;
    }

    // Otherwise do an element-wise conjunction.
    for (auto& statement : other.statements)
    {
        if (statements.containsKey(statement.first))
        {
            set(statement.first,
                relationIntersection(statement.second, statements[statement.first]));
        }
        else
        {
            set(statement.first, statement.second);
        }
    }
}

void StatementSet::conjunct(IRInst* inst, SimpleRelation relation)
{
    if (isTriviallyFalse())
        return;

    if (statements.containsKey(inst))
    {
        set(inst, relationIntersection(relation, statements[inst]));
    }
    else
    {
        set(inst, relation);
    }
}

// This function answers the question: "Can we prove that relationB is true if relationA is true?"
//
// Note that this is not the same as "Does relationA imply relationB", since there can be cases
// where this is indeed true, but we just don't have the logic to prove it.
//
bool doesRelationImply(SimpleRelation relationA, SimpleRelation relationB)
{
    // Equal relations imply each other
    if (relationA == relationB)
        return true;

    // If B is trivially true, then A implies B
    if (relationB.type == SimpleRelation::Any)
        return true;

    // If A is trivially true, then A implies B only if B is also trivially true
    if (relationA.type == SimpleRelation::Any)
        return (relationB.type == SimpleRelation::Any);

    // If A is impossible, then technically what we return doesn't matter...
    if (relationA.type == SimpleRelation::Impossible ||
        relationB.type == SimpleRelation::Impossible)
        return false;

    // If A is a boolean relation, then A implies B if B is also a boolean relation and the values
    // are the same.
    //
    if (relationA.type == SimpleRelation::BoolRelation)
        return (relationB.type == SimpleRelation::BoolRelation) &&
               (relationA.boolValue == relationB.boolValue);

    if (relationA.type == SimpleRelation::IntegerRelation)
    {
        if (relationB.type != SimpleRelation::IntegerRelation)
            return false;

        // Technically, the equality case is already handled above, so we'll only consider
        // cases where A and B are not the same relation, but where A -> B

        // If A is an equality, and B is an inequality, we can test
        if (relationA.comparator == SimpleRelation::Equal)
        {
            if (relationB.comparator == SimpleRelation::LessThanEqual)
                return relationA.integerValue <= relationB.integerValue;
            else if (relationB.comparator == SimpleRelation::GreaterThanEqual)
                return relationA.integerValue >= relationB.integerValue;
        }

        // If A is an equality, and B is an inequality with different values, then
        // A -> B
        //
        if (relationA.comparator == SimpleRelation::Equal &&
            relationB.comparator == SimpleRelation::NotEqual)
        {
            return relationA.integerValue != relationB.integerValue;
        }

        if (relationA.comparator == SimpleRelation::GreaterThanEqual &&
            relationB.comparator == SimpleRelation::GreaterThanEqual)
        {
            return relationA.integerValue >= relationB.integerValue;
        }

        if (relationA.comparator == SimpleRelation::LessThanEqual &&
            relationB.comparator == SimpleRelation::LessThanEqual)
        {
            return relationA.integerValue <= relationB.integerValue;
        }

        // TODO: Handle other cases.. these come up rarely, so we can
    }

    return false;
}

bool isIntegerConstantValue(IRInst* inst)
{
    return inst->getOp() == kIROp_IntLit;
}

bool isBoolConstantValue(IRInst* inst)
{
    return inst->getOp() == kIROp_BoolLit;
}

IRIntegerValue getConstantIntegerValue(IRInst* inst)
{
    SLANG_ASSERT(isIntegerConstantValue(inst));
    return as<IRIntLit>(inst)->getValue();
}

bool getConstantBoolValue(IRInst* inst)
{
    SLANG_ASSERT(isBoolConstantValue(inst));
    return as<IRBoolLit>(inst)->getValue();
}

StatementSet tryExtractStatements(IRTerminatorInst* inst, IRBlock* block)
{
    StatementSet statements;

    // From condInst, extract a statement about any inst such that we have an equality
    // statement (integer or boolean) on the inst.
    //
    if (auto ifElse = as<IRIfElse>(inst))
    {
        // Check that the block is the true or false block of the if-else
        bool isTrueBlock = ifElse->getTrueBlock() == block;
        bool isFalseBlock = ifElse->getFalseBlock() == block;
        if (!isTrueBlock && !isFalseBlock)
            goto done;

        auto condInst = inst->getOperand(0);
        statements.conjunct(condInst, SimpleRelation::boolRelation(isTrueBlock));

        if (condInst->getOp() == kIROp_Eql)
        {
            auto leftOperand = condInst->getOperand(0);
            auto rightOperand = condInst->getOperand(1);

            if (isIntegerConstantValue(leftOperand))
            {
                statements.conjunct(
                    rightOperand,
                    SimpleRelation::integerRelation(
                        (isTrueBlock ? SimpleRelation::Equal : SimpleRelation::NotEqual),
                        getConstantIntegerValue(leftOperand)));
            }
            else if (isIntegerConstantValue(rightOperand))
            {
                statements.conjunct(
                    leftOperand,
                    SimpleRelation::integerRelation(
                        (isTrueBlock ? SimpleRelation::Equal : SimpleRelation::NotEqual),
                        getConstantIntegerValue(rightOperand)));
            }
        }
        else if (isCompareCmpInst(condInst))
        {
            auto leftOperand = condInst->getOperand(0);
            auto rightOperand = condInst->getOperand(1);

            bool isParamLeft = !isIntegerConstantValue(leftOperand);
            bool isParamRight = !isIntegerConstantValue(rightOperand);

            // If neither operand is an inst, we can't say anything.
            if (!isParamLeft && !isParamRight)
                goto done;

            auto paramOperand = isParamLeft ? leftOperand : rightOperand;
            auto otherOperand = isParamLeft ? rightOperand : leftOperand;

            // Check if the "other" operand is a constant
            if (!isIntegerConstantValue(otherOperand))
                goto done;

            auto constantVal = getConstantIntegerValue(otherOperand);

            SimpleRelation::Comparator comparator;
            switch (condInst->getOp())
            {
            case kIROp_Less:
                comparator = SimpleRelation::LessThanEqual;
                constantVal = constantVal - 1;
                break;
            case kIROp_Greater:
                comparator = SimpleRelation::GreaterThanEqual;
                constantVal = constantVal + 1;
                break;
            case kIROp_Leq:
                comparator = SimpleRelation::LessThanEqual;
                break;
            case kIROp_Geq:
                comparator = SimpleRelation::GreaterThanEqual;
                break;
            case kIROp_Eql:
                comparator = SimpleRelation::Equal;
                break;
            case kIROp_Neq:
                comparator = SimpleRelation::NotEqual;
                break;
            default:
                SLANG_UNREACHABLE("unexpected op code");
            }
            auto relation = SimpleRelation::integerRelation(comparator, constantVal);
            statements.conjunct(
                paramOperand,
                ((isParamLeft ^ !isTrueBlock) ? relation : relation.negated()));
        }
    }
    else if (auto switchInst = as<IRSwitch>(inst))
    {
        // Check that the block is the default case of the switch
        if (switchInst->getDefaultLabel() == block)
            goto done;

        // Check each case block
        UInt caseCount = switchInst->getCaseCount();
        for (UInt i = 0; i < caseCount; i++)
        {
            auto caseValue = switchInst->getCaseValue(i);
            auto caseBlock = switchInst->getCaseLabel(i);

            if (caseBlock == block && isIntegerConstantValue(caseValue))
            {
                auto constantVal = getConstantIntegerValue(caseValue);
                statements.conjunct(
                    switchInst->getCondition(),
                    SimpleRelation::integerRelation(SimpleRelation::Equal, constantVal));
            }
        }
    }

done:
    return statements;
}

enum class BlockStateFlags
{
    UpwardPropCompleted = 1 << 0,
    DownwardPropCompleted = 1 << 1
};

void markUpwardPropCompleted(IRBlock* block)
{
    block->scratchData |= (UInt64)BlockStateFlags::UpwardPropCompleted;
}

void markDownwardPropCompleted(IRBlock* block)
{
    block->scratchData |= (UInt64)BlockStateFlags::DownwardPropCompleted;
}

bool isUpwardPropCompleted(IRBlock* block)
{
    return block->scratchData & (UInt64)BlockStateFlags::UpwardPropCompleted;
}

bool isDownwardPropCompleted(IRBlock* block)
{
    return block->scratchData & (UInt64)BlockStateFlags::DownwardPropCompleted;
}

void clearBlockState(IRBlock* block)
{
    block->scratchData = 0;
}

bool isLoopConditionBlock(IRBlock* block)
{
    for (auto use = block->firstUse; use; use = use->nextUse)
    {
        if (auto loop = as<IRLoop>(use->getUser()))
        {
            if (loop->getTargetBlock() == block)
                return true;
        }
    }

    return false;
}

bool isBlockReadyForUpwardProp(IRBlock* block)
{
    if (isLoopConditionBlock(block))
    {
        auto falseBlock = cast<IRIfElse>(block->getTerminator())->getFalseBlock();
        return isUpwardPropCompleted(falseBlock);
    }

    // Check that successors have completed upward propagation.
    for (auto successor : block->getSuccessors())
    {
        if (!isUpwardPropCompleted(successor))
            return false;
    }
    return true;
}

bool isBlockReadyForDownwardProp(IRBlock* block)
{
    // Check that predecessors have completed downward propagation.
    for (auto predecessor : block->getPredecessors())
    {
        if (!isDownwardPropCompleted(predecessor))
            return false;
    }
    return true;
}

StatementSet propagateStatementUpwards(IRInst* inst, SimpleRelation relation)
{
    // Lambda to make a single-statement set.
    auto makeStatementSet = [&](IRInst* inst, SimpleRelation relation)
    {
        StatementSet set;
        set.conjunct(inst, relation);
        return set;
    };

    if (as<IRParam>(inst))
        return makeStatementSet(inst, relation);

    if (isIntegerConstantValue(inst))
    {
        auto relationFromInst =
            SimpleRelation::integerRelation(SimpleRelation::Equal, getConstantIntegerValue(inst));
        if (doesRelationImply(relation, relationFromInst))
            return makeStatementSet(inst, SimpleRelation::anyRelation()); // Trivially true
        else if (doesRelationImply(relation, relationFromInst.negated()))
            return makeStatementSet(inst, SimpleRelation::impossibleRelation());
        else
            return makeStatementSet(inst, SimpleRelation::anyRelation());
    }
    else if (isBoolConstantValue(inst))
    {
        auto relationFromInst = SimpleRelation::boolRelation(getConstantBoolValue(inst));
        if (doesRelationImply(relation, relationFromInst))
            return makeStatementSet(inst, SimpleRelation::anyRelation()); // Trivially true
        else if (doesRelationImply(relation, relationFromInst.negated()))
            return makeStatementSet(inst, SimpleRelation::impossibleRelation());
        else
            return makeStatementSet(inst, SimpleRelation::anyRelation());
    }
    else if (inst->getOp() == kIROp_Add || inst->getOp() == kIROp_Sub)
    {
        // TODO: Translate equality/inequality.
    }

    return makeStatementSet(inst, SimpleRelation::anyRelation());
}

StatementSet propagateUpwards(
    RefPtr<IRDominatorTree> domTree,
    IRBlock* current,
    IRBlock* predecessor,
    StatementSet predicateSet)
{
    // Translate the set of predicates from the current block to the predecessor block.
    //
    // The key idea is that we need to find a set of predicate statements (A') for the predecessor
    // block, such that A => A'.
    //
    // During the downward phase, the predecessor will then return a set of
    // statements (B') such that A' => B'. This B' can be propagated "downwards" into a set
    // of statements B such that B' => B.
    //
    // We can then combine these three rules A => A', A' => B' and B' => B to get A => B
    // which is the statement set that we want for our current block.
    //

    StatementSet newPredicateSet;
    for (auto& statementInstPair : predicateSet.statements)
    {
        auto predicateRelation = statementInstPair.second;
        auto predicateInst = statementInstPair.first;
        if (as<IRParam>(predicateInst) && predicateInst->getParent() == current)
        {
            auto paramIndex = getParamIndexInBlock(cast<IRParam>(predicateInst));
            auto translatedInst =
                as<IRUnconditionalBranch>(predecessor->getTerminator())->getArg(paramIndex);

            // If the translate inst is outside the block, add it in as-is, otherwise,
            // we'll need to propagate it to the operands of the inst
            //
            auto statementSet = propagateStatementUpwards(translatedInst, predicateRelation);
            newPredicateSet.conjunct(statementSet);
        }
        else
        {
            newPredicateSet.conjunct(predicateInst, predicateRelation);
        }
    }

    // If our current block is a merge block for a conditional branch, we should add the condition
    // to the predicate set.
    //
    for (auto blockUse = current->firstUse; blockUse; blockUse = blockUse->nextUse)
    {
        if (auto ifElse = as<IRIfElse>(blockUse->getUser()))
        {
            if (ifElse->getAfterBlock() == current)
            {
                // We're looking at the merge block for a conditional branch.

                if (domTree->dominates(ifElse->getTrueBlock(), predecessor))
                {
                    // True branch
                    newPredicateSet.conjunct(tryExtractStatements(ifElse, ifElse->getTrueBlock()));
                }
                else if (domTree->dominates(ifElse->getFalseBlock(), predecessor))
                {
                    // False branch
                    newPredicateSet.conjunct(tryExtractStatements(ifElse, ifElse->getFalseBlock()));
                }
                else
                {
                    // It's possible that the predecessor block is the condition block itself (when
                    // either the true side or the false side is empty).
                    //
                    if (predecessor == ifElse->getParent() && ifElse->getFalseBlock() == current)
                    {
                        // True branch
                        newPredicateSet.conjunct(
                            tryExtractStatements(ifElse, ifElse->getFalseBlock()));
                    }
                    else if (
                        predecessor == ifElse->getParent() && ifElse->getTrueBlock() == current)
                    {
                        // False branch
                        newPredicateSet.conjunct(
                            tryExtractStatements(ifElse, ifElse->getTrueBlock()));
                    }
                    else
                    {

                        // Panic
                        SLANG_UNREACHABLE("Unreachable block in conditional branch");
                    }
                }
            }
        }

        // We'll ignore switch statements for now, but they're trivial to add.
        // TODO: Add switch statements.
    }

    // We have one more edge-case. The condition block of a loop inst.
    if (auto ifElse = as<IRIfElse>(current->getTerminator()))
    {
        if (domTree->dominates(ifElse->getTrueBlock(), predecessor) &&
            !domTree->dominates(ifElse->getFalseBlock(), predecessor))
        {
            // True branch
            newPredicateSet.conjunct(tryExtractStatements(ifElse, ifElse->getTrueBlock()));
        }
    }
    return newPredicateSet;
}

StatementSet propagateStatementDownwards(
    IRInst* srcInst,
    IRInst* dstInst,
    StatementSet srcStatements)
{
    // We'll keep translating through the inst, until we either hit a parameter
    // until we either hit a parameter, or we leave the current block.
    //

    // Lambda to make a single-statement set.
    auto singleStatement = [&](IRInst* inst, SimpleRelation relation)
    {
        StatementSet set;
        set.conjunct(inst, relation);
        return set;
    };

    if (srcStatements.statements.containsKey(srcInst))
        return singleStatement(dstInst, srcStatements.statements[srcInst]);

    if (isIntegerConstantValue(srcInst))
    {
        return singleStatement(
            dstInst,
            SimpleRelation::integerRelation(
                SimpleRelation::Equal,
                getConstantIntegerValue(srcInst)));
    }
    else if (isBoolConstantValue(srcInst))
    {
        return singleStatement(
            dstInst,
            SimpleRelation::boolRelation(getConstantBoolValue(srcInst)));
    }

    if (srcInst->getOp() == kIROp_Add || srcInst->getOp() == kIROp_Sub)
    {
        auto left = srcInst->getOperand(0);
        auto right = srcInst->getOperand(1);

        auto isLeftConstant = isIntegerConstantValue(left);
        auto isRightConstant = isIntegerConstantValue(right);

        if (!isLeftConstant && !isRightConstant)
            return singleStatement(dstInst,
                                   SimpleRelation::anyRelation()); // Can't say anything

        if (srcInst->getOp() == kIROp_Add || (srcInst->getOp() == kIROp_Sub && isRightConstant))
        {
            auto constant =
                isLeftConstant ? getConstantIntegerValue(left) : getConstantIntegerValue(right);
            auto operand = isLeftConstant ? right : left;

            constant = srcInst->getOp() == kIROp_Add ? constant : -constant;

            auto operandStatement = propagateStatementDownwards(operand, operand, srcStatements);
            auto relation = operandStatement.statements.containsKey(operand)
                                ? operandStatement.statements[operand]
                                : SimpleRelation::anyRelation();

            if (relation.type == SimpleRelation::IntegerRelation)
            {
                switch (relation.comparator)
                {
                case SimpleRelation::Equal:
                case SimpleRelation::NotEqual:
                case SimpleRelation::LessThanEqual:
                case SimpleRelation::GreaterThanEqual:
                    return singleStatement(
                        dstInst,
                        SimpleRelation::integerRelation(
                            relation.comparator,
                            constant + relation.integerValue));
                }
            }
        }
    }

    // Default
    return singleStatement(dstInst, SimpleRelation::anyRelation());
}

StatementSet propagateDownwards(
    RefPtr<IRDominatorTree> domTree,
    IRBlock* successor,
    IRBlock* predecessor,
    StatementSet statementSet)
{
    // Translate a set of statements from the current block to the successor block.
    //
    // That is, find a set of statements (B') for the successor block such that B => B'
    //
    StatementSet newStatementSet;

    if (statementSet.isTriviallyFalse())
    {
        return statementSet;
    }

    // Go over all the parameters of the successor block, find corresponding arguments, and
    // convert any statements to the new set.
    //
    UInt paramIndex = 0;
    for (auto param : successor->getParams())
    {
        auto arg = as<IRUnconditionalBranch>(predecessor->getTerminator())->getArg(paramIndex);
        auto statement = propagateStatementDownwards(arg, param, statementSet);
        newStatementSet.conjunct(statement);
        paramIndex++;
    }

    newStatementSet.conjunct(tryExtractStatements(predecessor->getTerminator(), successor));

    // For all other statements in the statementSet, we'll add them in, but only
    // if the predecessor dominates the successor.
    // An exception is parameters defined in the successor (since these are getting
    // redefined, we should not be considering existing statements)
    //
    for (auto& statement : statementSet.statements)
    {
        if (domTree->dominates(statement.first->getParent(), successor) &&
            !(as<IRParam>(statement.first) && statement.first->getParent() == successor))
            newStatementSet.conjunct(statement.first, statement.second);
    }

    return newStatementSet;
}

struct Edge
{
    IRBlock* predecessor;
    IRBlock* successor;

    bool operator==(const Edge& other) const
    {
        return predecessor == other.predecessor && successor == other.successor;
    }

    UInt64 getHashCode() const
    {
        UInt64 predHash = Slang::getHashCode(predecessor);
        UInt64 succHash = Slang::getHashCode(successor);
        return Slang::combineHash(predHash, succHash);
    }
};


// This routine returns a set of implications for any insts visible in a block.
//
// The process uses a modified version of abstract interpretation, by first propagating a set
// of predicates "backwards" repeatedly through the predecessors, then calculating the set of
// implications "forwards" repeatedly through the successors.
//
// Note that the resulting implications don't contain all possible statements that could be inferred
// statically (this is an undeciable problem), but rather whatever can be inferred in just two steps
// through the blocks. This suffices for the vast majority of common loop structures.
//
StatementSet collectImplications(
    RefPtr<IRDominatorTree> domTree,
    IRBlock* block,
    StatementSet Predicates)
{
    List<Edge> orderedEdgeList; // Edges in the order that they're processed.
    HashSet<Edge> falseEdges; // Edges between blocks where the successor's predicate does not imply
                              // the predecessor's predicate.

    // Initialize a work list.
    List<IRBlock*> workList;
    workList.add(block);

    // Clear scratch bits.
    IRFunc* func = cast<IRFunc>(domTree->code);
    for (auto _block : func->getBlocks())
    {
        clearBlockState(_block);
    }

    //
    // Upward pass: Propagate predicates through predecessors until
    // there're no more blocks left to process.
    //

    // We'll keep track of the predicates for each block.
    Dictionary<IRBlock*, StatementSet> blockPredicates;

    blockPredicates[block] = Predicates;

    while (workList.getCount() > 0)
    {
        auto current = workList.getLast();
        workList.removeLast();

        // If the block has already been processed, skip it.
        if (isUpwardPropCompleted(current))
            continue;

        // If the block is not ready for upward propagation, add it to the work list.
        if (current != block && !isBlockReadyForUpwardProp(current))
        {
            workList.add(current);
            // Then add all the successors to the work list.
            for (auto successor : current->getSuccessors())
                workList.add(successor);

            continue;
        }

        // Otherwise, we'll process the block.
        //
        // Get our predicate set, then propagate it to all predecessors.
        //
        auto predicates = blockPredicates[current];

        HashSet<IRBlock*> uniquePredecessors;
        for (auto predecessor : current->getPredecessors())
            uniquePredecessors.add(predecessor);

        for (auto predecessor : uniquePredecessors)
        {
            // We also need to handle the recursive case, where the predecessor
            // is already "sealed".
            //
            if (isUpwardPropCompleted(predecessor))
            {
                orderedEdgeList.add({predecessor, current});

                // Verify that "current predicate" => "predecessor predicate".

                // TODO: Implement later.
                // For now, we can default to assuming that this edge is not
                // valid. This works fine since we're not trying to prove anything recursive (like
                // inductivity), but we should revisit this if we do want to unify the induction
                // value inference pass with this loop analysis system.
                //

                // We'll add this to the set of false edges so that the downward prop pass
                // doesn't propagate any implications through this edge.
                //
                falseEdges.add({predecessor, current});
                continue;
            }

            auto newPredicates = propagateUpwards(domTree, current, predecessor, predicates);

            if (!blockPredicates.containsKey(predecessor))
                blockPredicates[predecessor] = newPredicates;
            else
                blockPredicates[predecessor].disjunct(newPredicates);

            orderedEdgeList.add({predecessor, current});

            // Add predecessors to work list.
            workList.add(predecessor);
        }

        markUpwardPropCompleted(current);
    }

    //
    // Downward pass: Propagate implications through successors until
    // there're no more blocks left to process.
    //

    Dictionary<IRBlock*, StatementSet> blockImplications;

    // Set 'block' to something trivial base case.
    // blockImplications[block] = blockPredicates[block]; // statement => statement

    while (orderedEdgeList.getCount() > 0)
    {
        auto edge = orderedEdgeList.getLast();
        orderedEdgeList.removeLast();

        // Get the predicate set for the predecessor.
        auto predecessorPredicates = blockPredicates[edge.predecessor];

        // Get the implication set for the predecessor.
        auto predecessorImplications = StatementSet();

        if (falseEdges.contains(edge))
        {
            // Since A' => B' is not true, effectively, we can't say anything..
            predecessorImplications = StatementSet();
        }
        else
        {
            // (A' => B') => (A' => A' ^ B')
            predecessorImplications = blockImplications[edge.predecessor];
            predecessorImplications.conjunct(predecessorPredicates);
        }

        // Propagate the implication set to the successor.
        auto successorImplications =
            propagateDownwards(domTree, edge.successor, edge.predecessor, predecessorImplications);

        if (!blockImplications.containsKey(edge.successor))
            blockImplications[edge.successor] = successorImplications;
        else
            blockImplications[edge.successor].disjunct(successorImplications);
    }

    // Clear scratch bits.
    for (auto _block : func->getBlocks())
    {
        clearBlockState(_block);
    }

    // We should have a final set of implications for our block.
    return blockImplications[block];
}

} // namespace Slang
