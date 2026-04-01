// slang-ir-inst-pass-base.h
#pragma once

#include "slang-ir-dce.h"
#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{
struct IRModule;

class InstPassBase
{
protected:
    IRModule* module;
    InstWorkList workList;
    InstHashSet workListSet;
    void addToWorkList(IRInst* inst)
    {
        SLANG_ASSERT(inst);
        if (workListSet.contains(inst))
            return;

        workList.add(inst);
        workListSet.add(inst);
    }

    IRInst* pop(bool removeFromSet = true)
    {
        if (workList.getCount() == 0)
            return nullptr;

        IRInst* inst = workList.getLast();
        workList.removeLast();
        if (removeFromSet)
            workListSet.remove(inst);
        return inst;
    }

public:
    InstPassBase(IRModule* inModule)
        : module(inModule), workList(inModule), workListSet(inModule)
    {
    }

    template<typename InstType, typename Func>
    void processInstsOfType(IROp instOp, const Func& f)
    {
        workList.clear();
        workListSet.clear();

        addToWorkList(module->getModuleInst());

        while (workList.getCount() != 0)
        {
            IRInst* inst = pop();

            if (inst->getOp() == instOp)
            {
                f(as<InstType>(inst));
            }

            for (auto child = inst->getLastChild(); child; child = child->getPrevInst())
            {
                addToWorkList(child);
            }
        }
    }

    template<typename InstType, typename Func>
    void processChildInstsOfType(IROp instOp, IRInst* parent, const Func& f)
    {
        workList.clear();
        workListSet.clear();

        addToWorkList(parent);

        while (workList.getCount() != 0)
        {
            IRInst* inst = pop();
            if (inst->getOp() == instOp)
            {
                f(as<InstType>(inst));
            }

            for (auto child = inst->getLastChild(); child; child = child->getPrevInst())
            {
                addToWorkList(child);
            }
        }
    }

    template<typename Func>
    void processChildInsts(IRInst* root, const Func& f)
    {
        workList.clear();
        workListSet.clear();

        addToWorkList(root);

        while (workList.getCount() != 0)
        {
            IRInst* inst = pop();

            f(inst);

            for (auto child = inst->getLastChild(); child; child = child->getPrevInst())
            {
                addToWorkList(child);
            }
        }
    }

    template<typename Func>
    void processAllInsts(const Func& f)
    {
        processChildInsts(module->getModuleInst(), f);
    }

    template<typename Func>
    void processAllReachableInsts(const Func& f)
    {
        workList.clear();
        workListSet.clear();

        addToWorkList(module->getModuleInst());
        while (workList.getCount() != 0)
        {
            IRInst* inst = pop(false);
            f(inst);
            for (auto child = inst->getLastChild(); child; child = child->getPrevInst())
            {
                if (as<IRDecoration>(child))
                    break;
                switch (child->getOp())
                {
                case kIROp_GenericSpecializationDictionary:
                case kIROp_ExistentialFuncSpecializationDictionary:
                case kIROp_ExistentialTypeSpecializationDictionary:
                    continue;
                default:
                    break;
                }
                SLANG_ASSERT(child);
                if (shouldInstBeLiveIfParentIsLive(child, IRDeadCodeEliminationOptions()))
                    addToWorkList(child);
            }
        }
    }
};

} // namespace Slang
