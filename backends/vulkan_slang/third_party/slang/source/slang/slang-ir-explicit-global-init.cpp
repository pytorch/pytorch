// slang-ir-explicit-global-init.cpp
#include "slang-ir-explicit-global-init.h"

#include "slang-ir-insts.h"

namespace Slang
{

// This pass is responsible for taking code in a form like:
//
//      static int gCounter = 1;
//
//      void computeMain()
//      {
//          ...
//          int tmp = gCounter++;
//      }
//
// and transforming it so that the initialization of global
// variables is performed explicitly at the start of each
// entry-point funciton:
//
//      static int gCounter;
//
//      void computeMain()
//      {
//          gCounter = 1;
//          ...
//          int tmp = gCounter++;
//      }
//
// Transforming the code in this way may be required for targets
// that do not support initial-value expressions on global
// variables (e.g., SPIR-V is such a target). It can also be
// useful as a pre-process before other transformations that
// might work with global variables, because after this change
// there cannot be any global variables with initializers.

struct MoveGlobalVarInitializationToEntryPointsPass
{
    IRModule* m_module;
    TargetProgram* m_targetProgram;

    // In the Slang IR, a global variable represents a pointer
    // to the storage for the variable but it *also* encodes
    // the logic used to compute the initial value of that
    // variable. This works because `IRGlobalVar` is a subtype
    // of `IRGlobalValueWithCode`, which is also the base
    // type of `IRFunc`. Thus a global variable behaves a
    // bit like a function, which just happens to compute
    // the initial value for the variable.
    //
    // Part of the work in this pass will be to split those
    // two pars of the variable, so that we end up with
    // a global variable with not initialization logic,
    // plus an ordinary `IRFunc` to compute the initial
    // value.
    //
    // We will compute this split representation and then
    // hold onto it so that we can use it for injecting
    // the initialization logic into entry points.
    //
    struct GlobalVarInfo
    {
        IRGlobalVar* globalVar = nullptr;
        IRFunc* initFunc = nullptr;
    };
    List<GlobalVarInfo> m_globalVarsWithInit;

    void processModule(IRModule* module, TargetProgram* targetProgram)
    {
        m_module = module;
        m_targetProgram = targetProgram;

        // We start by looking for global variables with
        // initialization logic in the IR, and processing
        // each to produce a split variable (now without
        // initialization) and function (to compute the
        // initial value).
        //
        for (auto inst : m_module->getGlobalInsts())
        {
            auto globalVar = as<IRGlobalVar>(inst);
            if (!globalVar)
                continue;

            // If it's an `Actual Global` we don't want to move initialization
            if (as<IRActualGlobalRate>(globalVar->getRate()))
            {
                continue;
            }

            auto firstBlock = globalVar->getFirstBlock();
            if (!firstBlock)
                continue;

            processGlobalVarWithInit(globalVar, firstBlock);
        }

        // Then we loop over all the entry points in the
        // module and modify them to explicitly initialize
        // all the global variables that were identified
        // and processed in the first pass.
        //
        for (auto inst : m_module->getGlobalInsts())
        {
            auto func = as<IRFunc>(inst);
            if (!func)
                continue;

            if (!func->findDecoration<IREntryPointDecoration>())
                continue;

            processEntryPoint(func);
        }
    }

    bool shouldMoveGlobalVarInitialization(IRGlobalVar* globalVar)
    {
        // Currently CoopVector for DXC cannot be created from
        // constructors with arguments. When CoopVector is used as a
        // global variable, its initialization has to happen at the
        // beginning of the entry point.
        //
        // At the same time, we don't want to apply
        // "moveGlobalVarInitializationToEntryPoints" to the rest of
        // the global variables when targeting HLSL.
        //
        if (isD3DTarget(m_targetProgram->getTargetReq()))
        {
            auto valueType = globalVar->getDataType()->getValueType();
            if (as<IRCoopVectorType>(valueType))
                return true;
            return false;
        }
        return true;
    }

    void processGlobalVarWithInit(IRGlobalVar* globalVar, IRBlock* firstBlock)
    {
        if (!shouldMoveGlobalVarInitialization(globalVar))
            return;

        IRBuilder builder(m_module);
        builder.setInsertBefore(globalVar);

        // Becaue an `IRGlobalVar` reprsents a pointer to the storage
        // for the variable, we need to extract the underlying value
        // type from the pointer type.
        //
        auto valueType = globalVar->getDataType()->getValueType();

        // We are going to construct an explicit IR function to compute
        // the initial value of the variable. That function will alway
        // take zero parameters.
        //
        auto initFunc = builder.createFunc();
        initFunc->setFullType(builder.getFuncType(0, nullptr, valueType));

        // The basic blocks under teh `IRGlobalVar` define its initialization
        // logic, and we can simply move those blocks over to the new
        // `IRFunc` to define its behavior.
        //
        // As a result, the `globalVar` will no longer have its own
        // initialization logic, which is a postcondition this pass
        // needed to guarantee.
        //
        IRBlock* nextBlock = nullptr;
        for (IRBlock* block = firstBlock; block; block = nextBlock)
        {
            nextBlock = block->getNextBlock();

            block->removeFromParent();
            block->insertAtEnd(initFunc);
        }

        // We need to remember the variable and the assocaited
        // initial-value function so that we can iterate over
        // them in the per-entry-point logic below.
        //
        GlobalVarInfo info;
        info.globalVar = globalVar;
        info.initFunc = initFunc;
        m_globalVarsWithInit.add(info);
    }

    void processEntryPoint(IRFunc* entryPointFunc)
    {
        // We can only process entry point definitions, not declarations.
        //
        auto firstBlock = entryPointFunc->getFirstBlock();
        if (!firstBlock)
            return;

        // We are going to insert initiailization logic at the start
        // of the first block of the entry point.
        //
        IRBuilder builder(m_module);
        builder.setInsertBefore(firstBlock->getFirstOrdinaryInst());

        for (auto globalVarInfo : m_globalVarsWithInit)
        {
            // The earlier step split each global variable into
            // a variable with no initialization logic, plus a function
            // that can be called to compute the initial value.
            //
            auto globalVar = globalVarInfo.globalVar;
            auto initFunc = globalVarInfo.initFunc;

            // Because the `IRGlobalVar` represents a pointer to
            // storage, we need to get the pointed-to type to
            // get the type of the initial value.
            //
            auto valType = globalVar->getDataType()->getValueType();

            // To simplify the resulting code a bit, if we see the
            // initFunc just returns a constant value, then we just
            // use that inline.
            IRInst* initVal = nullptr;
            if (auto initFirstBlock = initFunc->getFirstBlock())
            {
                if (auto returnInst = as<IRReturn>(initFirstBlock->getTerminator()))
                {
                    if (returnInst->getVal() &&
                        returnInst->getVal()->getParent() == m_module->getModuleInst())
                    {
                        initVal = returnInst->getVal();
                    }
                }
            }
            if (!initVal)
            {
                // We compute the initial value for the variable by calling
                // the initial-value function with no arguments, and then
                // we store that value into the corresponding global.
                //
                initVal = builder.emitCallInst(valType, initFunc, 0, nullptr);
            }
            builder.emitStore(globalVar, initVal);
        }
    }
};

/// Move initialization logic off of global variables and onto each entry point
void moveGlobalVarInitializationToEntryPoints(IRModule* module, TargetProgram* targetProgram)
{
    MoveGlobalVarInitializationToEntryPointsPass pass;
    pass.processModule(module, targetProgram);
}

} // namespace Slang
