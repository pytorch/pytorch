#pragma once

namespace Slang
{
struct IRModule;

// After IR lowering, an `expand each X` type will be defined in the IR as:
//    %X = ...
//    %e = IREach(%X)
//    %expand = IRExpandType(%e)
// This form allows our IR deduplication logic to find the deduplicate the same
// `exapnd` types into the same IR inst.
// However after lowering is done, we no longer need this deduplication service.
// But having expand types defined in this form is making it very difficult to
// specialize.
// This pass runs immediately after IR lowering process for a module (pre-linking)
// to turn `IRExpandType` into `IRExpand`, so that the above expand type will be
// represented as:
//     %expand = IRExpand : IRTypeKind
//     {
//         %eachIndex = IRParam : int;
//         %e = ...; // may use %eachIndex.
//         yield %e;
//     }
//
// After this translation, there should be no longer any IRExpandType/IREach instructions
// that are alive in the IR. All future passes will only need to deal with IRExpand.
//
void lowerExpandType(IRModule* module);
} // namespace Slang
