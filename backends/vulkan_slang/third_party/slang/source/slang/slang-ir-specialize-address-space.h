// slang-ir-specialize-address-space.h
#pragma once

#include <cinttypes>

namespace Slang
{
struct IRModule;
struct IRInst;
enum class AddressSpace : uint64_t;

struct AddressSpaceSpecializationContext
{
public:
    virtual AddressSpace getAddrSpace(IRInst* inst) = 0;
};

struct InitialAddressSpaceAssigner
{
    virtual bool tryAssignAddressSpace(IRInst* inst, AddressSpace& outAddressSpace) = 0;
    virtual AddressSpace getAddressSpaceFromVarType(IRInst* type) = 0;
    virtual AddressSpace getLeafInstAddressSpace(IRInst* inst) = 0;
};

/// Propagate address space information through the IR module.
/// Specialize functions with reference/pointer parameters to use the correct address space
/// based on the address space of the arguments.
///
void specializeAddressSpace(IRModule* module, InitialAddressSpaceAssigner* addrSpaceAssigner);
} // namespace Slang
