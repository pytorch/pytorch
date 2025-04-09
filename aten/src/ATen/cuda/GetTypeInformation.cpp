#include <iostream>
#include <string>
#include <functional>
#include <optional>
#include <memory>
#include <variant>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cassert>
#include <iomanip>
#include <sstream>

#include <dwarf.h>
#include <libdwarf.h>

#include <link.h>

#include "GetTypeInformation.h"

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <c10/core/CPUAllocator.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>

#include <etbl_private.h>

#include <iostream>
#include <fstream>

#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/syscall.h>

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif

// memfd_create wrapper for systems where it isn't in glibc yet
int memfd_create(const char *name, unsigned int flags) {
    return syscall(SYS_memfd_create, name, flags);
}

Dwarf_Die get_type_follow_typedefs(Dwarf_Debug dbg, Dwarf_Die typeDie) {
  Dwarf_Half typeTag;
  Dwarf_Error error;
  if (dwarf_tag(typeDie, &typeTag, &error) != DW_DLV_OK) {
    std::cerr << "GALVEZ: error!" << std::endl;
    return nullptr; // Return original die on error
  }

  if (typeTag == DW_TAG_typedef) {
    // Follow the typedef to its underlying type
    Dwarf_Attribute typeAttr;
    int res = dwarf_attr(typeDie, DW_AT_type, &typeAttr, &error);
    if (res == DW_DLV_OK) {
      Dwarf_Off typeOffset;
      Dwarf_Bool is_info;
      Dwarf_Die typeDieTmp;
      
      if (dwarf_global_formref(typeAttr, &typeOffset, &error) == DW_DLV_OK) {
        is_info = dwarf_get_die_infotypes_flag(typeDie);
        
        if (dwarf_offdie_b(dbg, typeOffset, 
                         is_info, &typeDieTmp, &error) == DW_DLV_OK) {
          // Clean up before recursion
          dwarf_dealloc_attribute(typeAttr);

          // maybe typeDieTmp does not have a CU context when obtained via dwarf_offdie_b?
          
          // Recursively follow the typedef
          Dwarf_Die result = get_type_follow_typedefs(dbg, typeDieTmp);
          // is this dealloc correct? Yes.
          // dwarf_dealloc_die(typeDieTmp);
          return result;
        }
      }
      dwarf_dealloc_attribute(typeAttr);
    }
  }
  return typeDie; // Return when we've found a non-typedef
}

// Function declarations
bool findFunctionWithLinkageName(Dwarf_Debug dbg, Dwarf_Die cu_die, 
                               const char* linkageName,
                               ArgumentInformation& result);
void extractFunctionArguments(Dwarf_Debug dbg, Dwarf_Die function_die, 
                           ArgumentInformation& result);
void processType(Dwarf_Debug dbg, Dwarf_Die type_die, 
              std::variant<BasicType, StructType, ArrayType>& member);
void processBasicType(Dwarf_Debug dbg, Dwarf_Die type_die, 
                      std::variant<BasicType, StructType, ArrayType>& member);
void processStructType(Dwarf_Debug dbg, Dwarf_Die struct_die,
                    std::variant<BasicType, StructType, ArrayType>& member);
void processArrayType(Dwarf_Debug dbg, Dwarf_Die array_die, 
                   std::variant<BasicType, StructType, ArrayType>& member);
int dwarf_attr_string(Dwarf_Die die, Dwarf_Half attr_code, char** result, Dwarf_Error* error);

ArgumentInformation
getArgumentInformation(const char* linkageName, void *buffer, size_t buffer_size) {
    int fd = memfd_create("galvezanonymous", MFD_CLOEXEC);
    if (fd == -1) {
      perror("memfd_create");
      free(buffer);
      exit(EXIT_FAILURE);
    }

    if (ftruncate(fd, buffer_size) == -1) {
      perror("ftruncate");
      free(buffer);
      close(fd);
      exit(EXIT_FAILURE);
    }

    void* mapped = mmap(NULL, buffer_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        perror("mmap");
        free(buffer);
        close(fd);
        exit(EXIT_FAILURE);
    }

    memcpy(mapped, buffer, buffer_size);

    // std::ofstream outFile("my_elf_files/"+ std::string(linkageName) + ".elf", std::ios::binary);
    // if (!outFile) {
    //     std::cerr << "Failed to open file: " << linkageName << std::endl;
    //     abort();
    // }
    // outFile.write(static_cast<const char*>(mapped), buffer_size);
    // outFile.close();

    // Result vector to return
    ArgumentInformation result;
    
    // Variables for error handling
    Dwarf_Error error = nullptr;
    int res = DW_DLV_ERROR;
    
    // Open the DWARF debug info
    Dwarf_Debug dbg = nullptr;

    res = dwarf_init_b(fd, DW_GROUPNUMBER_ANY, nullptr, nullptr, &dbg, &error);
    // res = dwarf_init_path(elfPath.c_str(), nullptr, 0, 
    //                      DW_GROUPNUMBER_ANY, nullptr, nullptr, &dbg, &error);
    if (res != DW_DLV_OK) {
        if (res == DW_DLV_ERROR) {
            dwarf_dealloc_error(dbg, error);
        }
        return result; // Empty result
    }
    
    // Iterate through compilation units
    Dwarf_Unsigned next_cu_offset;


    bool ran_twice = false;

    while (true) {
        Dwarf_Die cu_die = nullptr;
        Dwarf_Unsigned cu_header_length = 0;
        Dwarf_Half version_stamp = 0;
        Dwarf_Unsigned abbrev_offset = 0;
        Dwarf_Half address_size = 0;
        Dwarf_Half length_size = 0;
        Dwarf_Half extension_size = 0;
        Dwarf_Unsigned next_cu_header_offset = 0;
        Dwarf_Half cu_type = 0;

        // std::cout << "GALVEZ: while(true)" << std::endl;

        // Get next CU
        res = dwarf_next_cu_header_e(dbg, true, &cu_die, 
            &cu_header_length, &version_stamp, &abbrev_offset, &address_size,
            &length_size, &extension_size, nullptr, nullptr, &next_cu_header_offset,
            &cu_type, &error);
            
        if (res != DW_DLV_OK) {
            std::cout << "GALVEZ: no more CUs" << std::endl;
            break; // No more CUs or error
        }

        if (ran_twice) {
          std::cout << "GALVEZ: ran twice!!!" << std::endl;
          abort();
        }
        
        // Find function DIE with the specified linkage name
        bool found = findFunctionWithLinkageName(dbg, cu_die, linkageName, result);
        
        // Clean up the CU DIE
        dwarf_dealloc_die(cu_die);

        if (found) {
          break;
        }
        
        // Move to the next CU
        next_cu_offset = next_cu_header_offset;

        ran_twice = true;
    }
    
    // Clean up DWARF debug info
    dwarf_finish(dbg);

    std::cout << "GALVEZ: getArgumentInformation() done" << std::endl;

    // if (result.empty()) {
    //   std::cout << "GALVEZ: failed to get information of kernel was empty:" << std::endl;
    // }
    
    return result;
}

ArgumentInformation
getArgumentInformation(const char* linkageName, const std::string& elfPath) {
    // Result vector to return
    ArgumentInformation result;
    
    // Variables for error handling
    Dwarf_Error error = nullptr;
    int res = DW_DLV_ERROR;
    
    // Open the DWARF debug info
    Dwarf_Debug dbg = nullptr;
    res = dwarf_init_path(elfPath.c_str(), nullptr, 0, 
                         DW_GROUPNUMBER_ANY, nullptr, nullptr, &dbg, &error);
    if (res != DW_DLV_OK) {
        if (res == DW_DLV_ERROR) {
            dwarf_dealloc_error(dbg, error);
        }
        return result; // Empty result
    }
    
    // Iterate through compilation units
    Dwarf_Unsigned next_cu_offset;
    while (true) {
        Dwarf_Die cu_die = nullptr;
        Dwarf_Unsigned cu_header_length = 0;
        Dwarf_Half version_stamp = 0;
        Dwarf_Unsigned abbrev_offset = 0;
        Dwarf_Half address_size = 0;
        Dwarf_Half length_size = 0;
        Dwarf_Half extension_size = 0;
        Dwarf_Unsigned next_cu_header_offset = 0;
        Dwarf_Half cu_type = 0;
        
        // Get next CU
        res = dwarf_next_cu_header_e(dbg, true, &cu_die, 
            &cu_header_length, &version_stamp, &abbrev_offset, &address_size,
            &length_size, &extension_size, nullptr, nullptr, &next_cu_header_offset,
            &cu_type, &error);
            
        if (res != DW_DLV_OK) {
            break; // No more CUs or error
        }
        
        // Find function DIE with the specified linkage name
        findFunctionWithLinkageName(dbg, cu_die, linkageName, result);
        
        // Clean up the CU DIE
        dwarf_dealloc_die(cu_die);
        
        // Move to the next CU
        next_cu_offset = next_cu_header_offset;
    }
    
    // Clean up DWARF debug info
    dwarf_finish(dbg);
    
    return result;
}

// Helper function to find a function with the specified linkage name
bool findFunctionWithLinkageName(Dwarf_Debug dbg, Dwarf_Die cu_die, 
                                const char* linkageName,
                                ArgumentInformation& result) {
    if (!cu_die) {
      std::cout << "GALVEZ: cu_die is empty" << std::endl;
        return false;
    }
    
    // Get the first child of the CU DIE
    Dwarf_Die child = nullptr;
    Dwarf_Error error = nullptr;
    int res = dwarf_child(cu_die, &child, &error);
    if (res != DW_DLV_OK) {
      // std::cout << "GALVEZ: no child" << std::endl;
        return false;
    }
    
    // Iterate through siblings
    Dwarf_Die current_die = child;
    while (current_die) {
      // std::cout << "GALVEZ: current_die" << std::endl;
        // Check if this is a subprogram (function) DIE
        Dwarf_Half tag = 0;
        res = dwarf_tag(current_die, &tag, &error);
        if (res == DW_DLV_OK && tag == DW_TAG_subprogram) {
            // Check for DW_AT_linkage_name attribute
            char* name_str = nullptr;
            res = dwarf_attr_string(current_die, DW_AT_linkage_name, &name_str, &error);
            if (res != DW_DLV_OK) {
                // Try DW_AT_MIPS_linkage_name as fallback
                res = dwarf_attr_string(current_die, DW_AT_MIPS_linkage_name, &name_str, &error);
            }

            
            // Check if it matches our target linkage name
            if (res == DW_DLV_OK && name_str && strcmp(name_str, linkageName) == 0) {
              std::cout << "GALVEZ: found linkage_name=" << name_str << std::endl;
              // Found the function, extract its arguments
                extractFunctionArguments(dbg, current_die, result);
                return true; // We found the function, no need to continue
            }
        }
        
        // Recursively search in children
        if (findFunctionWithLinkageName(dbg, current_die, linkageName, result)) {
          return true;
        }
        
        // Move to next sibling
        Dwarf_Die sibling_die = nullptr;
        res = dwarf_siblingof_c(current_die, &sibling_die, &error);
        if (res != DW_DLV_OK) {
            break;
        }
        
        // Free current DIE and move to sibling
        dwarf_dealloc_die(current_die);
        current_die = sibling_die;
    }
    return false;
}

// Helper to extract function parameter information from a function DIE
void extractFunctionArguments(Dwarf_Debug dbg, Dwarf_Die function_die, 
                              ArgumentInformation& result) {
    Dwarf_Error error = nullptr;
    int res = DW_DLV_ERROR;
    
    // Get the first child of the function DIE (parameters are children)
    Dwarf_Die child = nullptr;
    res = dwarf_child(function_die, &child, &error);
    if (res != DW_DLV_OK) {
        return;
    }
    
    // Iterate through the function DIE's children
    Dwarf_Die current_die = child;
    while (current_die) {
        // Check if this is a formal parameter DIE
        Dwarf_Half tag = 0;
        res = dwarf_tag(current_die, &tag, &error);
        if (res == DW_DLV_OK && tag == DW_TAG_formal_parameter) {
            // This is a function parameter, extract its information
            // Get parameter name
            char* name_str = nullptr;
            res = dwarf_attr_string(current_die, DW_AT_name, &name_str, &error);
            if (res != DW_DLV_OK && name_str) {
                // argInfo.name = std::string(name_str);
            }
            
            result.members.emplace_back(std::string(name_str), std::variant<BasicType, StructType, ArrayType>());
            // Get parameter type
            Dwarf_Attribute type_attr = nullptr;
            res = dwarf_attr(current_die, DW_AT_type, &type_attr, &error);
            if (res == DW_DLV_OK) {
                // Get DIE offset from the type attribute
                Dwarf_Off type_offset = 0;
                Dwarf_Bool is_info = 0;
                res = dwarf_global_formref_b(type_attr, &type_offset, &is_info, &error);
                if (res == DW_DLV_OK) {
                    // Get the type DIE
                    Dwarf_Die type_die = nullptr;
                    res = dwarf_offdie_b(dbg, type_offset, is_info, &type_die, &error);
                    if (res == DW_DLV_OK) {
                        // Process the type
                      processType(dbg, type_die, result.members.back().second);
                        dwarf_dealloc_die(type_die);
                    }
                }
                dwarf_dealloc_attribute(type_attr);
            }
            
            // Add the parameter information to the result
            // result.push_back(std::move(argInfo));
        }
        
        // Move to next sibling
        Dwarf_Die sibling_die = nullptr;
        res = dwarf_siblingof_c(current_die, &sibling_die, &error);
        if (res != DW_DLV_OK) {
            break;
        }
        
        // Free current DIE and move to sibling
        dwarf_dealloc_die(current_die);
        current_die = sibling_die;
    }
}

// Helper function to extract type information
void processType(Dwarf_Debug dbg, Dwarf_Die type_die, 
                 std::variant<BasicType, StructType, ArrayType>& member) {
    // Follow typedefs to get the actual type
    Dwarf_Die actual_type_die = get_type_follow_typedefs(dbg, type_die);
    if (!actual_type_die) {
        return;
    }
    
    Dwarf_Error error = nullptr;
    int res = DW_DLV_ERROR;
    
    // Get the tag of the type
    Dwarf_Half tag = 0;
    res = dwarf_tag(actual_type_die, &tag, &error);
    if (res != DW_DLV_OK) {
        return;
    }
    
    // Process based on the type of DIE
    switch (tag) {
        case DW_TAG_base_type:
        case DW_TAG_pointer_type:
        case DW_TAG_reference_type:
            processBasicType(dbg, actual_type_die, member);
            break;
            
        case DW_TAG_structure_type:
        case DW_TAG_class_type:
        case DW_TAG_union_type:
            processStructType(dbg, actual_type_die, member);
            break;
            
        case DW_TAG_array_type:
            processArrayType(dbg, actual_type_die, member);
            break;
            
        default:
            // For other types, we'll just treat them as basic types
            processBasicType(dbg, actual_type_die, member);
            break;
    }
    
    // Clean up if we've made a copy of the DIE
    if (actual_type_die != type_die) {
        dwarf_dealloc_die(actual_type_die);
    }
}

// Process a basic type or pointer/reference
void processBasicType(Dwarf_Debug dbg, Dwarf_Die type_die, 
                     std::variant<BasicType, StructType, ArrayType>& member) {
    Dwarf_Error error = nullptr;
    BasicType basicType;
    
    // Check if it's a pointer or reference
    Dwarf_Half tag = 0;
    int res = dwarf_tag(type_die, &tag, &error);
    // TODO: I'm not sure that it is correct to call this a pointer
    // type if it's a reference type...
    basicType.is_pointer = (tag == DW_TAG_pointer_type || tag == DW_TAG_reference_type);

    // Get type name
    char* name_str = nullptr;
    res = dwarf_attr_string(type_die, DW_AT_name, &name_str, &error);
    if (res == DW_DLV_OK && name_str) {
        basicType.type_name = std::string(name_str);
    } else if (tag == DW_TAG_pointer_type || tag == DW_TAG_reference_type) {
        // int res = dwarf_attr(type_die, DW_AT_type, &typeAttr, &error);
        // if (dwarf_global_formref(typeAttr, &typeOffset, &error) == DW_DLV_OK) {
        //   bool is_info = dwarf_get_die_infotypes_flag(typeDie);
        //   if (dwarf_offdie_b(dbg, typeOffset, 
        //                      is_info, &typeDieTmp, &error) == DW_DLV_OK) {
        //     get_type_follow_typedefs(dbg, typeDieTmp);
        //     res = dwarf_attr_string(type_die, DW_AT_name, &name_str, &error);
        //     type_die
        //   }
        // }
        basicType.type_name =  "*";
    } else {
        basicType.type_name = "unnamed_type";
    }
    
    // Get byte size
    Dwarf_Unsigned size = 0;
    res = dwarf_bytesize(type_die, &size, &error);
    if (res == DW_DLV_OK) {
        basicType.size = size;
    } else if(res != DW_DLV_OK && basicType.is_pointer) {
        // clang does not generate size information for pointers for whatever reason.
        basicType.size = 8;
    } else {
        basicType.size = 0;
    }
    
    // Get the offset (member position)
    basicType.offset = 0; // Default to 0 for standalone types
    
    // Add to the members list
    member = basicType;
}

// Process a struct/class/union type
void processStructType(Dwarf_Debug dbg, Dwarf_Die struct_die, 
                      std::variant<BasicType, StructType, ArrayType>& member) {
    Dwarf_Error error = nullptr;
    StructType structType;
    
    // Get struct name
    char* name_str = nullptr;
    int res = dwarf_attr_string(struct_die, DW_AT_name, &name_str, &error);
    if (res == DW_DLV_OK && name_str) {
        structType.type_name = std::string(name_str);
    } else {
        structType.type_name = "unnamed_struct";
    }


    Dwarf_Unsigned size;
    res = dwarf_bytesize(struct_die, &size, &error);
    TORCH_INTERNAL_ASSERT(res == DW_DLV_OK, "Must have a size for structs");
    structType.size = size;
    
    // Initialize members vector
    // structType.members = std::vector<std::variant<BasicType, StructType, ArrayType>>();
    
    // Process struct members
    Dwarf_Die child = nullptr;
    res = dwarf_child(struct_die, &child, &error);
    if (res == DW_DLV_OK) {
        Dwarf_Die current_die = child;
        while (current_die) {
            // Check if this is a member
            Dwarf_Half tag = 0;
            res = dwarf_tag(current_die, &tag, &error);
            if (res == DW_DLV_OK && tag == DW_TAG_member) {
                // Get member type
                Dwarf_Attribute type_attr = nullptr;
                res = dwarf_attr(current_die, DW_AT_type, &type_attr, &error);
                if (res == DW_DLV_OK) {
                    // Get DIE offset from the type attribute
                    Dwarf_Off type_offset = 0;
                    Dwarf_Bool is_info = 0;
                    res = dwarf_global_formref_b(type_attr, &type_offset, &is_info, &error);
                    if (res == DW_DLV_OK) {
                        // Get the type DIE
                        Dwarf_Die type_die = nullptr;
                        res = dwarf_offdie_b(dbg, type_offset, is_info, &type_die, &error);
                        if (res == DW_DLV_OK) {
                            // Get data member offset
                            Dwarf_Unsigned member_offset = 0;
                            Dwarf_Attribute offset_attr = nullptr;
                            res = dwarf_attr(current_die, DW_AT_data_member_location, &offset_attr, &error);
                            if (res == DW_DLV_OK) {
                                // The attribute could be either a constant or an exprloc
                                Dwarf_Half form = 0;
                                res = dwarf_whatform(offset_attr, &form, &error);
                                if (res == DW_DLV_OK) {
                                    if (form == DW_FORM_data1 || form == DW_FORM_data2 || 
                                        form == DW_FORM_data4 || form == DW_FORM_data8 ||
                                        form == DW_FORM_udata) {
                                        // It's a constant
                                        Dwarf_Unsigned offset_val = 0;
                                        res = dwarf_formudata(offset_attr, &offset_val, &error);
                                        if (res == DW_DLV_OK) {
                                            member_offset = offset_val;
                                        }
                                    } else if (form == DW_FORM_exprloc || form == DW_FORM_block1 ||
                                              form == DW_FORM_block2 || form == DW_FORM_block4) {
                                        // It's a location expression
                                        Dwarf_Loc_Head_c loc_head = nullptr;
                                        Dwarf_Unsigned loc_count = 0;
                                        res = dwarf_get_loclist_c(offset_attr, &loc_head, &loc_count, &error);
                                        if (res == DW_DLV_OK && loc_count > 0) {
                                            // Process the first location description
                                            Dwarf_Small lle_value = 0;
                                            Dwarf_Unsigned raw_lowpc = 0, raw_highpc = 0;
                                            Dwarf_Bool debug_addr_unavailable = false;
                                            Dwarf_Addr lowpc = 0, highpc = 0;
                                            Dwarf_Unsigned loc_expr_op_count = 0;
                                            Dwarf_Locdesc_c locdesc = nullptr;
                                            Dwarf_Small loclist_source = 0;
                                            Dwarf_Unsigned expr_offset = 0, locdesc_offset = 0;
                                            
                                            res = dwarf_get_locdesc_entry_d(loc_head, 0,
                                                &lle_value, &raw_lowpc, &raw_highpc,
                                                &debug_addr_unavailable, &lowpc, &highpc,
                                                &loc_expr_op_count, &locdesc,
                                                &loclist_source, &expr_offset, &locdesc_offset,
                                                &error);
                                                
                                            if (res == DW_DLV_OK && loc_expr_op_count > 0) {
                                                // Check if the expression is a simple constant
                                                Dwarf_Small op = 0;
                                                Dwarf_Unsigned op1 = 0, op2 = 0, op3 = 0, offset_for_branch = 0;
                                                
                                                res = dwarf_get_location_op_value_c(locdesc, 0,
                                                    &op, &op1, &op2, &op3, &offset_for_branch, &error);
                                                    
                                                if (res == DW_DLV_OK && op == DW_OP_plus_uconst) {
                                                    member_offset = op1;
                                                }
                                            }
                                            dwarf_dealloc_loc_head_c(loc_head);
                                        }
                                    }
                                }
                                dwarf_dealloc_attribute(offset_attr);
                            }
                            
                            // Now process the member type, but we need to create a vector for its members
                            std::variant<BasicType, StructType, ArrayType> member_data;
                            processType(dbg, type_die, member_data);
                            
                            // Update the offset of the first member based on the location
                            std::visit([member_offset](auto&& arg) {
                              using T = std::decay_t<decltype(arg)>;
                              if constexpr (std::is_same_v<T, BasicType>) {
                                arg.offset = member_offset;
                              } else if constexpr (std::is_same_v<T, StructType>) {
                                arg.offset = member_offset;
                              } else if constexpr (std::is_same_v<T, ArrayType>) {
                                arg.offset = member_offset;
                              }
                            }, member_data);


                            std::string member_name;
                            res = dwarf_attr_string(current_die, DW_AT_name, &name_str, &error);
                            if (res == DW_DLV_OK) {
                              member_name = name_str;
                            } else {
                              member_name = "Could not get member name";
                            }
                            
                            // Add all the member data to the struct
                            structType.members.emplace_back(member_name, member_data);
                            
                            dwarf_dealloc_die(type_die);
                        }
                    }
                    dwarf_dealloc_attribute(type_attr);
                }
            }
            
            // Move to next sibling
            Dwarf_Die sibling_die = nullptr;
            res = dwarf_siblingof_c(current_die, &sibling_die, &error);
            if (res != DW_DLV_OK) {
                break;
            }
            
            // Free current DIE and move to sibling
            dwarf_dealloc_die(current_die);
            current_die = sibling_die;
        }
    }
    
    // Add the struct to the members list
    member = structType;
}


using In  = std::variant<BasicType, StructType, ArrayType>;
using Out = std::variant<BasicType, StructType>;

// helper for overloaded lambdas
template<class... Fs> struct overloaded : Fs... { using Fs::operator()...; };
template<class... Fs> overloaded(Fs...) -> overloaded<Fs...>;

Out filter(In const& in) {
    return std::visit(overloaded {
        // BasicType  → keep as-is
        [](BasicType const& b) -> Out { 
            return b; 
        },
        // StructType → keep as-is
        [](StructType const& s) -> Out { 
            return s; 
        },
        // ArrayType  → error
        [](ArrayType const&) -> Out { 
            throw std::runtime_error("ArrayType not allowed here"); 
        }
    }, in);
}

// Process an array type
void processArrayType(Dwarf_Debug dbg, Dwarf_Die array_die, 
                     std::variant<BasicType, StructType, ArrayType>& member) {
    Dwarf_Error error = nullptr;
    ArrayType arrayType;
    
    arrayType.type_name = "array";
    arrayType.num_elements = 0;
    
    // Get the element type
    Dwarf_Attribute type_attr = nullptr;
    int res = dwarf_attr(array_die, DW_AT_type, &type_attr, &error);
    if (res == DW_DLV_OK) {
        // Get DIE offset from the type attribute
        Dwarf_Off type_offset = 0;
        Dwarf_Bool is_info = 0;
        res = dwarf_global_formref_b(type_attr, &type_offset, &is_info, &error);
        if (res == DW_DLV_OK) {
            // Get the type DIE
            Dwarf_Die type_die = nullptr;
            res = dwarf_offdie_b(dbg, type_offset, is_info, &type_die, &error);
            if (res == DW_DLV_OK) {
                // We need to create a vector for the element type
                std::variant<BasicType, StructType, ArrayType> element_data;
                processType(dbg, type_die, element_data);

                arrayType.element_type = filter(element_data);
                
                dwarf_dealloc_die(type_die);
            }
        }
        dwarf_dealloc_attribute(type_attr);
    }
    
    // Try to get the array size
    Dwarf_Die child = nullptr;
    res = dwarf_child(array_die, &child, &error);
    if (res == DW_DLV_OK) {
        Dwarf_Die current_die = child;
        while (current_die) {
            // Check if this is a subrange type
            Dwarf_Half tag = 0;
            res = dwarf_tag(current_die, &tag, &error);
            if (res == DW_DLV_OK && tag == DW_TAG_subrange_type) {
                // Get the upper bound or count
                Dwarf_Attribute count_attr = nullptr;
                res = dwarf_attr(current_die, DW_AT_count, &count_attr, &error);
                if (res == DW_DLV_OK) {
                    Dwarf_Unsigned count = 0;
                    res = dwarf_formudata(count_attr, &count, &error);
                    if (res == DW_DLV_OK) {
                        arrayType.num_elements = count;
                    }
                    dwarf_dealloc_attribute(count_attr);
                } else {
                    // Try upper bound
                    Dwarf_Attribute upper_attr = nullptr;
                    res = dwarf_attr(current_die, DW_AT_upper_bound, &upper_attr, &error);
                    if (res == DW_DLV_OK) {
                        Dwarf_Unsigned upper = 0;
                        res = dwarf_formudata(upper_attr, &upper, &error);
                        if (res == DW_DLV_OK) {
                            // Array size is upper bound + 1
                            arrayType.num_elements = upper + 1;
                        }
                        dwarf_dealloc_attribute(upper_attr);
                    }
                }
            }
            
            // Move to next sibling
            Dwarf_Die sibling_die = nullptr;
            res = dwarf_siblingof_c(current_die, &sibling_die, &error);
            if (res != DW_DLV_OK) {
                break;
            }
            
            // Free current DIE and move to sibling
            dwarf_dealloc_die(current_die);
            current_die = sibling_die;
        }
    }
    
    // Add the array to the members list
    member = arrayType;
}

// Helper function to get a stringified value from an attribute
int dwarf_attr_string(Dwarf_Die die, Dwarf_Half attr_code, char** result, Dwarf_Error* error) {
    Dwarf_Attribute attr = nullptr;
    int res = dwarf_attr(die, attr_code, &attr, error);
    if (res != DW_DLV_OK) {
        return res;
    }
    
    res = dwarf_formstring(attr, result, error);
    dwarf_dealloc_attribute(attr);
    return res;
}

// Forward declarations for the print operators
std::ostream& operator<<(std::ostream& os, const BasicType& bt);
std::ostream& operator<<(std::ostream& os, const StructType& st);
std::ostream& operator<<(std::ostream& os, const ArrayType& at);

// Helper function to print indentation
inline std::string indent(int level) {
    return std::string(level * 2, ' ');
}

// Generic visitor for std::variant printing
struct VariantPrinter {
    std::ostream& os;
    int indent_level;
    
    VariantPrinter(std::ostream& os, int indent_level) : os(os), indent_level(indent_level) {}
    
    void operator()(const BasicType& bt) const {
        os << bt;
    }
    
    void operator()(const StructType& st) const {
        os << st;
    }
    
    void operator()(const ArrayType& at) const {
        os << at;
    }
};

// Print operator for BasicType
std::ostream& operator<<(std::ostream& os, const BasicType& bt) {
    os << "BasicType { ";
    os << "type_name: \"" << bt.type_name << "\", ";
    os << "offset: " << bt.offset << ", ";
    os << "size: " << bt.size << ", ";
    os << "is_pointer: " << (bt.is_pointer ? "true" : "false");
    os << " }";
    return os;
}

// Print operator for StructType
std::ostream& operator<<(std::ostream& os, const StructType& st) {
    os << "StructType { type_name: \"" << st.type_name << "\"";
    
    if (!st.members.empty()) {
        os << ", members: [\n";
        
        for (size_t i = 0; i < st.members.size(); ++i) {
            os << indent(1) << st.members[i].first << ": ";
            
            // Use a new VariantPrinter with increased indent level
            std::visit([&os, i](const auto& value) {
                os << value;
            }, st.members[i].second);
            
            if (i < st.members.size() - 1) {
                os << ",";
            }
            os << "\n";
        }
        
        os << "  ]";
    } else {
        os << ", members: []";
    }
    
    os << " }";
    return os;
}

// Print operator for ArrayType
std::ostream& operator<<(std::ostream& os, const ArrayType& at) {
    os << "ArrayType { ";
    os << "type_name: \"" << at.type_name << "\", ";
    os << "num_elements: " << at.num_elements;
    
    os << ", element_type: ";
    std::visit([&os](const auto& value) {
        os << value;
    }, at.element_type);
    
    os << " }";
    return os;
}

// Specialized print function with proper indentation for nested structures
void prettyPrintType(std::ostream& os, const std::variant<BasicType, StructType, ArrayType>& type, int level = 0) {
    std::visit(overloaded {
        [&os, level](const BasicType& bt) {
            os << indent(level) << bt;
        },
        [&os, level](const StructType& st) {
            os << indent(level) << "StructType { type_name: \"" << st.type_name << "\"";
            
            if (!st.members.empty()) {
                os << ", members: [\n";
                
                for (size_t i = 0; i < st.members.size(); ++i) {
                    os << indent(level + 1) << st.members[i].first << ": ";
                    prettyPrintType(os, st.members[i].second, level + 2);
                    
                    if (i < st.members.size() - 1) {
                        os << ",";
                    }
                    os << "\n";
                }
                
                os << indent(level) << "]";
            } else {
                os << ", members: []";
            }
            
            os << " }";
        },
        [&os, level](const ArrayType& at) {
            os << indent(level) << "ArrayType { ";
            os << "type_name: \"" << at.type_name << "\", ";
            os << "num_elements: " << at.num_elements;
            
            os << ", element_type: ";
            prettyPrintType(os, std::visit(conversion_visitor, at.element_type), level + 1);
            
            os << " }";
        }
    }, type);
}

// Updated prettyPrintArgumentInfo to use the new formatting
void prettyPrintArgumentInfo(const ArgumentInformation& args) {
    if (args.members.empty()) {
        std::cout << "No arguments found." << std::endl;
        return;
    }
    
    std::cout << "Arguments information:" << std::endl;
    
    std::cout << "StructType { type_name: \"" << args.type_name << "\"";
    
    if (!args.members.empty()) {
        std::cout << ", members: [\n";
        
        for (size_t i = 0; i < args.members.size(); ++i) {
            std::cout << indent(1) << args.members[i].first << ": ";
            prettyPrintType(std::cout, args.members[i].second, 1);
            
            if (i < args.members.size() - 1) {
                std::cout << ",";
            }
            std::cout << "\n";
        }
        
        std::cout << "  ]";
    } else {
        std::cout << ", members: []";
    }
    
    std::cout << " }" << std::endl;
}


int collect_so_file_elf_paths(struct dl_phdr_info *info, size_t size, void *data) {
  auto paths = (std::vector<const char*>*) data;
  std::string_view view(info->dlpi_name);
  if (view.find("libtorch_cuda") != std::string_view::npos) {
    paths->push_back(info->dlpi_name);
  }
  return 0;
}


std::unordered_map<std::string, ArgumentInformation>
get_argument_information(const std::vector<std::string> &function_names) {
  std::vector<const char*> paths;
  dl_iterate_phdr(collect_so_file_elf_paths, &paths);

  for (auto&& path: paths) {
    std::cout << "GALVEZ: SO path=" << path << std::endl;
  }
  
  std::unordered_map<std::string, ArgumentInformation> function_to_data;
  for(auto&& function_name: function_names) {
    for(auto&& path: paths) {
      ArgumentInformation type_information = getArgumentInformation(function_name.c_str(), path);
      // TODO: What if the function takes no arguments
      if (!type_information.members.empty()) {
        function_to_data.emplace(function_name, std::move(type_information));
        break;
      }
    }
  }
  return function_to_data;
}

ArgumentInformation
get_argument_information(CUfunction func) {
  const char* func_name;
  AT_CUDA_DRIVER_CHECK(
                       at::globalContext().getNVRTC().cuFuncGetName(&func_name, func));

  CUmodule module;
  AT_CUDA_DRIVER_CHECK(
                       at::globalContext().getNVRTC().cuFuncGetModule(&module, func));

  size_t image_size;
  void *void_image;
  module_get_image(module, &void_image, &image_size);
  // char *image = new char[image_size];
  // void *void_image = reinterpret_cast<void*>(image);
  // module_get_image(module, &void_image, &image_size);

  ArgumentInformation type_information = getArgumentInformation(func_name, void_image, image_size);
// delete image;
  return type_information;
}


bool sizeof_type(const std::variant<BasicType, StructType, ArrayType>& type_info) {
  return std::visit([](auto&& arg) {
    using T = std::decay_t<decltype(arg)>;
    if constexpr (std::is_same_v<T, BasicType>) {
      return arg.size;
    } else if constexpr (std::is_same_v<T, StructType>) {
      // TODO: Double check that that this is always true.
      // not true: https://chatgpt.com/share/67dd036a-a250-8000-9355-8dd04db0c6c6
      // tail padding exists
      // I need to have a size field for Structs unfortunately
      // TODO: Fill this out properly!!!
      return arg.size;  // offsetof_type(arg.members->back()) + sizeof_type(arg.members->back())
    } else if constexpr (std::is_same_v<T, ArrayType>) {


      return arg.num_elements * sizeof_type(std::visit(conversion_visitor, arg.element_type));
    }
  }, type_info);
}

struct A {
  bool b;
  struct B {
    int c;
    bool d;
    double e;
  };
  struct C {
    int f;
    bool g;
    double h;
    struct D {
      int i;
    };
  };
};

bool is_equal(char *arg1, char *arg2,
              const std::variant<BasicType, StructType, ArrayType>& type_info,
              bool is_last_member_of_struct,
              size_t global_offset_bytes, const std::string& name) {
  return std::visit([arg1, arg2, &is_last_member_of_struct, &global_offset_bytes, &name](auto&& arg) -> bool {
    using T = std::decay_t<decltype(arg)>;
    if constexpr (std::is_same_v<T, BasicType>) {
      // TODO: Check if this is a void pointer to be extract precise.
      // also check name of struct? Don't know if we can verify that a struct is a lambda though.
      if (is_last_member_of_struct && arg.is_pointer && name == "Could not get member name") {
        // not always true, but works around a tricky situation with
        // host-device lambdas, which store a host pointer in the
        // final field of the class called "data". We know that device
        // code will never access this. For some reason, "data" is not
        // being put into the dwarf debug info, though, which is
        // strange... Maybe because it is never accessed by the cuda
        // kernel?
        return true;
      } else {
        // Calculate the absolute offset including array offset if inside an array
        size_t offset = global_offset_bytes + arg.offset;
        return std::memcmp(arg1 + offset, arg2 + offset, arg.size) == 0;
      }
    } else if constexpr (std::is_same_v<T, StructType>) {
      // For structs, we need to consider its base offset plus any array offset
      size_t base_offset = global_offset_bytes + arg.offset;
      
      for (size_t i = 0; i < arg.members.size(); ++i) {
        // Pass the adjusted pointers that already include the base_offset
        bool equal = is_equal(arg1,
                              arg2,
                              arg.members[i].second,
                              i == arg.members.size() - 1,
                              base_offset,
                              arg.members[i].first);

        if (!equal) {
          return false;
        } else {
          // base_offset = next_offset;
        }
      }
      return true;
    } else if constexpr (std::is_same_v<T, ArrayType>) {
      // Calculate the base offset of the array
      
      // Calculate the size of each element in this array
      size_t element_size = std::visit([](auto&& element) -> size_t {
        using ElementT = std::decay_t<decltype(element)>;
        if constexpr (std::is_same_v<ElementT, BasicType>) {
          return element.size;
        } else if constexpr (std::is_same_v<ElementT, StructType>) {
          return element.size;
        }
        return 0; // Should not happen
      }, arg.element_type);
      
      // Check each element of the array
      for (size_t i = 0; i < arg.num_elements; ++i) {
        auto &&up_cast = std::visit(conversion_visitor, arg.element_type);
        // Pass the base pointers and let the recursive call handle the element offset

        bool equal = is_equal(arg1, arg2, up_cast, false, global_offset_bytes, std::string("array[") + std::to_string(i) + "]");

        if (!equal) {
          return false;
        } else {
          global_offset_bytes += element_size;
        }
      }

      return true;
    }
    return false;
  }, type_info);
}

bool is_equal(void *arg1, void *arg2, std::variant<BasicType, StructType, ArrayType> info) {
  return is_equal((char *)arg1, (char *)arg2, info, false, 0, "Parameters");
}
