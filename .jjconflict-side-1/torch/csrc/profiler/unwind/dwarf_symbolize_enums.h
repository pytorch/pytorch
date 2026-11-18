#pragma once
#include <torch/csrc/profiler/unwind/unwind_error.h>
#include <cstdint>
#include <optional>

enum {
  DW_TAG_subprogram = 0x2e,
  DW_TAG_inlined_subroutine = 0x1d,
  DW_TAG_compile_unit = 0x11,
  DW_AT_sibling = 0x1, // reference
  DW_AT_name = 0x3, // string
  DW_AT_stmt_list = 0x10, // lineptr
  DW_AT_addr_base = 0x73, // sec_offset
  DW_AT_rnglists_base = 0x74, // sec_offset
  DW_AT_low_pc = 0x11, // address
  DW_AT_high_pc = 0x12, // address
  DW_AT_specification = 0x47, // reference
  DW_AT_abstract_origin = 0x31, // reference
  DW_AT_linkage_name = 0x6e, // string
  DW_AT_ranges = 0x55, // rnglist
  DW_AT_str_offsets_base = 0x72, // sec_offset
  DW_FORM_addr = 0x01,
  DW_FORM_block2 = 0x03,
  DW_FORM_block4 = 0x04,
  DW_FORM_data2 = 0x05,
  DW_FORM_data4 = 0x06,
  DW_FORM_data8 = 0x07,
  DW_FORM_string = 0x08,
  DW_FORM_block = 0x09,
  DW_FORM_block1 = 0x0a,
  DW_FORM_data1 = 0x0b,
  DW_FORM_flag = 0x0c,
  DW_FORM_sdata = 0x0d,
  DW_FORM_strp = 0x0e,
  DW_FORM_udata = 0x0f,
  DW_FORM_ref_addr = 0x10,
  DW_FORM_ref1 = 0x11,
  DW_FORM_ref2 = 0x12,
  DW_FORM_ref4 = 0x13,
  DW_FORM_ref8 = 0x14,
  DW_FORM_ref_udata = 0x15,
  DW_FORM_indirect = 0x16,
  DW_FORM_sec_offset = 0x17,
  DW_FORM_exprloc = 0x18,
  DW_FORM_flag_present = 0x19,
  DW_FORM_strx = 0x1a,
  DW_FORM_addrx = 0x1b,
  DW_FORM_ref_sup4 = 0x1c,
  DW_FORM_strp_sup = 0x1d,
  DW_FORM_data16 = 0x1e,
  DW_FORM_line_strp = 0x1f,
  DW_FORM_ref_sig8 = 0x20,
  DW_FORM_implicit_const = 0x21,
  DW_FORM_loclistx = 0x22,
  DW_FORM_rnglistx = 0x23,
  DW_FORM_ref_sup8 = 0x24,
  DW_FORM_strx1 = 0x25,
  DW_FORM_strx2 = 0x26,
  DW_FORM_strx3 = 0x27,
  DW_FORM_strx4 = 0x28,
  DW_FORM_addrx1 = 0x29,
  DW_FORM_addrx2 = 0x2a,
  DW_FORM_addrx3 = 0x2b,
  DW_FORM_addrx4 = 0x2c,
  /* GNU Debug Fission extensions.  */
  DW_FORM_GNU_addr_index = 0x1f01,
  DW_FORM_GNU_str_index = 0x1f02,
  DW_FORM_GNU_ref_alt = 0x1f20, /* offset in alternate .debuginfo.  */
  DW_FORM_GNU_strp_alt = 0x1f21, /* offset in alternate .debug_str. */
  DW_LNCT_path = 0x1,
  DW_LNCT_directory_index = 0x2,
  DW_LNS_extended_op = 0x00,
  DW_LNE_end_sequence = 0x01,
  DW_LNE_set_address = 0x02,
  DW_LNS_copy = 0x01,
  DW_LNS_advance_pc = 0x02,
  DW_LNS_advance_line = 0x03,
  DW_LNS_set_file = 0x04,
  DW_LNS_const_add_pc = 0x08,
  DW_LNS_fixed_advance_pc = 0x09,
  DW_RLE_end_of_list = 0x0,
  DW_RLE_base_addressx = 0x1,
  DW_RLE_startx_endx = 0x2,
  DW_RLE_startx_length = 0x3,
  DW_RLE_offset_pair = 0x4,
  DW_RLE_base_address = 0x5,
  DW_RLE_start_end = 0x6,
  DW_RLE_start_length = 0x7
};

static std::optional<size_t> formSize(uint64_t form, uint8_t sec_offset_size) {
  switch (form) {
    case DW_FORM_addr:
      return sizeof(void*);
    case DW_FORM_block2:
    case DW_FORM_block4:
      return std::nullopt;
    case DW_FORM_data2:
      return 2;
    case DW_FORM_data4:
      return 4;
    case DW_FORM_data8:
      return 8;
    case DW_FORM_string:
    case DW_FORM_block:
    case DW_FORM_block1:
      return std::nullopt;
    case DW_FORM_data1:
    case DW_FORM_flag:
      return 1;
    case DW_FORM_sdata:
      return std::nullopt;
    case DW_FORM_strp:
      return sec_offset_size;
    case DW_FORM_udata:
      return std::nullopt;
    case DW_FORM_ref_addr:
      return sec_offset_size;
    case DW_FORM_ref1:
      return 1;
    case DW_FORM_ref2:
      return 2;
    case DW_FORM_ref4:
      return 4;
    case DW_FORM_ref8:
      return 8;
    case DW_FORM_ref_udata:
    case DW_FORM_indirect:
      return std::nullopt;
    case DW_FORM_sec_offset:
      return sec_offset_size;
    case DW_FORM_exprloc:
      return std::nullopt;
    case DW_FORM_flag_present:
      return 0;
    case DW_FORM_strx:
    case DW_FORM_addrx:
      return std::nullopt;
    case DW_FORM_ref_sup4:
      return 4;
    case DW_FORM_strp_sup:
      return sec_offset_size;
    case DW_FORM_data16:
      return 16;
    case DW_FORM_line_strp:
      return sec_offset_size;
    case DW_FORM_ref_sig8:
      return 8;
    case DW_FORM_implicit_const:
      return 0;
    case DW_FORM_loclistx:
    case DW_FORM_rnglistx:
      return std::nullopt;
    case DW_FORM_ref_sup8:
      return 8;
    case DW_FORM_strx1:
      return 1;
    case DW_FORM_strx2:
      return 2;
    case DW_FORM_strx3:
      return 3;
    case DW_FORM_strx4:
      return 4;
    case DW_FORM_addrx1:
      return 1;
    case DW_FORM_addrx2:
      return 2;
    case DW_FORM_addrx3:
      return 3;
    case DW_FORM_addrx4:
      return 4;
    case DW_FORM_GNU_addr_index:
    case DW_FORM_GNU_str_index:
    case DW_FORM_GNU_ref_alt:
    case DW_FORM_GNU_strp_alt:
    default:
      return std::nullopt;
  }
}
