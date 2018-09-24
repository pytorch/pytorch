/* A centralized location to provide legacy macro support, and a warning about
 * when this legacy compatibility symbol is going to removed in the future.
 *
 * Do NOT include this file directly. Instead, use c10/macros/Macros.h
 */

#pragma once

// Note: this is for caffe2/*. Will need to codemod to use direct C10.
#define CAFFE2_EXPORT C10_EXPORT
#define CAFFE2_IMPORT C10_IMPORT

// Note: this is for aten/src/*. Will need to codemod.
#define AT_CORE_API CAFFE2_API
#define AT_CORE_EXPORT C10_EXPORT
#define AT_CORE_IMPORT C10_IMPORT

// Note: this is for both aten and c2, due to cross reference between c2 and
// aten that we try to unentangle. Will need to codemod.
#define AT_DISABLE_COPY_AND_ASSIGN C10_DISABLE_COPY_AND_ASSIGN
