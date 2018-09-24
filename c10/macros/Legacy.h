/* A centralized location to provide legacy macro support, and a warning about
 * when this legacy compatibility symbol is going to removed in the future.
 *
 * Do NOT include this file directly. Instead, use c10/macros/Macros.h
 */

#pragma once

// Note: this is for both aten and c2, due to cross reference between c2 and
// aten that we try to unentangle. Will need to codemod.
#define AT_DISABLE_COPY_AND_ASSIGN C10_DISABLE_COPY_AND_ASSIGN
