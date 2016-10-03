#ifndef THS_GENERIC_FILE
#define THS_GENERIC_FILE "generic/THSTensorMath.h"
#else

/* operations
 * S indicates sparseness
 * t is one of matrix (m) or vector (v)
 * OP is the operation implemented
 *
 * Opcode grammar:
 *    [S]OP[S]t[S]t
 * For example, SgemSm = sparse output, Y = dense * sparse matrix
 */

// dense = dense + real * sparse * dense
TH_API void THSTensor_(addSmm)(THTensor *r_, real beta, THTensor *t, real alpha, THSTensor *sparse, THTensor *dense);
// sparse = dense * sparse
TH_API void THSTensor_(SgemSm)(THSTensor *r_, THTensor *mat1, THTensor *mat2);

#endif

