/* -*- c -*-
   ----------------------------------------------------------------

   Notice that the following BSD-style license applies to this one
   file (valgrind.h) only.  The rest of Valgrind is licensed under the
   terms of the GNU General Public License, version 2, unless
   otherwise indicated.  See the COPYING file in the source
   distribution for details.

   ----------------------------------------------------------------

   This file is part of Valgrind, a dynamic binary instrumentation
   framework.

   Copyright (C) 2000-2017 Julian Seward.  All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

   2. The origin of this software must not be misrepresented; you must
      not claim that you wrote the original software.  If you use this
      software in a product, an acknowledgment in the product
      documentation would be appreciated but is not required.

   3. Altered source versions must be plainly marked as such, and must
      not be misrepresented as being the original software.

   4. The name of the author may not be used to endorse or promote
      products derived from this software without specific prior written
      permission.

   THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
   OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
   DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
   GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   ----------------------------------------------------------------

   Notice that the above BSD-style license applies to this one file
   (valgrind.h) only.  The entire rest of Valgrind is licensed under
   the terms of the GNU General Public License, version 2.  See the
   COPYING file in the source distribution for details.

   ----------------------------------------------------------------
*/


/*

NB: Much of this file has been removed. It is only used to access
    VG_USERREQ_TOOL_BASE and VALGRIND_DO_CLIENT_REQUEST_STMT.

*/


/* This file is for inclusion into client (your!) code.

   You can use these macros to manipulate and query Valgrind's
   execution inside your own programs.

   The resulting executables will still run without Valgrind, just a
   little bit more slowly than they otherwise would, but otherwise
   unchanged.  When not running on valgrind, each client request
   consumes very few (eg. 7) instructions, so the resulting performance
   loss is negligible unless you plan to execute client requests
   millions of times per second.  Nevertheless, if that is still a
   problem, you can compile with the NVALGRIND symbol defined (gcc
   -DNVALGRIND) so that client requests are not even compiled in.  */

#ifndef __VALGRIND_H
#define __VALGRIND_H


/* ------------------------------------------------------------------ */
/* VERSION NUMBER OF VALGRIND                                         */
/* ------------------------------------------------------------------ */

/* Specify Valgrind's version number, so that user code can
   conditionally compile based on our version number.  Note that these
   were introduced at version 3.6 and so do not exist in version 3.5
   or earlier.  The recommended way to use them to check for "version
   X.Y or later" is (eg)

#if defined(__VALGRIND_MAJOR__) && defined(__VALGRIND_MINOR__)   \
    && (__VALGRIND_MAJOR__ > 3                                   \
        || (__VALGRIND_MAJOR__ == 3 && __VALGRIND_MINOR__ >= 6))
*/
#define __VALGRIND_MAJOR__    3
#define __VALGRIND_MINOR__    15


#include <stdarg.h>

/* Nb: this file might be included in a file compiled with -ansi.  So
   we can't use C++ style "//" comments nor the "asm" keyword (instead
   use "__asm__"). */

/* Derive some tags indicating what the target platform is.  Note
   that in this file we're using the compiler's CPP symbols for
   identifying architectures, which are different to the ones we use
   within the rest of Valgrind.  Note, __powerpc__ is active for both
   32 and 64-bit PPC, whereas __powerpc64__ is only active for the
   latter (on Linux, that is).

   Misc note: how to find out what's predefined in gcc by default:
   gcc -Wp,-dM somefile.c
*/
#undef PLAT_x86_darwin
#undef PLAT_amd64_darwin
#undef PLAT_x86_win32
#undef PLAT_amd64_win64
#undef PLAT_x86_linux
#undef PLAT_amd64_linux
#undef PLAT_ppc32_linux
#undef PLAT_ppc64be_linux
#undef PLAT_ppc64le_linux
#undef PLAT_arm_linux
#undef PLAT_arm64_linux
#undef PLAT_s390x_linux
#undef PLAT_mips32_linux
#undef PLAT_mips64_linux
#undef PLAT_x86_solaris
#undef PLAT_amd64_solaris


#if defined(__APPLE__) && defined(__i386__)
#  define PLAT_x86_darwin 1
#elif defined(__APPLE__) && defined(__x86_64__)
#  define PLAT_amd64_darwin 1
#elif (defined(__MINGW32__) && !defined(__MINGW64__)) \
      || defined(__CYGWIN32__) \
      || (defined(_WIN32) && defined(_M_IX86))
#  define PLAT_x86_win32 1
#elif defined(__MINGW64__) \
      || (defined(_WIN64) && defined(_M_X64))
#  define PLAT_amd64_win64 1
#elif defined(__linux__) && defined(__i386__)
#  define PLAT_x86_linux 1
#elif defined(__linux__) && defined(__x86_64__) && !defined(__ILP32__)
#  define PLAT_amd64_linux 1
#elif defined(__linux__) && defined(__powerpc__) && !defined(__powerpc64__)
#  define PLAT_ppc32_linux 1
#elif defined(__linux__) && defined(__powerpc__) && defined(__powerpc64__) && _CALL_ELF != 2
/* Big Endian uses ELF version 1 */
#  define PLAT_ppc64be_linux 1
#elif defined(__linux__) && defined(__powerpc__) && defined(__powerpc64__) && _CALL_ELF == 2
/* Little Endian uses ELF version 2 */
#  define PLAT_ppc64le_linux 1
#elif defined(__linux__) && defined(__arm__) && !defined(__aarch64__)
#  define PLAT_arm_linux 1
#elif defined(__linux__) && defined(__aarch64__) && !defined(__arm__)
#  define PLAT_arm64_linux 1
#elif defined(__linux__) && defined(__s390__) && defined(__s390x__)
#  define PLAT_s390x_linux 1
#elif defined(__linux__) && defined(__mips__) && (__mips==64)
#  define PLAT_mips64_linux 1
#elif defined(__linux__) && defined(__mips__) && (__mips!=64)
#  define PLAT_mips32_linux 1
#elif defined(__sun) && defined(__i386__)
#  define PLAT_x86_solaris 1
#elif defined(__sun) && defined(__x86_64__)
#  define PLAT_amd64_solaris 1
#else
/* If we're not compiling for our target platform, don't generate
   any inline asms.  */
#  if !defined(NVALGRIND)
#    define NVALGRIND 1
#  endif
#endif


/* ------------------------------------------------------------------ */
/* ARCHITECTURE SPECIFICS for SPECIAL INSTRUCTIONS.  There is nothing */
/* in here of use to end-users -- skip to the next section.           */
/* ------------------------------------------------------------------ */

/*
 * VALGRIND_DO_CLIENT_REQUEST(): a statement that invokes a Valgrind client
 * request. Accepts both pointers and integers as arguments.
 *
 * VALGRIND_DO_CLIENT_REQUEST_STMT(): a statement that invokes a Valgrind
 * client request that does not return a value.

 * VALGRIND_DO_CLIENT_REQUEST_EXPR(): a C expression that invokes a Valgrind
 * client request and whose value equals the client request result.  Accepts
 * both pointers and integers as arguments.  Note that such calls are not
 * necessarily pure functions -- they may have side effects.
 */

#define VALGRIND_DO_CLIENT_REQUEST(_zzq_rlval, _zzq_default,            \
                                   _zzq_request, _zzq_arg1, _zzq_arg2,  \
                                   _zzq_arg3, _zzq_arg4, _zzq_arg5)     \
  do { (_zzq_rlval) = VALGRIND_DO_CLIENT_REQUEST_EXPR((_zzq_default),   \
                        (_zzq_request), (_zzq_arg1), (_zzq_arg2),       \
                        (_zzq_arg3), (_zzq_arg4), (_zzq_arg5)); } while (0)

#define VALGRIND_DO_CLIENT_REQUEST_STMT(_zzq_request, _zzq_arg1,        \
                           _zzq_arg2,  _zzq_arg3, _zzq_arg4, _zzq_arg5) \
  do { (void) VALGRIND_DO_CLIENT_REQUEST_EXPR(0,                        \
                    (_zzq_request), (_zzq_arg1), (_zzq_arg2),           \
                    (_zzq_arg3), (_zzq_arg4), (_zzq_arg5)); } while (0)

#if defined(NVALGRIND)

/* Define NVALGRIND to completely remove the Valgrind magic sequence
   from the compiled code (analogous to NDEBUG's effects on
   assert()) */
#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                          \
        _zzq_default, _zzq_request,                               \
        _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)    \
      (_zzq_default)

#else  /* ! NVALGRIND */

/* The following defines the magic code sequences which the JITter
   spots and handles magically.  Don't look too closely at them as
   they will rot your brain.

   The assembly code sequences for all architectures is in this one
   file.  This is because this file must be stand-alone, and we don't
   want to have multiple files.

   For VALGRIND_DO_CLIENT_REQUEST, we must ensure that the default
   value gets put in the return slot, so that everything works when
   this is executed not under Valgrind.  Args are passed in a memory
   block, and so there's no intrinsic limit to the number that could
   be passed, but it's currently five.

   The macro args are:
      _zzq_rlval    result lvalue
      _zzq_default  default value (result returned when running on real CPU)
      _zzq_request  request code
      _zzq_arg1..5  request params

   The other two macros are used to support function wrapping, and are
   a lot simpler.  VALGRIND_GET_NR_CONTEXT returns the value of the
   guest's NRADDR pseudo-register and whatever other information is
   needed to safely run the call original from the wrapper: on
   ppc64-linux, the R2 value at the divert point is also needed.  This
   information is abstracted into a user-visible type, OrigFn.

   VALGRIND_CALL_NOREDIR_* behaves the same as the following on the
   guest, but guarantees that the branch instruction will not be
   redirected: x86: call *%eax, amd64: call *%rax, ppc32/ppc64:
   branch-and-link-to-r11.  VALGRIND_CALL_NOREDIR is just text, not a
   complete inline asm, since it needs to be combined with more magic
   inline asm stuff to be useful.
*/

/* ----------------- x86-{linux,darwin,solaris} ---------------- */

#if defined(PLAT_x86_linux)  ||  defined(PLAT_x86_darwin)  \
    ||  (defined(PLAT_x86_win32) && defined(__GNUC__)) \
    ||  defined(PLAT_x86_solaris)

typedef
   struct {
      unsigned int nraddr; /* where's the code? */
   }
   OrigFn;

#define __SPECIAL_INSTRUCTION_PREAMBLE                            \
                     "roll $3,  %%edi ; roll $13, %%edi\n\t"      \
                     "roll $29, %%edi ; roll $19, %%edi\n\t"

#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                          \
        _zzq_default, _zzq_request,                               \
        _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)    \
  __extension__                                                   \
  ({volatile unsigned int _zzq_args[6];                           \
    volatile unsigned int _zzq_result;                            \
    _zzq_args[0] = (unsigned int)(_zzq_request);                  \
    _zzq_args[1] = (unsigned int)(_zzq_arg1);                     \
    _zzq_args[2] = (unsigned int)(_zzq_arg2);                     \
    _zzq_args[3] = (unsigned int)(_zzq_arg3);                     \
    _zzq_args[4] = (unsigned int)(_zzq_arg4);                     \
    _zzq_args[5] = (unsigned int)(_zzq_arg5);                     \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %EDX = client_request ( %EAX ) */         \
                     "xchgl %%ebx,%%ebx"                          \
                     : "=d" (_zzq_result)                         \
                     : "a" (&_zzq_args[0]), "0" (_zzq_default)    \
                     : "cc", "memory"                             \
                    );                                            \
    _zzq_result;                                                  \
  })

#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                       \
  { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                   \
    volatile unsigned int __addr;                                 \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %EAX = guest_NRADDR */                    \
                     "xchgl %%ecx,%%ecx"                          \
                     : "=a" (__addr)                              \
                     :                                            \
                     : "cc", "memory"                             \
                    );                                            \
    _zzq_orig->nraddr = __addr;                                   \
  }

#define VALGRIND_CALL_NOREDIR_EAX                                 \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* call-noredir *%EAX */                     \
                     "xchgl %%edx,%%edx\n\t"

#define VALGRIND_VEX_INJECT_IR()                                 \
 do {                                                            \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE              \
                     "xchgl %%edi,%%edi\n\t"                     \
                     : : : "cc", "memory"                        \
                    );                                           \
 } while (0)

#endif /* PLAT_x86_linux || PLAT_x86_darwin || (PLAT_x86_win32 && __GNUC__)
          || PLAT_x86_solaris */

/* ------------------------- x86-Win32 ------------------------- */

#if defined(PLAT_x86_win32) && !defined(__GNUC__)

typedef
   struct {
      unsigned int nraddr; /* where's the code? */
   }
   OrigFn;

#if defined(_MSC_VER)

#define __SPECIAL_INSTRUCTION_PREAMBLE                            \
                     __asm rol edi, 3  __asm rol edi, 13          \
                     __asm rol edi, 29 __asm rol edi, 19

#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                          \
        _zzq_default, _zzq_request,                               \
        _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)    \
    valgrind_do_client_request_expr((uintptr_t)(_zzq_default),    \
        (uintptr_t)(_zzq_request), (uintptr_t)(_zzq_arg1),        \
        (uintptr_t)(_zzq_arg2), (uintptr_t)(_zzq_arg3),           \
        (uintptr_t)(_zzq_arg4), (uintptr_t)(_zzq_arg5))

static __inline uintptr_t
valgrind_do_client_request_expr(uintptr_t _zzq_default, uintptr_t _zzq_request,
                                uintptr_t _zzq_arg1, uintptr_t _zzq_arg2,
                                uintptr_t _zzq_arg3, uintptr_t _zzq_arg4,
                                uintptr_t _zzq_arg5)
{
    volatile uintptr_t _zzq_args[6];
    volatile unsigned int _zzq_result;
    _zzq_args[0] = (uintptr_t)(_zzq_request);
    _zzq_args[1] = (uintptr_t)(_zzq_arg1);
    _zzq_args[2] = (uintptr_t)(_zzq_arg2);
    _zzq_args[3] = (uintptr_t)(_zzq_arg3);
    _zzq_args[4] = (uintptr_t)(_zzq_arg4);
    _zzq_args[5] = (uintptr_t)(_zzq_arg5);
    __asm { __asm lea eax, _zzq_args __asm mov edx, _zzq_default
            __SPECIAL_INSTRUCTION_PREAMBLE
            /* %EDX = client_request ( %EAX ) */
            __asm xchg ebx,ebx
            __asm mov _zzq_result, edx
    }
    return _zzq_result;
}

#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                       \
  { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                   \
    volatile unsigned int __addr;                                 \
    __asm { __SPECIAL_INSTRUCTION_PREAMBLE                        \
            /* %EAX = guest_NRADDR */                             \
            __asm xchg ecx,ecx                                    \
            __asm mov __addr, eax                                 \
    }                                                             \
    _zzq_orig->nraddr = __addr;                                   \
  }

#define VALGRIND_CALL_NOREDIR_EAX ERROR

#define VALGRIND_VEX_INJECT_IR()                                 \
 do {                                                            \
    __asm { __SPECIAL_INSTRUCTION_PREAMBLE                       \
            __asm xchg edi,edi                                   \
    }                                                            \
 } while (0)

#else
#error Unsupported compiler.
#endif

#endif /* PLAT_x86_win32 */

/* ----------------- amd64-{linux,darwin,solaris} --------------- */

#if defined(PLAT_amd64_linux)  ||  defined(PLAT_amd64_darwin) \
    ||  defined(PLAT_amd64_solaris) \
    ||  (defined(PLAT_amd64_win64) && defined(__GNUC__))

typedef
   struct {
      unsigned long int nraddr; /* where's the code? */
   }
   OrigFn;

#define __SPECIAL_INSTRUCTION_PREAMBLE                            \
                     "rolq $3,  %%rdi ; rolq $13, %%rdi\n\t"      \
                     "rolq $61, %%rdi ; rolq $51, %%rdi\n\t"

#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                          \
        _zzq_default, _zzq_request,                               \
        _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)    \
    __extension__                                                 \
    ({ volatile unsigned long int _zzq_args[6];                   \
    volatile unsigned long int _zzq_result;                       \
    _zzq_args[0] = (unsigned long int)(_zzq_request);             \
    _zzq_args[1] = (unsigned long int)(_zzq_arg1);                \
    _zzq_args[2] = (unsigned long int)(_zzq_arg2);                \
    _zzq_args[3] = (unsigned long int)(_zzq_arg3);                \
    _zzq_args[4] = (unsigned long int)(_zzq_arg4);                \
    _zzq_args[5] = (unsigned long int)(_zzq_arg5);                \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %RDX = client_request ( %RAX ) */         \
                     "xchgq %%rbx,%%rbx"                          \
                     : "=d" (_zzq_result)                         \
                     : "a" (&_zzq_args[0]), "0" (_zzq_default)    \
                     : "cc", "memory"                             \
                    );                                            \
    _zzq_result;                                                  \
    })

#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                       \
  { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                   \
    volatile unsigned long int __addr;                            \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %RAX = guest_NRADDR */                    \
                     "xchgq %%rcx,%%rcx"                          \
                     : "=a" (__addr)                              \
                     :                                            \
                     : "cc", "memory"                             \
                    );                                            \
    _zzq_orig->nraddr = __addr;                                   \
  }

#define VALGRIND_CALL_NOREDIR_RAX                                 \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* call-noredir *%RAX */                     \
                     "xchgq %%rdx,%%rdx\n\t"

#define VALGRIND_VEX_INJECT_IR()                                 \
 do {                                                            \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE              \
                     "xchgq %%rdi,%%rdi\n\t"                     \
                     : : : "cc", "memory"                        \
                    );                                           \
 } while (0)

#endif /* PLAT_amd64_linux || PLAT_amd64_darwin || PLAT_amd64_solaris */

/* ------------------------- amd64-Win64 ------------------------- */

#if defined(PLAT_amd64_win64) && !defined(__GNUC__)

#error Unsupported compiler.

#endif /* PLAT_amd64_win64 */

/* ------------------------ ppc32-linux ------------------------ */

#if defined(PLAT_ppc32_linux)

typedef
   struct {
      unsigned int nraddr; /* where's the code? */
   }
   OrigFn;

#define __SPECIAL_INSTRUCTION_PREAMBLE                            \
                    "rlwinm 0,0,3,0,31  ; rlwinm 0,0,13,0,31\n\t" \
                    "rlwinm 0,0,29,0,31 ; rlwinm 0,0,19,0,31\n\t"

#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                          \
        _zzq_default, _zzq_request,                               \
        _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)    \
                                                                  \
    __extension__                                                 \
  ({         unsigned int  _zzq_args[6];                          \
             unsigned int  _zzq_result;                           \
             unsigned int* _zzq_ptr;                              \
    _zzq_args[0] = (unsigned int)(_zzq_request);                  \
    _zzq_args[1] = (unsigned int)(_zzq_arg1);                     \
    _zzq_args[2] = (unsigned int)(_zzq_arg2);                     \
    _zzq_args[3] = (unsigned int)(_zzq_arg3);                     \
    _zzq_args[4] = (unsigned int)(_zzq_arg4);                     \
    _zzq_args[5] = (unsigned int)(_zzq_arg5);                     \
    _zzq_ptr = _zzq_args;                                         \
    __asm__ volatile("mr 3,%1\n\t" /*default*/                    \
                     "mr 4,%2\n\t" /*ptr*/                        \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %R3 = client_request ( %R4 ) */           \
                     "or 1,1,1\n\t"                               \
                     "mr %0,3"     /*result*/                     \
                     : "=b" (_zzq_result)                         \
                     : "b" (_zzq_default), "b" (_zzq_ptr)         \
                     : "cc", "memory", "r3", "r4");               \
    _zzq_result;                                                  \
    })

#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                       \
  { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                   \
    unsigned int __addr;                                          \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %R3 = guest_NRADDR */                     \
                     "or 2,2,2\n\t"                               \
                     "mr %0,3"                                    \
                     : "=b" (__addr)                              \
                     :                                            \
                     : "cc", "memory", "r3"                       \
                    );                                            \
    _zzq_orig->nraddr = __addr;                                   \
  }

#define VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                   \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* branch-and-link-to-noredir *%R11 */       \
                     "or 3,3,3\n\t"

#define VALGRIND_VEX_INJECT_IR()                                 \
 do {                                                            \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE              \
                     "or 5,5,5\n\t"                              \
                    );                                           \
 } while (0)

#endif /* PLAT_ppc32_linux */

/* ------------------------ ppc64-linux ------------------------ */

#if defined(PLAT_ppc64be_linux)

typedef
   struct {
      unsigned long int nraddr; /* where's the code? */
      unsigned long int r2;  /* what tocptr do we need? */
   }
   OrigFn;

#define __SPECIAL_INSTRUCTION_PREAMBLE                            \
                     "rotldi 0,0,3  ; rotldi 0,0,13\n\t"          \
                     "rotldi 0,0,61 ; rotldi 0,0,51\n\t"

#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                          \
        _zzq_default, _zzq_request,                               \
        _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)    \
                                                                  \
  __extension__                                                   \
  ({         unsigned long int  _zzq_args[6];                     \
             unsigned long int  _zzq_result;                      \
             unsigned long int* _zzq_ptr;                         \
    _zzq_args[0] = (unsigned long int)(_zzq_request);             \
    _zzq_args[1] = (unsigned long int)(_zzq_arg1);                \
    _zzq_args[2] = (unsigned long int)(_zzq_arg2);                \
    _zzq_args[3] = (unsigned long int)(_zzq_arg3);                \
    _zzq_args[4] = (unsigned long int)(_zzq_arg4);                \
    _zzq_args[5] = (unsigned long int)(_zzq_arg5);                \
    _zzq_ptr = _zzq_args;                                         \
    __asm__ volatile("mr 3,%1\n\t" /*default*/                    \
                     "mr 4,%2\n\t" /*ptr*/                        \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %R3 = client_request ( %R4 ) */           \
                     "or 1,1,1\n\t"                               \
                     "mr %0,3"     /*result*/                     \
                     : "=b" (_zzq_result)                         \
                     : "b" (_zzq_default), "b" (_zzq_ptr)         \
                     : "cc", "memory", "r3", "r4");               \
    _zzq_result;                                                  \
  })

#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                       \
  { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                   \
    unsigned long int __addr;                                     \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %R3 = guest_NRADDR */                     \
                     "or 2,2,2\n\t"                               \
                     "mr %0,3"                                    \
                     : "=b" (__addr)                              \
                     :                                            \
                     : "cc", "memory", "r3"                       \
                    );                                            \
    _zzq_orig->nraddr = __addr;                                   \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %R3 = guest_NRADDR_GPR2 */                \
                     "or 4,4,4\n\t"                               \
                     "mr %0,3"                                    \
                     : "=b" (__addr)                              \
                     :                                            \
                     : "cc", "memory", "r3"                       \
                    );                                            \
    _zzq_orig->r2 = __addr;                                       \
  }

#define VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R11                   \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* branch-and-link-to-noredir *%R11 */       \
                     "or 3,3,3\n\t"

#define VALGRIND_VEX_INJECT_IR()                                 \
 do {                                                            \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE              \
                     "or 5,5,5\n\t"                              \
                    );                                           \
 } while (0)

#endif /* PLAT_ppc64be_linux */

#if defined(PLAT_ppc64le_linux)

typedef
   struct {
      unsigned long int nraddr; /* where's the code? */
      unsigned long int r2;     /* what tocptr do we need? */
   }
   OrigFn;

#define __SPECIAL_INSTRUCTION_PREAMBLE                            \
                     "rotldi 0,0,3  ; rotldi 0,0,13\n\t"          \
                     "rotldi 0,0,61 ; rotldi 0,0,51\n\t"

#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                          \
        _zzq_default, _zzq_request,                               \
        _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)    \
                                                                  \
  __extension__                                                   \
  ({         unsigned long int  _zzq_args[6];                     \
             unsigned long int  _zzq_result;                      \
             unsigned long int* _zzq_ptr;                         \
    _zzq_args[0] = (unsigned long int)(_zzq_request);             \
    _zzq_args[1] = (unsigned long int)(_zzq_arg1);                \
    _zzq_args[2] = (unsigned long int)(_zzq_arg2);                \
    _zzq_args[3] = (unsigned long int)(_zzq_arg3);                \
    _zzq_args[4] = (unsigned long int)(_zzq_arg4);                \
    _zzq_args[5] = (unsigned long int)(_zzq_arg5);                \
    _zzq_ptr = _zzq_args;                                         \
    __asm__ volatile("mr 3,%1\n\t" /*default*/                    \
                     "mr 4,%2\n\t" /*ptr*/                        \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %R3 = client_request ( %R4 ) */           \
                     "or 1,1,1\n\t"                               \
                     "mr %0,3"     /*result*/                     \
                     : "=b" (_zzq_result)                         \
                     : "b" (_zzq_default), "b" (_zzq_ptr)         \
                     : "cc", "memory", "r3", "r4");               \
    _zzq_result;                                                  \
  })

#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                       \
  { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                   \
    unsigned long int __addr;                                     \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %R3 = guest_NRADDR */                     \
                     "or 2,2,2\n\t"                               \
                     "mr %0,3"                                    \
                     : "=b" (__addr)                              \
                     :                                            \
                     : "cc", "memory", "r3"                       \
                    );                                            \
    _zzq_orig->nraddr = __addr;                                   \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %R3 = guest_NRADDR_GPR2 */                \
                     "or 4,4,4\n\t"                               \
                     "mr %0,3"                                    \
                     : "=b" (__addr)                              \
                     :                                            \
                     : "cc", "memory", "r3"                       \
                    );                                            \
    _zzq_orig->r2 = __addr;                                       \
  }

#define VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R12                   \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* branch-and-link-to-noredir *%R12 */       \
                     "or 3,3,3\n\t"

#define VALGRIND_VEX_INJECT_IR()                                 \
 do {                                                            \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE              \
                     "or 5,5,5\n\t"                              \
                    );                                           \
 } while (0)

#endif /* PLAT_ppc64le_linux */

/* ------------------------- arm-linux ------------------------- */

#if defined(PLAT_arm_linux)

typedef
   struct {
      unsigned int nraddr; /* where's the code? */
   }
   OrigFn;

#define __SPECIAL_INSTRUCTION_PREAMBLE                            \
            "mov r12, r12, ror #3  ; mov r12, r12, ror #13 \n\t"  \
            "mov r12, r12, ror #29 ; mov r12, r12, ror #19 \n\t"

#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                          \
        _zzq_default, _zzq_request,                               \
        _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)    \
                                                                  \
  __extension__                                                   \
  ({volatile unsigned int  _zzq_args[6];                          \
    volatile unsigned int  _zzq_result;                           \
    _zzq_args[0] = (unsigned int)(_zzq_request);                  \
    _zzq_args[1] = (unsigned int)(_zzq_arg1);                     \
    _zzq_args[2] = (unsigned int)(_zzq_arg2);                     \
    _zzq_args[3] = (unsigned int)(_zzq_arg3);                     \
    _zzq_args[4] = (unsigned int)(_zzq_arg4);                     \
    _zzq_args[5] = (unsigned int)(_zzq_arg5);                     \
    __asm__ volatile("mov r3, %1\n\t" /*default*/                 \
                     "mov r4, %2\n\t" /*ptr*/                     \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* R3 = client_request ( R4 ) */             \
                     "orr r10, r10, r10\n\t"                      \
                     "mov %0, r3"     /*result*/                  \
                     : "=r" (_zzq_result)                         \
                     : "r" (_zzq_default), "r" (&_zzq_args[0])    \
                     : "cc","memory", "r3", "r4");                \
    _zzq_result;                                                  \
  })

#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                       \
  { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                   \
    unsigned int __addr;                                          \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* R3 = guest_NRADDR */                      \
                     "orr r11, r11, r11\n\t"                      \
                     "mov %0, r3"                                 \
                     : "=r" (__addr)                              \
                     :                                            \
                     : "cc", "memory", "r3"                       \
                    );                                            \
    _zzq_orig->nraddr = __addr;                                   \
  }

#define VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_R4                    \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* branch-and-link-to-noredir *%R4 */        \
                     "orr r12, r12, r12\n\t"

#define VALGRIND_VEX_INJECT_IR()                                 \
 do {                                                            \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE              \
                     "orr r9, r9, r9\n\t"                        \
                     : : : "cc", "memory"                        \
                    );                                           \
 } while (0)

#endif /* PLAT_arm_linux */

/* ------------------------ arm64-linux ------------------------- */

#if defined(PLAT_arm64_linux)

typedef
   struct {
      unsigned long int nraddr; /* where's the code? */
   }
   OrigFn;

#define __SPECIAL_INSTRUCTION_PREAMBLE                            \
            "ror x12, x12, #3  ;  ror x12, x12, #13 \n\t"         \
            "ror x12, x12, #51 ;  ror x12, x12, #61 \n\t"

#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                          \
        _zzq_default, _zzq_request,                               \
        _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)    \
                                                                  \
  __extension__                                                   \
  ({volatile unsigned long int  _zzq_args[6];                     \
    volatile unsigned long int  _zzq_result;                      \
    _zzq_args[0] = (unsigned long int)(_zzq_request);             \
    _zzq_args[1] = (unsigned long int)(_zzq_arg1);                \
    _zzq_args[2] = (unsigned long int)(_zzq_arg2);                \
    _zzq_args[3] = (unsigned long int)(_zzq_arg3);                \
    _zzq_args[4] = (unsigned long int)(_zzq_arg4);                \
    _zzq_args[5] = (unsigned long int)(_zzq_arg5);                \
    __asm__ volatile("mov x3, %1\n\t" /*default*/                 \
                     "mov x4, %2\n\t" /*ptr*/                     \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* X3 = client_request ( X4 ) */             \
                     "orr x10, x10, x10\n\t"                      \
                     "mov %0, x3"     /*result*/                  \
                     : "=r" (_zzq_result)                         \
                     : "r" ((unsigned long int)(_zzq_default)),   \
                       "r" (&_zzq_args[0])                        \
                     : "cc","memory", "x3", "x4");                \
    _zzq_result;                                                  \
  })

#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                       \
  { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                   \
    unsigned long int __addr;                                     \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* X3 = guest_NRADDR */                      \
                     "orr x11, x11, x11\n\t"                      \
                     "mov %0, x3"                                 \
                     : "=r" (__addr)                              \
                     :                                            \
                     : "cc", "memory", "x3"                       \
                    );                                            \
    _zzq_orig->nraddr = __addr;                                   \
  }

#define VALGRIND_BRANCH_AND_LINK_TO_NOREDIR_X8                    \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* branch-and-link-to-noredir X8 */          \
                     "orr x12, x12, x12\n\t"

#define VALGRIND_VEX_INJECT_IR()                                 \
 do {                                                            \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE              \
                     "orr x9, x9, x9\n\t"                        \
                     : : : "cc", "memory"                        \
                    );                                           \
 } while (0)

#endif /* PLAT_arm64_linux */

/* ------------------------ s390x-linux ------------------------ */

#if defined(PLAT_s390x_linux)

typedef
  struct {
     unsigned long int nraddr; /* where's the code? */
  }
  OrigFn;

/* __SPECIAL_INSTRUCTION_PREAMBLE will be used to identify Valgrind specific
 * code. This detection is implemented in platform specific toIR.c
 * (e.g. VEX/priv/guest_s390_decoder.c).
 */
#define __SPECIAL_INSTRUCTION_PREAMBLE                           \
                     "lr 15,15\n\t"                              \
                     "lr 1,1\n\t"                                \
                     "lr 2,2\n\t"                                \
                     "lr 3,3\n\t"

#define __CLIENT_REQUEST_CODE "lr 2,2\n\t"
#define __GET_NR_CONTEXT_CODE "lr 3,3\n\t"
#define __CALL_NO_REDIR_CODE  "lr 4,4\n\t"
#define __VEX_INJECT_IR_CODE  "lr 5,5\n\t"

#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                         \
       _zzq_default, _zzq_request,                               \
       _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)    \
  __extension__                                                  \
 ({volatile unsigned long int _zzq_args[6];                      \
   volatile unsigned long int _zzq_result;                       \
   _zzq_args[0] = (unsigned long int)(_zzq_request);             \
   _zzq_args[1] = (unsigned long int)(_zzq_arg1);                \
   _zzq_args[2] = (unsigned long int)(_zzq_arg2);                \
   _zzq_args[3] = (unsigned long int)(_zzq_arg3);                \
   _zzq_args[4] = (unsigned long int)(_zzq_arg4);                \
   _zzq_args[5] = (unsigned long int)(_zzq_arg5);                \
   __asm__ volatile(/* r2 = args */                              \
                    "lgr 2,%1\n\t"                               \
                    /* r3 = default */                           \
                    "lgr 3,%2\n\t"                               \
                    __SPECIAL_INSTRUCTION_PREAMBLE               \
                    __CLIENT_REQUEST_CODE                        \
                    /* results = r3 */                           \
                    "lgr %0, 3\n\t"                              \
                    : "=d" (_zzq_result)                         \
                    : "a" (&_zzq_args[0]), "0" (_zzq_default)    \
                    : "cc", "2", "3", "memory"                   \
                   );                                            \
   _zzq_result;                                                  \
 })

#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                      \
 { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                   \
   volatile unsigned long int __addr;                            \
   __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                    __GET_NR_CONTEXT_CODE                        \
                    "lgr %0, 3\n\t"                              \
                    : "=a" (__addr)                              \
                    :                                            \
                    : "cc", "3", "memory"                        \
                   );                                            \
   _zzq_orig->nraddr = __addr;                                   \
 }

#define VALGRIND_CALL_NOREDIR_R1                                 \
                    __SPECIAL_INSTRUCTION_PREAMBLE               \
                    __CALL_NO_REDIR_CODE

#define VALGRIND_VEX_INJECT_IR()                                 \
 do {                                                            \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE              \
                     __VEX_INJECT_IR_CODE);                      \
 } while (0)

#endif /* PLAT_s390x_linux */

/* ------------------------- mips32-linux ---------------- */

#if defined(PLAT_mips32_linux)

typedef
   struct {
      unsigned int nraddr; /* where's the code? */
   }
   OrigFn;

/* .word  0x342
 * .word  0x742
 * .word  0xC2
 * .word  0x4C2*/
#define __SPECIAL_INSTRUCTION_PREAMBLE          \
                     "srl $0, $0, 13\n\t"       \
                     "srl $0, $0, 29\n\t"       \
                     "srl $0, $0, 3\n\t"        \
                     "srl $0, $0, 19\n\t"

#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                          \
       _zzq_default, _zzq_request,                                \
       _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)     \
  __extension__                                                   \
  ({ volatile unsigned int _zzq_args[6];                          \
    volatile unsigned int _zzq_result;                            \
    _zzq_args[0] = (unsigned int)(_zzq_request);                  \
    _zzq_args[1] = (unsigned int)(_zzq_arg1);                     \
    _zzq_args[2] = (unsigned int)(_zzq_arg2);                     \
    _zzq_args[3] = (unsigned int)(_zzq_arg3);                     \
    _zzq_args[4] = (unsigned int)(_zzq_arg4);                     \
    _zzq_args[5] = (unsigned int)(_zzq_arg5);                     \
        __asm__ volatile("move $11, %1\n\t" /*default*/           \
                     "move $12, %2\n\t" /*ptr*/                   \
                     __SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* T3 = client_request ( T4 ) */             \
                     "or $13, $13, $13\n\t"                       \
                     "move %0, $11\n\t"     /*result*/            \
                     : "=r" (_zzq_result)                         \
                     : "r" (_zzq_default), "r" (&_zzq_args[0])    \
                     : "$11", "$12", "memory");                   \
    _zzq_result;                                                  \
  })

#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                       \
  { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                   \
    volatile unsigned int __addr;                                 \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE               \
                     /* %t9 = guest_NRADDR */                     \
                     "or $14, $14, $14\n\t"                       \
                     "move %0, $11"     /*result*/                \
                     : "=r" (__addr)                              \
                     :                                            \
                     : "$11"                                      \
                    );                                            \
    _zzq_orig->nraddr = __addr;                                   \
  }

#define VALGRIND_CALL_NOREDIR_T9                                 \
                     __SPECIAL_INSTRUCTION_PREAMBLE              \
                     /* call-noredir *%t9 */                     \
                     "or $15, $15, $15\n\t"

#define VALGRIND_VEX_INJECT_IR()                                 \
 do {                                                            \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE              \
                     "or $11, $11, $11\n\t"                      \
                    );                                           \
 } while (0)


#endif /* PLAT_mips32_linux */

/* ------------------------- mips64-linux ---------------- */

#if defined(PLAT_mips64_linux)

typedef
   struct {
      unsigned long nraddr; /* where's the code? */
   }
   OrigFn;

/* dsll $0,$0, 3
 * dsll $0,$0, 13
 * dsll $0,$0, 29
 * dsll $0,$0, 19*/
#define __SPECIAL_INSTRUCTION_PREAMBLE                              \
                     "dsll $0,$0, 3 ; dsll $0,$0,13\n\t"            \
                     "dsll $0,$0,29 ; dsll $0,$0,19\n\t"

#define VALGRIND_DO_CLIENT_REQUEST_EXPR(                            \
       _zzq_default, _zzq_request,                                  \
       _zzq_arg1, _zzq_arg2, _zzq_arg3, _zzq_arg4, _zzq_arg5)       \
  __extension__                                                     \
  ({ volatile unsigned long int _zzq_args[6];                       \
    volatile unsigned long int _zzq_result;                         \
    _zzq_args[0] = (unsigned long int)(_zzq_request);               \
    _zzq_args[1] = (unsigned long int)(_zzq_arg1);                  \
    _zzq_args[2] = (unsigned long int)(_zzq_arg2);                  \
    _zzq_args[3] = (unsigned long int)(_zzq_arg3);                  \
    _zzq_args[4] = (unsigned long int)(_zzq_arg4);                  \
    _zzq_args[5] = (unsigned long int)(_zzq_arg5);                  \
        __asm__ volatile("move $11, %1\n\t" /*default*/             \
                         "move $12, %2\n\t" /*ptr*/                 \
                         __SPECIAL_INSTRUCTION_PREAMBLE             \
                         /* $11 = client_request ( $12 ) */         \
                         "or $13, $13, $13\n\t"                     \
                         "move %0, $11\n\t"     /*result*/          \
                         : "=r" (_zzq_result)                       \
                         : "r" (_zzq_default), "r" (&_zzq_args[0])  \
                         : "$11", "$12", "memory");                 \
    _zzq_result;                                                    \
  })

#define VALGRIND_GET_NR_CONTEXT(_zzq_rlval)                         \
  { volatile OrigFn* _zzq_orig = &(_zzq_rlval);                     \
    volatile unsigned long int __addr;                              \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE                 \
                     /* $11 = guest_NRADDR */                       \
                     "or $14, $14, $14\n\t"                         \
                     "move %0, $11"     /*result*/                  \
                     : "=r" (__addr)                                \
                     :                                              \
                     : "$11");                                      \
    _zzq_orig->nraddr = __addr;                                     \
  }

#define VALGRIND_CALL_NOREDIR_T9                                    \
                     __SPECIAL_INSTRUCTION_PREAMBLE                 \
                     /* call-noredir $25 */                         \
                     "or $15, $15, $15\n\t"

#define VALGRIND_VEX_INJECT_IR()                                    \
 do {                                                               \
    __asm__ volatile(__SPECIAL_INSTRUCTION_PREAMBLE                 \
                     "or $11, $11, $11\n\t"                         \
                    );                                              \
 } while (0)

#endif /* PLAT_mips64_linux */

/* Insert assembly code for other platforms here... */

#endif /* NVALGRIND */

/*





    _____                                                  _
   |  __ \                                                | |
   | |__) |   ___   _ __ ___     ___   __   __   ___    __| |
   |  _  /   / _ \ | '_ ` _ \   / _ \  \ \ / /  / _ \  / _` |
   | | \ \  |  __/ | | | | | | | (_) |  \ V /  |  __/ | (_| |
   |_|  \_\  \___| |_| |_| |_|  \___/    \_/    \___|  \__,_|





*/

/* ------------------------------------------------------------------ */
/* ARCHITECTURE INDEPENDENT MACROS for CLIENT REQUESTS.               */
/*                                                                    */
/* ------------------------------------------------------------------ */

/* Some request codes.  There are many more of these, but most are not
   exposed to end-user view.  These are the public ones, all of the
   form 0x1000 + small_number.

   Core ones are in the range 0x00000000--0x0000ffff.  The non-public
   ones start at 0x2000.
*/

/* These macros are used by tools -- they must be public, but don't
   embed them into other programs. */
#define VG_USERREQ_TOOL_BASE(a,b) \
   ((unsigned int)(((a)&0xff) << 24 | ((b)&0xff) << 16))
#define VG_IS_TOOL_USERREQ(a, b, v) \
   (VG_USERREQ_TOOL_BASE(a,b) == ((v) & 0xffff0000))


/*





    _____                                                  _
   |  __ \                                                | |
   | |__) |   ___   _ __ ___     ___   __   __   ___    __| |
   |  _  /   / _ \ | '_ ` _ \   / _ \  \ \ / /  / _ \  / _` |
   | | \ \  |  __/ | | | | | | | (_) |  \ V /  |  __/ | (_| |
   |_|  \_\  \___| |_| |_| |_|  \___/    \_/    \___|  \__,_|





*/


#undef PLAT_x86_darwin
#undef PLAT_amd64_darwin
#undef PLAT_x86_win32
#undef PLAT_amd64_win64
#undef PLAT_x86_linux
#undef PLAT_amd64_linux
#undef PLAT_ppc32_linux
#undef PLAT_ppc64be_linux
#undef PLAT_ppc64le_linux
#undef PLAT_arm_linux
#undef PLAT_s390x_linux
#undef PLAT_mips32_linux
#undef PLAT_mips64_linux
#undef PLAT_x86_solaris
#undef PLAT_amd64_solaris

#endif   /* __VALGRIND_H */
