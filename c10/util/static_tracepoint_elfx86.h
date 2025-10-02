#pragma once

// clang-format off

// Default constraint for the probe arguments as operands.
#ifndef TORCH_SDT_ARG_CONSTRAINT
#define TORCH_SDT_ARG_CONSTRAINT      "nor"
#endif

// Instruction to emit for the probe.
#define TORCH_SDT_NOP                 nop

// Note section properties.
#define TORCH_SDT_NOTE_NAME           "stapsdt"
#define TORCH_SDT_NOTE_TYPE           3

// Semaphore variables are put in this section
#define TORCH_SDT_SEMAPHORE_SECTION   ".probes"

// Size of address depending on platform.
#ifdef __LP64__
#define TORCH_SDT_ASM_ADDR            .8byte
#else
#define TORCH_SDT_ASM_ADDR            .4byte
#endif

// Assembler helper Macros.
#define TORCH_SDT_S(x)                #x
#define TORCH_SDT_ASM_1(x)            TORCH_SDT_S(x) "\n"
#define TORCH_SDT_ASM_2(a, b)         TORCH_SDT_S(a) "," TORCH_SDT_S(b) "\n"
#define TORCH_SDT_ASM_3(a, b, c)      TORCH_SDT_S(a) "," TORCH_SDT_S(b) ","    \
                                      TORCH_SDT_S(c) "\n"
#define TORCH_SDT_ASM_STRING(x)       TORCH_SDT_ASM_1(.asciz TORCH_SDT_S(x))

// Helper to determine the size of an argument.
#define TORCH_SDT_IS_ARRAY_POINTER(x)  ((__builtin_classify_type(x) == 14) ||  \
                                        (__builtin_classify_type(x) == 5))
#define TORCH_SDT_ARGSIZE(x)  (TORCH_SDT_IS_ARRAY_POINTER(x)                   \
                               ? sizeof(void*)                                 \
                               : sizeof(x))

// Format of each probe arguments as operand.
// Size of the argument tagged with TORCH_SDT_Sn, with "n" constraint.
// Value of the argument tagged with TORCH_SDT_An, with configured constraint.
#define TORCH_SDT_ARG(n, x)                                                    \
  [TORCH_SDT_S##n] "n"                ((size_t)TORCH_SDT_ARGSIZE(x)),          \
  [TORCH_SDT_A##n] TORCH_SDT_ARG_CONSTRAINT (x)

// Templates to append arguments as operands.
#define TORCH_SDT_OPERANDS_0()        [__sdt_dummy] "g" (0)
#define TORCH_SDT_OPERANDS_1(_1)      TORCH_SDT_ARG(1, _1)
#define TORCH_SDT_OPERANDS_2(_1, _2)                                           \
  TORCH_SDT_OPERANDS_1(_1), TORCH_SDT_ARG(2, _2)
#define TORCH_SDT_OPERANDS_3(_1, _2, _3)                                       \
  TORCH_SDT_OPERANDS_2(_1, _2), TORCH_SDT_ARG(3, _3)
#define TORCH_SDT_OPERANDS_4(_1, _2, _3, _4)                                   \
  TORCH_SDT_OPERANDS_3(_1, _2, _3), TORCH_SDT_ARG(4, _4)
#define TORCH_SDT_OPERANDS_5(_1, _2, _3, _4, _5)                               \
  TORCH_SDT_OPERANDS_4(_1, _2, _3, _4), TORCH_SDT_ARG(5, _5)
#define TORCH_SDT_OPERANDS_6(_1, _2, _3, _4, _5, _6)                           \
  TORCH_SDT_OPERANDS_5(_1, _2, _3, _4, _5), TORCH_SDT_ARG(6, _6)
#define TORCH_SDT_OPERANDS_7(_1, _2, _3, _4, _5, _6, _7)                       \
  TORCH_SDT_OPERANDS_6(_1, _2, _3, _4, _5, _6), TORCH_SDT_ARG(7, _7)
#define TORCH_SDT_OPERANDS_8(_1, _2, _3, _4, _5, _6, _7, _8)                   \
  TORCH_SDT_OPERANDS_7(_1, _2, _3, _4, _5, _6, _7), TORCH_SDT_ARG(8, _8)
#define TORCH_SDT_OPERANDS_9(_1, _2, _3, _4, _5, _6, _7, _8, _9)               \
  TORCH_SDT_OPERANDS_8(_1, _2, _3, _4, _5, _6, _7, _8), TORCH_SDT_ARG(9, _9)

// Templates to reference the arguments from operands in note section.
#define TORCH_SDT_ARGFMT(no)        %n[TORCH_SDT_S##no]@%[TORCH_SDT_A##no]
#define TORCH_SDT_ARG_TEMPLATE_0    /*No arguments*/
#define TORCH_SDT_ARG_TEMPLATE_1    TORCH_SDT_ARGFMT(1)
#define TORCH_SDT_ARG_TEMPLATE_2    TORCH_SDT_ARG_TEMPLATE_1 TORCH_SDT_ARGFMT(2)
#define TORCH_SDT_ARG_TEMPLATE_3    TORCH_SDT_ARG_TEMPLATE_2 TORCH_SDT_ARGFMT(3)
#define TORCH_SDT_ARG_TEMPLATE_4    TORCH_SDT_ARG_TEMPLATE_3 TORCH_SDT_ARGFMT(4)
#define TORCH_SDT_ARG_TEMPLATE_5    TORCH_SDT_ARG_TEMPLATE_4 TORCH_SDT_ARGFMT(5)
#define TORCH_SDT_ARG_TEMPLATE_6    TORCH_SDT_ARG_TEMPLATE_5 TORCH_SDT_ARGFMT(6)
#define TORCH_SDT_ARG_TEMPLATE_7    TORCH_SDT_ARG_TEMPLATE_6 TORCH_SDT_ARGFMT(7)
#define TORCH_SDT_ARG_TEMPLATE_8    TORCH_SDT_ARG_TEMPLATE_7 TORCH_SDT_ARGFMT(8)
#define TORCH_SDT_ARG_TEMPLATE_9    TORCH_SDT_ARG_TEMPLATE_8 TORCH_SDT_ARGFMT(9)

// Resolvable by name macros
// An attribute that marks a function or variable as needing to be resolvable
// by name. This generally is needed if inline assembly refers to the variable
// by string name.
#ifdef __roar__
#define TORCH_NAME_RESOLVABLE __attribute__((roar_resolvable_by_name))
#else
#define TORCH_NAME_RESOLVABLE
#endif

// Semaphore define, declare and probe note format

#define TORCH_SDT_SEMAPHORE(provider, name)                                    \
  torch_sdt_semaphore_##provider##_##name

#define TORCH_SDT_DEFINE_SEMAPHORE(name)                                       \
  extern "C" {                                                                 \
    TORCH_NAME_RESOLVABLE                                                      \
    volatile unsigned short TORCH_SDT_SEMAPHORE(pytorch, name)                 \
    __attribute__((section(TORCH_SDT_SEMAPHORE_SECTION), used)) = 0;           \
  }

#define TORCH_SDT_DECLARE_SEMAPHORE(name)                                      \
  extern "C" TORCH_NAME_RESOLVABLE volatile unsigned short                     \
    TORCH_SDT_SEMAPHORE(pytorch, name)

#define TORCH_SDT_SEMAPHORE_NOTE_0(provider, name)                             \
  TORCH_SDT_ASM_1(     TORCH_SDT_ASM_ADDR 0) /*No Semaphore*/                  \

#define TORCH_SDT_SEMAPHORE_NOTE_1(provider, name)                             \
  TORCH_SDT_ASM_1(TORCH_SDT_ASM_ADDR TORCH_SDT_SEMAPHORE(provider, name))

// Structure of note section for the probe.
#define TORCH_SDT_NOTE_CONTENT(provider, name, has_semaphore, arg_template)    \
  TORCH_SDT_ASM_1(990: TORCH_SDT_NOP)                                          \
  TORCH_SDT_ASM_3(     .pushsection .note.stapsdt,"","note")                   \
  TORCH_SDT_ASM_1(     .balign 4)                                              \
  TORCH_SDT_ASM_3(     .4byte 992f-991f, 994f-993f, TORCH_SDT_NOTE_TYPE)       \
  TORCH_SDT_ASM_1(991: .asciz TORCH_SDT_NOTE_NAME)                             \
  TORCH_SDT_ASM_1(992: .balign 4)                                              \
  TORCH_SDT_ASM_1(993: TORCH_SDT_ASM_ADDR 990b)                                \
  TORCH_SDT_ASM_1(     TORCH_SDT_ASM_ADDR 0) /*Reserved for Base Address*/     \
  TORCH_SDT_SEMAPHORE_NOTE_##has_semaphore(provider, name)                     \
  TORCH_SDT_ASM_STRING(provider)                                               \
  TORCH_SDT_ASM_STRING(name)                                                   \
  TORCH_SDT_ASM_STRING(arg_template)                                           \
  TORCH_SDT_ASM_1(994: .balign 4)                                              \
  TORCH_SDT_ASM_1(     .popsection)

// Main probe Macro.
#define TORCH_SDT_PROBE(provider, name, has_semaphore, n, arglist)             \
    __asm__ __volatile__ (                                                     \
      TORCH_SDT_NOTE_CONTENT(                                                  \
        provider, name, has_semaphore, TORCH_SDT_ARG_TEMPLATE_##n)             \
      :: TORCH_SDT_OPERANDS_##n arglist                                        \
    )                                                                          \

// Helper Macros to handle variadic arguments.
#define TORCH_SDT_NARG_(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, N, ...) N
#define TORCH_SDT_NARG(...)                                                    \
  TORCH_SDT_NARG_(__VA_ARGS__, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define TORCH_SDT_PROBE_N(provider, name, has_semaphore, N, ...)               \
  TORCH_SDT_PROBE(provider, name, has_semaphore, N, (__VA_ARGS__))
