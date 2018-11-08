#pragma once

// Default constraint for the probe arguments as operands.
#ifndef CAFFE_SDT_ARG_CONSTRAINT
#define CAFFE_SDT_ARG_CONSTRAINT      "nor"
#endif

// Instruction to emit for the probe.
#define CAFFE_SDT_NOP                 nop

// Note section properties.
#define CAFFE_SDT_NOTE_NAME           "stapsdt"
#define CAFFE_SDT_NOTE_TYPE           3

// Size of address depending on platform.
#ifdef __LP64__
#define CAFFE_SDT_ASM_ADDR            .8byte
#else
#define CAFFE_SDT_ASM_ADDR            .4byte
#endif

// Assembler helper Macros.
#define CAFFE_SDT_S(x)                #x
#define CAFFE_SDT_ASM_1(x)            CAFFE_SDT_S(x) "\n"
#define CAFFE_SDT_ASM_2(a, b)         CAFFE_SDT_S(a) "," CAFFE_SDT_S(b) "\n"
#define CAFFE_SDT_ASM_3(a, b, c)      CAFFE_SDT_S(a) "," CAFFE_SDT_S(b) ","    \
                                      CAFFE_SDT_S(c) "\n"
#define CAFFE_SDT_ASM_STRING(x)       CAFFE_SDT_ASM_1(.asciz CAFFE_SDT_S(x))

// Helper to determine the size of an argument.
#define CAFFE_SDT_ISARRAY(x)  (__builtin_classify_type(x) == 14)
#define CAFFE_SDT_ARGSIZE(x)  (CAFFE_SDT_ISARRAY(x) ? sizeof(void*) : sizeof(x))

// Format of each probe arguments as operand.
// Size of the arugment tagged with CAFFE_SDT_Sn, with "n" constraint.
// Value of the argument tagged with CAFFE_SDT_An, with configured constraint.
#define CAFFE_SDT_ARG(n, x)                                                    \
  [CAFFE_SDT_S##n] "n"                ((size_t)CAFFE_SDT_ARGSIZE(x)),          \
  [CAFFE_SDT_A##n] CAFFE_SDT_ARG_CONSTRAINT (x)

// Templates to append arguments as operands.
#define CAFFE_SDT_OPERANDS_0()        [__sdt_dummy] "g" (0)
#define CAFFE_SDT_OPERANDS_1(_1)      CAFFE_SDT_ARG(1, _1)
#define CAFFE_SDT_OPERANDS_2(_1, _2)                                           \
  CAFFE_SDT_OPERANDS_1(_1), CAFFE_SDT_ARG(2, _2)
#define CAFFE_SDT_OPERANDS_3(_1, _2, _3)                                       \
  CAFFE_SDT_OPERANDS_2(_1, _2), CAFFE_SDT_ARG(3, _3)
#define CAFFE_SDT_OPERANDS_4(_1, _2, _3, _4)                                   \
  CAFFE_SDT_OPERANDS_3(_1, _2, _3), CAFFE_SDT_ARG(4, _4)
#define CAFFE_SDT_OPERANDS_5(_1, _2, _3, _4, _5)                               \
  CAFFE_SDT_OPERANDS_4(_1, _2, _3, _4), CAFFE_SDT_ARG(5, _5)
#define CAFFE_SDT_OPERANDS_6(_1, _2, _3, _4, _5, _6)                           \
  CAFFE_SDT_OPERANDS_5(_1, _2, _3, _4, _5), CAFFE_SDT_ARG(6, _6)
#define CAFFE_SDT_OPERANDS_7(_1, _2, _3, _4, _5, _6, _7)                       \
  CAFFE_SDT_OPERANDS_6(_1, _2, _3, _4, _5, _6), CAFFE_SDT_ARG(7, _7)
#define CAFFE_SDT_OPERANDS_8(_1, _2, _3, _4, _5, _6, _7, _8)                   \
  CAFFE_SDT_OPERANDS_7(_1, _2, _3, _4, _5, _6, _7), CAFFE_SDT_ARG(8, _8)

// Templates to reference the arguments from operands in note section.
#define CAFFE_SDT_ARGFMT(no)        %n[CAFFE_SDT_S##no]@%[CAFFE_SDT_A##no]
#define CAFFE_SDT_ARG_TEMPLATE_0    /*No arguments*/
#define CAFFE_SDT_ARG_TEMPLATE_1    CAFFE_SDT_ARGFMT(1)
#define CAFFE_SDT_ARG_TEMPLATE_2    CAFFE_SDT_ARG_TEMPLATE_1 CAFFE_SDT_ARGFMT(2)
#define CAFFE_SDT_ARG_TEMPLATE_3    CAFFE_SDT_ARG_TEMPLATE_2 CAFFE_SDT_ARGFMT(3)
#define CAFFE_SDT_ARG_TEMPLATE_4    CAFFE_SDT_ARG_TEMPLATE_3 CAFFE_SDT_ARGFMT(4)
#define CAFFE_SDT_ARG_TEMPLATE_5    CAFFE_SDT_ARG_TEMPLATE_4 CAFFE_SDT_ARGFMT(5)
#define CAFFE_SDT_ARG_TEMPLATE_6    CAFFE_SDT_ARG_TEMPLATE_5 CAFFE_SDT_ARGFMT(6)
#define CAFFE_SDT_ARG_TEMPLATE_7    CAFFE_SDT_ARG_TEMPLATE_6 CAFFE_SDT_ARGFMT(7)
#define CAFFE_SDT_ARG_TEMPLATE_8    CAFFE_SDT_ARG_TEMPLATE_7 CAFFE_SDT_ARGFMT(8)

// Structure of note section for the probe.
#define CAFFE_SDT_NOTE_CONTENT(provider, name, arg_template)                   \
  CAFFE_SDT_ASM_1(990: CAFFE_SDT_NOP)                                          \
  CAFFE_SDT_ASM_3(     .pushsection .note.stapsdt,"","note")                   \
  CAFFE_SDT_ASM_1(     .balign 4)                                              \
  CAFFE_SDT_ASM_3(     .4byte 992f-991f, 994f-993f, CAFFE_SDT_NOTE_TYPE)       \
  CAFFE_SDT_ASM_1(991: .asciz CAFFE_SDT_NOTE_NAME)                             \
  CAFFE_SDT_ASM_1(992: .balign 4)                                              \
  CAFFE_SDT_ASM_1(993: CAFFE_SDT_ASM_ADDR 990b)                                \
  CAFFE_SDT_ASM_1(     CAFFE_SDT_ASM_ADDR 0) /*Reserved for Semaphore address*/\
  CAFFE_SDT_ASM_1(     CAFFE_SDT_ASM_ADDR 0) /*Reserved for Semaphore name*/   \
  CAFFE_SDT_ASM_STRING(provider)                                               \
  CAFFE_SDT_ASM_STRING(name)                                                   \
  CAFFE_SDT_ASM_STRING(arg_template)                                           \
  CAFFE_SDT_ASM_1(994: .balign 4)                                              \
  CAFFE_SDT_ASM_1(     .popsection)

// Main probe Macro.
#define CAFFE_SDT_PROBE(provider, name, n, arglist)                            \
    __asm__ __volatile__ (                                                     \
      CAFFE_SDT_NOTE_CONTENT(provider, name, CAFFE_SDT_ARG_TEMPLATE_##n)       \
      :: CAFFE_SDT_OPERANDS_##n arglist                                        \
    )                                                                          \

// Helper Macros to handle variadic arguments.
#define CAFFE_SDT_NARG_(_0, _1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
#define CAFFE_SDT_NARG(...)                                                    \
  CAFFE_SDT_NARG_(__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define CAFFE_SDT_PROBE_N(provider, name, N, ...)                              \
  CAFFE_SDT_PROBE(provider, name, N, (__VA_ARGS__))
