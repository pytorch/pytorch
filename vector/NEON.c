static void THFloatVector_fill_NEON(float *x, const float c, const long n) {
  float ctemp = c;
  float * caddr = &ctemp;
  __asm__ __volatile__ (
      "mov         r0, %0           @ \n\t"
      "ldr         r4, [%1]         @ \n\t"
      "vdup.32     q12, r4          @ \n\t"
      "vdup.32     q13, r4          @ \n\t"
      "lsrs        r4, %2, #3       @ \n\t"
      "beq         3f               @ \n\t"
      "1:                           @ \n\t"
      "vst1.32     {d24-d27}, [r0]! @ \n\t"
      "subs        r4, r4, #1       @ \n\t"
      "bne         1b               @ \n\t"
      "3:                           @ \n\t"
      "ands        r4, %2, #7       @ \n\t"
      "beq         5f               @ \n\t"
      "4:                           @ \n\t"
      "subs        r4, r4, #1       @ \n\t"
      "vst1.32     {d24[0]}, [r0]!  @ \n\t"
      "bne         4b               @ \n\t"
      "5:                           @ "
      :
      :"r" (x), "r"(caddr),"r"(n)
      : "cc", "r0", "r4",  "memory",
        "q12",
        "d24", "d25", "d26", "d27"
      );
}


static void THFloatVector_diff_NEON(float *y, const float *x, const float c, const long n) {
  __asm__ __volatile__ (
      "mov         r0, %2           @ \n\t"
      "mov         r1, %1           @ \n\t"
      "mov         r2, %0           @ \n\t"
      "lsrs        r4, %3, #3       @ \n\t"
      "beq         3f               @ \n\t"
      "vld1.32     {d16-d19}, [r1]! @ \n\t"
      "vld1.32     {d0-d3}, [r0]!   @ \n\t"
      "1:                           @ \n\t"
      "vsub.f32    q12, q8, q0      @ \n\t"
      "vsub.f32    q13, q9, q1      @ \n\t"
      "subs        r4, r4, #1       @ \n\t"
      "beq         2f               @ \n\t"
      "vld1.32     {d16-d19}, [r1]! @ \n\t"
      "vld1.32     {d0-d3}, [r0]!   @ \n\t"
      "vst1.32     {d24-d27}, [r2]! @ \n\t"
      "b           1b               @ \n\t"
      "2:                           @ \n\t"
      "vst1.32     {d24-d27}, [r2]! @ \n\t"
      "3:                           @ \n\t"
      "ands        r4, %3, #7       @ \n\t"
      "beq         5f               @ \n\t"
      "4:                           @ \n\t"
      "subs        r4, r4, #1       @ \n\t"
      "vld1.32     {d16[0]}, [r1]!  @ \n\t"
      "vld1.32     {d0[0]}, [r0]!   @ \n\t"
      "vsub.f32    d24, d16, d0     @ \n\t"
      "vst1.32     {d24[0]}, [r2]!  @ \n\t"
      "bne         4b               @ \n\t"
      "5:                           @ "
      :
      :"r" (z), "r" (x),"r" (y), "r"(n)
      : "cc", "r0", "r1", "r2", "r4", "memory",
        "q0", "q1", "q8", "q9", "q12", "q13",
        "d0", "d1", "d2", "d3",
        "d16", "d17", "d18", "d19", "d24", "d25", "d26", "d27"
      );
}


static void THFloatVector_scale_NEON(float *y, const float c, const long n) {
  float ctemp = c;
  float * caddr = &ctemp;
  __asm__ __volatile__ (
      "mov         r0, %0           @ \n\t"
      "mov         r2, r0           @ \n\t"
      "ldr         r5, [%1]         @ \n\t"
      "vdup.32     q14, r5          @ \n\t"
      "lsrs        r5, %2, #5       @ \n\t"
      "beq         3f               @ \n\t"
      "vld1.32     {d0-d3}, [r0]!   @ \n\t"
      "vld1.32     {d4-d7}, [r0]!   @ \n\t"
      "vld1.32     {d8-d11}, [r0]!  @ \n\t"
      "vld1.32     {d12-d15}, [r0]! @ \n\t"
      "1:                           @ \n\t"
      "vmul.f32    q0, q0, q14      @ \n\t"
      "vmul.f32    q1, q1, q14      @ \n\t"
      "vmul.f32    q2, q2, q14      @ \n\t"
      "vmul.f32    q3, q3, q14      @ \n\t"
      "vmul.f32    q4, q4, q14      @ \n\t"
      "vmul.f32    q5, q5, q14      @ \n\t"
      "vmul.f32    q6, q6, q14      @ \n\t"
      "vmul.f32    q7, q7, q14      @ \n\t"
      "subs        r5, r5, #1       @ \n\t"
      "beq         2f               @ \n\t"
      "vst1.32     {d0-d3}, [r2]!   @ \n\t"
      "vld1.32     {d0-d3}, [r0]!   @ \n\t"
      "vst1.32     {d4-d7}, [r2]!   @ \n\t"
      "vld1.32     {d4-d7}, [r0]!   @ \n\t"
      "vst1.32     {d8-d11}, [r2]!  @ \n\t"
      "vld1.32     {d8-d11}, [r0]!  @ \n\t"
      "vst1.32     {d12-d15}, [r2]! @ \n\t"
      "vld1.32     {d12-d15}, [r0]! @ \n\t"
      "b           1b               @ \n\t"
      "2:                           @ \n\t"
      "vst1.32     {d0-d3}, [r2]!   @ \n\t"
      "vst1.32     {d4-d7}, [r2]!   @ \n\t"
      "vst1.32     {d8-d11}, [r2]!  @ \n\t"
      "vst1.32     {d12-d15}, [r2]! @ \n\t"
      "3:                           @ \n\t"
      "lsrs        r5, %2, #4       @ \n\t"
      "ands        r5, r5, #1       @ \n\t"
      "beq         4f               @ \n\t"
      "vld1.32     {d0-d3}, [r0]!   @ \n\t"
      "vld1.32     {d4-d7}, [r0]!   @ \n\t"
      "vmul.f32    q0, q0, q14      @ \n\t"
      "vmul.f32    q1, q1, q14      @ \n\t"
      "vmul.f32    q2, q2, q14      @ \n\t"
      "vmul.f32    q3, q3, q14      @ \n\t"
      "vst1.32     {d0-d3}, [r2]!   @ \n\t"
      "vst1.32     {d4-d7}, [r2]!   @ \n\t"
      "4:                           @ \n\t"
      "lsrs        r5, %2, #3       @ \n\t"
      "ands        r5, r5, #1       @ \n\t"
      "beq         5f               @ \n\t"
      "vld1.32     {d0-d3}, [r0]!   @ \n\t"
      "vmul.f32    q0, q0, q14      @ \n\t"
      "vmul.f32    q1, q1, q14      @ \n\t"
      "vst1.32     {d0-d3}, [r2]!   @ \n\t"
      "5:                           @ \n\t"
      "ands        r5, %2, #7       @ \n\t"
      "beq         7f               @ \n\t"
      "6:                           @ \n\t"
      "subs        r5, r5, #1       @ \n\t"
      "vld1.32     d0[0], [r0]!     @ \n\t"
      "vmul.f32    d0, d0, d28      @ \n\t"
      "vst1.32     d0[0], [r2]!     @ \n\t"
      "bne         6b               @ \n\t"
      "7:                           @ "
      :
      :"r" (y), "r"(caddr),"r"(n)
      : "cc", "r0", "r2", "r5", "memory",
        "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q14",
        "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15",
        "d28", "d29"
      );

}

static void THFloatVector_mul_NEON(float *y, const float *x, const long n) {
  __asm__ __volatile__ (
      "mov         r0, %0           @ \n\t"
      "mov         r1, %1           @ \n\t"
      "mov         r2, r0           @ \n\t"
      "lsrs        r4, %2, #3       @ \n\t"
      "beq         3f               @ \n\t"
      "vld1.32     {d16-d19}, [r1]! @ \n\t"
      "vld1.32     {d0-d3}, [r0]!   @ \n\t"
      "1:                           @ \n\t"
      "vmul.f32    q12, q8, q0      @ \n\t"
      "vmul.f32    q13, q9, q1      @ \n\t"
      "subs        r4, r4, #1       @ \n\t"
      "beq         2f               @ \n\t"
      "vld1.32     {d16-d19}, [r1]! @ \n\t"
      "vld1.32     {d0-d3}, [r0]!   @ \n\t"
      "vst1.32     {d24-d27}, [r2]! @ \n\t"
      "b           1b               @ \n\t"
      "2:                           @ \n\t"
      "vst1.32     {d24-d27}, [r2]! @ \n\t"
      "3:                           @ \n\t"
      "ands        r4, %2, #7       @ \n\t"
      "beq         5f               @ \n\t"
      "4:                           @ \n\t"
      "subs        r4, r4, #1       @ \n\t"
      "vld1.32     {d16[0]}, [r1]!  @ \n\t"
      "vld1.32     {d0[0]}, [r0]!   @ \n\t"
      "vmul.f32    q12, q8, q0      @ \n\t"
      "vst1.32     {d24[0]}, [r2]!  @ \n\t"
      "bne         4b               @ \n\t"
      "5:                           @ "
      :
      :"r" (y),"r" (x),"r"(n)
      : "cc", "r0", "r1", "r2", "r4", "memory",
        "q0", "q1", "q8", "q9", "q12", "q13",
        "d0", "d1", "d2", "d3",
        "d16", "d17", "d18", "d19", "d24", "d25", "d26", "d27"
      );
}

static void THFloatVector_add_NEON(float *y, const float *x, const float c, const long n) {
  float ctemp = c;
  float * caddr = &ctemp;
  __asm__ __volatile__ (
      "mov         r0, %0           @ \n\t"
      "mov         r1, %1           @ \n\t"
      "mov         r2, r0           @ \n\t"
      "ldr         r5, [%2]         @ \n\t"
      "vdup.32     q14, r5          @ \n\t"
      "lsrs        r5, %3, #4       @ \n\t"
      "beq         3f               @ \n\t"
      "vld1.32     {d16-d19}, [r1]! @ \n\t"
      "vld1.32     {d0-d3}, [r0]!   @ \n\t"
      "vld1.32     {d20-d23}, [r1]! @ \n\t"
      "vld1.32     {d4-d7}, [r0]!   @ \n\t"
      "1:                           @ \n\t"
      "vmla.f32    q0, q8, q14      @ \n\t"
      "vmla.f32    q1, q9, q14      @ \n\t"
      "vmla.f32    q2, q10, q14     @ \n\t"
      "vmla.f32    q3, q11, q14     @ \n\t"
      "subs        r5, r5, #1       @ \n\t"
      "beq         2f               @ \n\t"
      "vld1.32     {d16-d19}, [r1]! @ \n\t"
      "vld1.32     {d20-d23}, [r1]! @ \n\t"
      "vst1.32     {d0-d3}, [r2]!   @ \n\t"
      "vld1.32     {d0-d3}, [r0]!   @ \n\t"
      "vst1.32     {d4-d7}, [r2]!   @ \n\t"
      "vld1.32     {d4-d7}, [r0]!   @ \n\t"
      "b           1b               @ \n\t"
      "2:                           @ \n\t"
      "vst1.32     {d0-d3}, [r2]!   @ \n\t"
      "vst1.32     {d4-d7}, [r2]!   @ \n\t"
      "3:                           @ \n\t"
      "lsrs        r5, %3, #3       @ \n\t"
      "ands        r5, #1           @ \n\t"
      "beq         4f               @ \n\t"
      "vld1.32     {d16-d19}, [r1]! @ \n\t"
      "vld1.32     {d0-d3}, [r0]!   @ \n\t"
      "vmla.f32    q0, q8, q14      @ \n\t"
      "vmla.f32    q1, q9, q14      @ \n\t"
      "vst1.32     {d0-d3}, [r2]!   @ \n\t"
      "4:                           @ \n\t"
      "ands        r5, %3, #7       @ \n\t"
      "beq         6f               @ \n\t"
      "5:                           @ \n\t"
      "subs        r5, r5, #1       @ \n\t"
      "vld1.32     {d16[0]}, [r1]!  @ \n\t"
      "vld1.32     {d0[0]}, [r0]!   @ \n\t"
      "vmla.f32    d0, d16, d28     @ \n\t"
      "vst1.32     d0[0], [r2]!     @ \n\t"
      "bne         5b               @ \n\t"
      "6:                           @ "
      :
      :"r" (y),"r" (x), "r"(caddr),"r"(n)
      : "cc", "r0", "r1", "r2", "r5", "memory",
        "q0", "q1", "q2", "q3", "q14",
        "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d28", "d29"
      );
}
