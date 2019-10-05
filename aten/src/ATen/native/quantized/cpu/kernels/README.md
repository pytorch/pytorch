 The files in this directory are compiled multiple times for different CPU vector instruction
 sets (e.g. AVX, AVX2). The purpose of putting code in this directory is to make
 sure we can generate the optimal code for a given processor's vector
 capabilities. Much of this is done via preprocessor guards in vec256_qint.h.

 The considerations for code written in this directory include:
  - Keep code in this directory to a minimum, since we're compiling it several
    times.
  - All code in this file should go through the DECLARE_DISPATCH,
    DEFINE_DISPATCH, and REGISTER_DISPATCH mechanism to ensure the correct
    runtime dispatch occurs.
  - THE CODE MUST RESIDE IN THE ANONYMOUS NAMESPACE. FAILURE TO ENSURE THIS
    IS THE CASE CAN LEAD TO HARD-TO-DEBUG ODR VIOLATIONS.
  - **Make sure different variants of the code (AVX, AVX2) are tested!**
    There are build variants that do things like have NO AVX and NO AVX2 in
    CI. Make sure they work!