rm *.ll *.s

BC_FILE=$1

/opt/rocm/llvm/bin/llvm-dis $BC_FILE -o original.ll
/opt/rocm/llvm/bin/opt -S -inline -inline-threshold=104857 original.ll > inline.ll
/opt/rocm/llvm/bin/opt -S -sroa inline.ll > sroa.ll
/opt/rocm/llvm/bin/opt -S -O3 sroa.ll > o3.ll

/opt/rocm/llvm/bin/llc -mcpu=gfx906 original.ll
/opt/rocm/llvm/bin/llc -mcpu=gfx906 inline.ll
/opt/rocm/llvm/bin/llc -mcpu=gfx906 sroa.ll
/opt/rocm/llvm/bin/llc -mcpu=gfx906 o3.ll

#/opt/rocm/llvm/bin/opt -S -O3 -sroa inline.ll > o3.ll
#/opt/rocm/llvm/bin/opt -S -O3 -sroa o3.ll > o3_2.ll
#/opt/rocm/llvm/bin/opt -S -O3 -sroa o3_2.ll > o3_3.ll
#/opt/rocm/llvm/bin/opt -S -O3 -sroa o3_3.ll > o3_4.ll

#/opt/rocm/llvm/bin/llc -mcpu=gfx908 opt.ll
#/opt/rocm/llvm/bin/llc -mcpu=gfx908 inline.ll
#/opt/rocm/llvm/bin/llc -mcpu=gfx908 o3.ll
#/opt/rocm/llvm/bin/llc -mcpu=gfx908 o3_2.ll
#/opt/rocm/llvm/bin/llc -mcpu=gfx908 o3_3.ll
#/opt/rocm/llvm/bin/llc -mcpu=gfx908 o3_4.ll
