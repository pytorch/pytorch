# `GEMMDesc`: how to compute a GEMM, written down precisely enough to pin the bits

This is the long form of `gemm_desc.py`. The module keeps the short rules a reader
needs while editing a line; everything here is the worked examples, the
measurements they came from, and the reasoning behind what the type does and does
not carry. Nothing here is needed to use the type. This file, in the repo next to
the module, is the canonical long form; anything else is a copy of it.

## Contents

- [What a descriptor promises](#what-a-descriptor-promises)
- [The five algorithms](#the-five-algorithms)
- [`input_precision`: how fp32 is cut down](#input_precision-how-fp32-is-cut-down)
- [Why a cut changes the answer](#why-a-cut-changes-the-answer)
- [How cuts nest](#how-cuts-nest)
- [Which k elements one part owns](#which-k-elements-one-part-owns)
- [A part length, not a part count](#a-part-length-not-a-part-count)
- [Two dtypes for a merge, not one](#two-dtypes-for-a-merge-not-one)
- [Merge order: the count-down butterfly](#merge-order-the-count-down-butterfly)
- [`period` and `divisor`: labelling a strided tile](#period-and-divisor-labelling-a-strided-tile)
- [`instruction_k` is not the loop's k step](#instruction_k-is-not-the-loops-k-step)
- [`use_fast_accum`](#use_fast_accum)
- [`k_loop_step`: the short turn first](#k_loop_step-the-short-turn-first)
- [The epilogue pipelines](#the-epilogue-pipelines)
- [The three dtypes](#the-three-dtypes)
- [Operand dtypes have no fall-through](#operand-dtypes-have-no-fall-through)
- [What a real GEMM looks like here](#what-a-real-gemm-looks-like-here)
- [What is deliberately not a field](#what-is-deliberately-not-a-field)
- [What this cannot say](#what-this-cannot-say)
- [The two relations](#the-two-relations)

## What a descriptor promises

A GEMM's floating-point result depends on the order its k products are added in,
because floating-point addition is not associative. Two kernels that compute the
same matrix product can therefore hand back different bits. `GEMMDesc` writes that
order down: two runs on the same architecture that follow the same `GEMMDesc` owe
you the same bits.

Across architectures that is an open question, not a promise. What has been
measured is narrower -- within one architecture the form of the matrix instruction
often does not change the result, MMAv5 agreeing with MMAv2 on Blackwell and
`mma.sync` with `wgmma` on Hopper -- and even that has fp8 as a known exception.
Whether one descriptor holds across, say, an H100 and a GB300 has not been tested,
so the module does not claim it.

That exception is why `algorithm` names the matrix instruction that ran --
`mma.sync`, `wgmma` or `tcgen05.mma` -- instead of saying only that a tensor core
did it. One name covering all three would be one name for two answers on fp8, which
is the thing this type exists to stop.

There are two ways to get a descriptor. A person writes it by hand, so a result
stays reproducible anywhere. Or a later layer produces one by reading what an
existing library would do for a shape, so a Triton kernel can be made to agree with
that library bit for bit. The module is only the description; the producers come
later, and so do the `Equivalences` they measure.

A `GEMMDesc` holds exactly two kinds of thing: which GEMM algorithm, and every
parameter that changes the floating-point order. It holds nothing that is only a
speed knob. Tile sizes, warp counts, pipeline depth, cluster shape and how much
scratch memory a library may use are all absent for that reason: they move the run
time, not the result.

They can move the result indirectly, and a producer has to watch for it. A tile too
small for an instruction makes a compiler fall back to a different one -- Triton
leaves its native fp8 matrix-instruction path when the m tile drops below 64 -- and
then the bits move, but they move through `algorithm`, which is a field. So the rule
holds as stated and the obligation it creates is on the producer: emit a launch that
really runs the instruction the descriptor names, and check it rather than assume it.

Nothing in the module names a library, an algorithm id, or a vendor enum value. A
library is a *producer* of a `GEMMDesc`, never a part of one. Where an algorithm
there happens to equal a known library kernel, a comment says which one. A hardware
instruction is not a vendor enum value in that sense: `mma.sync` is the name the
instruction set gives the instruction, and which instruction ran is one of the
things that decides the bits.

## The five algorithms

`GEMMAlgorithm` says how operands become an update to the accumulator. The five
round in different places, so for the same inputs they are different answers:

```
a matrix instruction       [sum of instruction_k products + acc] -> acc
                           one rounding per instruction
SCALAR_FUSED_MULTIPLY_ADD  fma(a, b, acc) -> acc
                           one rounding per k element
SCALAR_MULTIPLY_THEN_ADD   t = a * b, then acc + t -> acc
                           two roundings per k element
```

`MMA_SYNC`, `WGMMA` and `TCGEN05_MMA` all run on the tensor cores and differ in who
issues the instruction and where the accumulator lives:

| value | PTX | who issues it | accumulator | hardware | Triton path |
| --- | --- | --- | --- | --- | --- |
| `MMA_SYNC` | `mma.sync.aligned` | one warp, its 32 lanes cooperating | registers, a piece in each lane | Volta (sm_70) onward | MMAv2 |
| `WGMMA` | `wgmma.mma_async.sync.aligned` | one warpgroup of four warps, asynchronously | registers | Hopper (sm_90) only | MMAv3 |
| `TCGEN05_MMA` | `tcgen05.mma` | a single thread | tensor memory | Blackwell (sm_100) onward | MMAv5 |

`wgmma` can also read its operands straight from shared memory.

The two scalar values are the CUDA-core loop, one k element per step.
`SCALAR_FUSED_MULTIPLY_ADD` is what cuBLAS's CUDA-core kernels do, and what Triton
emits with `enable_fp_fusion=True` when no matrix instruction is involved.
`SCALAR_MULTIPLY_THEN_ADD` is what Triton reaches with `enable_fp_fusion=False`.

`mma.sync` is the oldest of the three matrix instructions. On most dtypes they do
agree -- but only on most, and that exception is what stops them being one value.
Naming them separately costs nothing where they do agree, because
`produces_same_bits` reads a machine's measured agreements from `Equivalences`.

They are the NVIDIA families. AMD's `v_mfma` and `v_wmma` are a separate family with
no value in the enum yet, and so is NVIDIA's older `wmma.mma.sync`. A producer for
one of those must add its own value rather than borrow a name; with no value to
borrow, it declines instead, which is the safe way round.

Adding an instruction means adding a value to `GEMMAlgorithm` and a member to
`MATRIX_INSTRUCTIONS`. Membership of that set, rather than a check spelled out at
each use, is what makes the field rules in the `GEMMDesc` docstring cover the new
value.

## `input_precision`: how fp32 is cut down

fp32 operands can be fed to a matrix instruction in more than one way, and the
choice changes the result while the dtype stays fp32. It cannot be read off
`operand_dtype`, so it is its own field. The names are the ones Triton uses for
`tl.dot(input_precision=...)`.

`IEEE` multiplies the operands at their own precision, and is the only choice for
anything that is not fp32. `TF32` rounds each fp32 operand to tf32 -- an 8-bit
exponent and a 10-bit mantissa -- before the multiply, which is what cuBLAS calls
`CUBLAS_COMPUTE_32F_FAST_TF32`. `TF32X3` splits each fp32 operand into three tf32
pieces and sums the pieces' products, which recovers most of the fp32 mantissa; it
is also known as 3xTF32.

Both tf32 values need a matrix instruction and fp32 operands, because tf32 exists
only as a matrix instruction's input format.

## Why a cut changes the answer

A `KCut` is worth describing because an accumulator that restarts at zero puts a
bracket in the sum. Take K = 12 with one matrix instruction per 2 k elements, so
six instruction results d0..d5. With no cut, one accumulator walks all six:

```
((((( d0+d1 )+d2 )+d3 )+d4 )+d5 )
```

With one cut at span 4, each group of two gets its own accumulator and the three
totals are added:

```
( ( (d0+d1) + (d2+d3) ) + (d4+d5) )
```

Same six numbers, every level still added left to right, different brackets.
Floating-point addition is not associative, so those are different answers. This is
not a corner case: the Triton study this type replaces first missed the second level
because a flat sum matched the reference exactly while k fit in one group and parted
from it at the first k that needed two.

## How cuts nest

Cuts nest, outermost first. Level 0 cuts the whole k axis; level 1 cuts each part
level 0 produced; and so on. With `k_cuts = (KCut(span=6, ...), KCut(span=2, ...))`
over K = 12:

```
K = 12
|-- part k[0:6]              level 0, its own accumulator
|   |-- part k[0:2]          level 1, its own accumulator
|   |-- part k[2:4]
|   +-- part k[4:6]
|   total = ((k[0:2] + k[2:4]) + k[4:6])
+-- part k[6:12]
    |-- part k[6:8]
    |-- part k[8:10]
    +-- part k[10:12]
    total = ((k[6:8] + k[8:10]) + k[10:12])
result = k[0:6] total + k[6:12] total
```

The innermost part has no cut under it. One accumulator walks it in index order, and
`GEMMDesc.algorithm` and `GEMMDesc.instruction_k` say what one step of that walk is.

## Which k elements one part owns

`KCutLayout` answers that. With K = 12:

```
CONTIGUOUS, span=4
  part0 = k[0:4]   part1 = k[4:8]   part2 = k[8:12]

STRIDED, span=2, count=3
  tile:   t0     t1     t2     t3     t4     t5
          k0 k1  k2 k3  k4 k5  k6 k7  k8 k9  k10 k11
  part0 = t0 t3    part1 = t1 t4    part2 = t2 t5
```

Both hand three parts four k elements each, and the two answers differ, because
which k elements share an accumulator is what decides the sum.

cuBLAS's two-kernel gemv hands tiles out the `STRIDED` way.

## A part length, not a part count

`KCut.span` is the length itself and not a part count, and there is deliberately no
`num_splits` field next to it.

For a `CONTIGUOUS` cut the number of parts follows from the length being cut, as
`ceil(length / span)`, so storing it too could only ever disagree with it.

The length is also the more faithful of the two to write down, because the count a
library is asked for and the count it then performs are not the same number. Ask for
5 splits of K = 1000 on a 64-element grain: 1000 / 5 is 200, rounded up to the grain
that is a 256-long slice, and K = 1000 in 256-long slices is 4 slices, not 5. A
descriptor written from the request would describe a sum that never happened. The
one written from `span` records the sum that did.

A `STRIDED` cut is the other way round and does carry `count`, because which tiles
land in which part cannot be recovered from the span alone.

## Two dtypes for a merge, not one

`KCut.partial_dtype` is the dtype a finished part is written at before it is merged.
`KCut.merge_dtype` is the dtype the running merge sum is kept in. They are two
fields because all the useful combinations are in use: fp32 partials summed in fp32,
output-dtype partials summed in fp32, and a chain kept in the output dtype the whole
way.

Assuming the second one follows from the first is a bug that hid for a long time in
the Triton study this type replaces -- the merge rounded to fp16 whatever the output
dtype was, so every bf16 split-k result was wrong.

## Merge order: the count-down butterfly

`MergeOrder.SEQUENTIAL` is part 0, then part 1, and so on: `(((p0 + p1) + p2) + p3)`.
That is what a loop over the parts does, so it is what a split-k merge kernel that
walks the partials and adds each into a running sum does.

`MergeOrder.PAIRWISE_TREE` is a balanced binary tree over the parts **in index
order**: `((p0 + p1) + (p2 + p3))`. A cross-lane reduction that folds with butterfly
shuffles is a balanced tree, but it is only this one when it pairs lanes in index
order -- a count-up butterfly. A count-down butterfly is a balanced tree in
bit-reversed order, which is a different sum.

That was measured, not assumed. Over 88 measured cases, writing cuBLAS's count-down
gemv merge as a single `PAIRWISE_TREE` matched 16, and those 16 are exactly the
two-part rows where the two orders coincide.

Write a fold whose pairing is not index order as nested two-part cuts instead, one
level per round, which needs no vocabulary beyond `KCut` and always matched. Reach
for `PAIRWISE_TREE` only when the tree really is over the parts as this cut numbers
them.

The two orders agree while a cut has two parts or fewer -- `(p0 + p1)` either way --
so the distinction only starts to bite at three, and a producer that has only ever
seen two-part cuts has not yet learned which one a kernel does.

## `period` and `divisor`: labelling a strided tile

These two fields on `KCut` say how a `STRIDED` cut labels a tile, for the case where
the label does not simply repeat every `count` tiles:

```
label = (k // span) % period          None means period == count
part  = (label // divisor) % count    which digit of the label to read
```

Left unset this is the plain rotation the layout describes, tile j to part
`j % count`, so no descriptor written before these fields existed changes meaning.

They are needed because a nest of plain `STRIDED` cuts can only ever label a tile by
its index modulo the product of the counts, and a real kernel deals tiles out on a
period the tree below it does not factor. cuBLAS's two-kernel gemv is one: its
reduction tree has fixed level counts 4, 8, 2, 2 and so a product of 128, while the
tiles are dealt to blocks on the split count, which over a sweep of 2,384 gemv shapes
took values including 10, 37, 74, 148, 253, 296 and 592 -- 117 of 299 hits on a
period 128 cannot express. Stating the period once and letting each level read its
own digit of the label covers all of them.

A part may end up empty, and that is not an error: with period 37 a level with
count 2 at divisor 64 never sees a label that large, so its second part gets nothing
and contributes a zero to the merge. The real kernel does the same.

## `instruction_k` is not the loop's k step

`GEMMDesc.instruction_k` is how many k elements one matrix instruction sums into the
accumulator in a single rounding. It is the step size of the innermost chain, so it
sets where that chain rounds, which is why it is a field rather than left to the
compiler.

It is not the loop's k step, and seeing why is also why the loop's k step is not a
field at all. With a threadblock k step of 64 and `instruction_k` 16, one turn of the
loop issues four instructions into the same accumulator:

```
turn 0:  [16][16][16][16]  -> added into acc one after another
turn 1:  [16][16][16][16]  -> added into acc one after another
```

The chain steps by 16 either way. Widening the turn to 128 regroups the loop but not
the chain, so it cannot move the bits.

## `use_fast_accum`

The field says where a matrix instruction's result meets the running accumulator:

```
True   a, b -> [instruction sums and adds into acc] -round-> acc

False  a, b -> [instruction sums into zero]         -round-> part
       acc + part                                   -round-> acc
```

`True` is `tl.dot(a, b, acc)` and rounds once a step; `False` is `acc + tl.dot(a, b)`
and rounds twice. The two scalar algorithms draw the same distinction in their own
names, so this is their matrix-instruction twin: it is required for a matrix
instruction and `None` otherwise. The name is torch's -- the `use_fast_accum`
argument of `torch._scaled_mm`, and `USE_FAST_ACCUM` in Inductor's GEMM templates.

The name carries two things, and a producer needs both.

One is the structural difference above, and it moves the result on every
architecture. That is measured rather than assumed: over a 640x512 output at
K = 4096 with fp16 operands and an fp32 accumulator, 326,703 of 327,680 elements
differ between the two forms.

The other is why it is called *fast* accumulate and why torch hands it to the caller
at all. On Hopper an fp8 matrix instruction can keep accumulating inside the tensor
core at reduced precision, which is quicker and loses bits, so there the choice is
not only about where the add sits. Triton spells that out as `max_num_imprecise_acc`,
how many instructions may pass before the value is pulled back into a full-precision
accumulator.

On Blackwell the two forms normally cannot be told apart, because Triton folds one
into the other before codegen: `CombineDotAddFPattern` in
`lib/Dialect/Triton/Transforms/Combine.cpp` turns `addf(dot(a, b, 0), acc)` into
`dot(a, b, acc)`, so both reach the same machine code. The measurement above comes
from blocking that fold. Triton leaves it unfolded on sm_90 for fp8 operands --
`max_num_imprecise_acc_default` is non-zero only there -- which is exactly the case
torch exposes.

There is no default for the field, for the same reason the dtypes have none: it
decides bits, so it has to be stated.

## `k_loop_step`: the short turn first

`GEMMDesc.k_loop_step` is the k loop step of one accumulator: how many k elements one
turn of the mainloop covers. It is a field for one reason. When a part is not a whole
number of turns, a kernel that runs the short turn FIRST puts the instruction group
boundaries somewhere else than one that runs it last. With K = 10 and
`instruction_k` 4:

```
None, short group last    [k0 k1 k2 k3][k4 k5 k6 k7][k8 k9]
                                d0           d1        d2
                          boundaries at k = 0, 4, 8

4, short group first      [k0 k1][k2 k3 k4 k5][k6 k7 k8 k9]
                             d0        d1          d2
                          boundaries at k = 0, 2, 6
```

Same K, same `instruction_k`, three groups either way -- but different k elements are
folded into the same rounding, so the two are different sums. `None` means the
boundaries run on an even grid from 0 and the short group is last.

## The epilogue pipelines

`EpilogueOrder` says where the single rounding to `output_dtype` sits relative to the
epilogue. A bias added to the accumulator before that rounding and the same bias
added after it give different bits, so this has to be said out loud, not assumed:

```
NONE            [k sum] -round-> out
IN_ACCUMULATOR  [k sum] -> +bias -> +beta*C -round-> out
AFTER_ROUNDING  [k sum] -round-> +bias -round-> +beta*C
```

`AFTER_ROUNDING` is what an epilogue that is a separate pass over the stored result
does: the GEMM writes `output_dtype` to memory and something else reads it back and
adds the bias. Inductor's fused GEMM templates deliberately reproduce it --
`select_algorithm.py` rounds the accumulator to the output dtype before the fused
graph ops run, under a comment saying it is emulating unfused numerics -- so a fused
kernel is `AFTER_ROUNDING`, not `IN_ACCUMULATOR`, for the ops the graph fused in. A
library epilogue that really runs inside the GEMM kernel, such as cuBLAS adding a
bias to the live accumulator, is `IN_ACCUMULATOR` instead.

One thing the enum cannot say: Inductor puts the rounding in the *middle*, after the
template's own epilogue and before the fused graph ops, so an addmm with a fused relu
adds the bias in the accumulator dtype and applies the relu after the round. That is
neither value. A descriptor for such a kernel would need to place the rounding within
the epilogue rather than at one end of it, and a producer that meets one must decline.

## The three dtypes

`operand_dtype` is the dtype both operands are read at. It is one field and not two
because a GEMM with two different operand dtypes is out of scope; when that changes
the field splits in two, rather than growing a rule about which one wins.

`accumulate_dtype` is the dtype the k sum is kept in. It is a field of its own
because it is not implied by the operand dtype: fp16 operands are accumulated in
fp32 by almost everything but not by everything, and fp32 and fp64 operands make
"always fp32" wrong.

`output_dtype` is the dtype the finished value is rounded to on the store. All three
must be floating point, or all three integer.

## Operand dtypes have no fall-through

An operand may be read at any dtype in `OPERAND_DTYPES`. A dtype that is not listed
is rejected, never quietly treated as one that is. That mistake is how a study of
this problem in Triton ran an fp32 operand through the fp16 recipe and returned a
wrong answer instead of declining.

fp8 is missing from `ACCUMULATE_DTYPES` on purpose: no hardware accumulates in it,
and allowing it would let a producer write a descriptor nobody can honour.

## What a real GEMM looks like here

The kinds of GEMM this vocabulary was checked against. Each one names the shape and
the cuBLAS kernel it was read off, then says what `k_cuts` comes out as:

```
one accumulator over the whole k axis (nvjet)
    no cut at all, so k_cuts is the empty tuple

the same, but the short group of k runs first (CUTLASS)
    still no cut; k_loop_step is what says the short group runs first

split-k (nvjet split-k)
    one cut: k in contiguous slices, each slice with its own accumulator

split-k whose slices close an accumulator every block (nvjet split-k)
    two cuts: first the slices, then the blocks inside one slice

split-k over instruction groups (CUTLASS split-k)
    one cut into slices, with k_loop_step describing the groups inside a slice

a scalar chain of chunks (gemmSN_NN, magma_sgemmEx)
    two cuts: first the chunks, then the smaller chunks inside one of them

a gemv whose lanes interleave (gemv2T, gemv2N)
    two cuts: first the lanes, each taking every count-th tile of k, then the
    chunks that one lane walks in order

a gemv whose lanes fold in bit-reversed order (gemv2T, gemv2N)
    one cut per round of the fold, each cutting what is left in two

a gemv dealt out to blocks on a period (dot_kernel + reduce_1Block)
    strided cuts carrying `period`, because the tiles repeat on a number that
    the cut counts do not multiply to
```

The first five run a matrix instruction and the rest a scalar one. A part count is
never a field: for a contiguous cut it follows from `span` and the length being cut,
and for a strided cut it is `count`.

The kernel names are only a signpost. They are where each shape was read off, they
are not part of any descriptor, and a shape stands on its own wherever else it turns
up. Each was written out in full against the kernel it describes, and the four scalar
ones were then checked by byte-comparing a literal walk of the descriptor against
what the kernel returned.

## What is deliberately not a field

**Operand layout.** Row-major, column-major and arbitrary strides say which bytes are
which matrix element; they describe the *problem*, not the algorithm. Reading an
element never rounds, so two runs with the same `GEMMDesc` over differently laid out
operands add the same products in the same order and return the same bits. Layout is
an input to a producer -- it is one of the things a library looks at when it picks a
kernel -- but it is not part of the description that comes out. Leaving it out is the
point: Inductor sees arbitrary layouts, and a descriptor that pinned one layout per
dtype would refuse work it can do.

**The batch axis.** A batched GEMM is a set of independent GEMMs, and nothing is ever
summed across the batch, so the batch cannot change the order of any one output
element. One `GEMMDesc` describes every GEMM in the batch. Only a GEMM that reduced
over the batch would need a field, and that is a different operation.

**The values of alpha, beta and the bias.** Those are data, and data does not belong
in a description of an algorithm. What changes the bits is *where* the single
rounding to `output_dtype` sits relative to them, and that question is `epilogue`.

**How operand scales are applied.** A scale that is one constant per output element
could be described as an epilogue step, but a block-wise scale cannot: it multiplies
each block of the k sum by its own factor, inside the k loop, so it cuts the k axis
rather than following it. Covering only the first kind would be a field that quietly
means "the quantization strategies we happened to think of", so this version has no
scale field at all, and a descriptor for a scaled matmul is out of scope. When it
returns it belongs on `KCut`, as a property of a cut, not as another epilogue enum.

## What this cannot say

The k reduction here is a nest of cuts that ends in one chain of hardware steps. Real
kernels exist that do more than that -- for example a vector-wide load whose elements
form their own summation group underneath a lane's chain. A producer that meets one
must decline rather than write down the nearest `GEMMDesc`: a nearly-right order is
wrong bits, and losing coverage is much cheaper than answering wrong.

## The two relations

There are two relations in the module, and they are not the same. `a == b` says the
two descriptions are identical. `produces_same_bits(a, b, known)` says a particular
machine hands back the same bits for both, which is weaker: it also holds for
descriptions that differ in a way that machine cannot show you. Which differences
those are is measured per architecture, so `known` is an argument. The module defines
the shape of that answer and ships none of its content.

A `GEMMDesc` is a value: it is frozen, it compares and hashes by its fields, and
there is no module state anywhere near it, so it can be used as a cache key without
a second thought. Equality never depends on which machine you are on; that is what
the other relation is for.

`Equivalences` is that argument. It holds groups of algorithms a machine cannot tell
apart, and whether the two values of `use_fast_accum` are free there. Building one is
the job of whoever did the measuring. One instance covers one machine and one
question: the Blackwell fp8 exception is the reason a producer asking about fp8 there
and one asking about fp16 need different instances, and a single instance that
averaged the two would be wrong for both. Picking the matching one is the caller's
job. `Equivalences()` -- nothing declared -- is the right value for a machine nobody
has measured, and under it `produces_same_bits` is exactly `==`.

Two descriptors that are equal except that one says `MMA_SYNC` and the other
`TCGEN05_MMA` produce the same bits when one group holds both.

The groups must not overlap, and construction rejects a pair that does. `{A, B}` and
`{B, C}` together would say that A and C agree, which is a wider claim than either
group makes; growing a claim quietly is how a wrong answer becomes a silent one, so
it has to be written out instead.

A group may name a scalar algorithm, which is what an integer GEMM would want:
integer arithmetic is exact, so there fma and multiply-then-add really do agree.
Integer addition is also associative, so for an integer GEMM the ordering fields
describe the kernel without constraining the answer. Mixing a scalar algorithm and a
matrix instruction into one group is allowed but can never make two descriptors
equivalent, because the matrix one carries `instruction_k` and the scalar one may
not, and that difference stays.

`fast_accum_is_free` says nothing about a scalar algorithm: there `use_fast_accum`
is `None`, so there is no second form for a machine to make free.

The relation fails closed. The declared differences are rewritten to one form and
then every remaining field is compared, rather than the significant fields being
listed out. So a difference nobody has measured keeps the answer `False` --
including a difference in a field added to `GEMMDesc` after the relation was
written, which is the case a hand-written list would have got wrong. A difference
stops counting only when someone measures it and gives it a field on `Equivalences`.

The canonical form it compares is a dict of every field and not another `GEMMDesc`,
for two reasons: a canonical form need not be a legal descriptor, and rebuilding one
would re-run the constructor's checks and raise where the answer should simply be
`False`. Comparing the dicts compares `k_cuts` too -- it is a tuple of frozen
`KCut`s, so `==` walks it element by element and compares every field of every cut.
