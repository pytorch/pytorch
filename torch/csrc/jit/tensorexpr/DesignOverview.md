# Current workflow

## Step 1: input from the user.

User construct a kernel from tensor expressions, like:
```
    Buffer a_buf("a", kFloat32, {M, N});
    Buffer b_buf("b", kFloat32, {N, K});
    Buffer c_buf("c", kFloat32, {M, N});
    Buffer d_buf("d", kFloat32, {M, K});

    Tensor* x = Compute(
        "x",
        {{M, "m1"}, {N, "n1"}, {K, "k1"}},
        [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
          return a_buf(m, n) * b_buf(n, k);
        });
    Tensor* y = ...;
    Tensor* z = ...;
    std::vector<Tensor*> tensors_to_compute = {x, z}; // Tensor y might be used in x or z - in this case it will also be computed. 
```

## Step 2: Lower to a LoopNest:
```
   LoopNest l(tensors_to_compute);
```
LoopNest consists of a root statement (`Stmt`) and some metadata. The root statement of a loop nest is a block statement containing other statements.

A statement can be one of the following:
 - `Store` statement: such statements represent access to tensor elements. They specify the base variable (`Var`), an expression for the index, an expression for the stored value, and the mask.
 - `LetStmt` statement: 'let' statements are used for binding variables to given expressions. Such statements consist of the variable to bind (`Var`), the expression to bind to, and the body statement in which the binding should be performed.
 - `For` statement: these statements represent a loop. They specify the index variable (`Var`), expressions for the beginning and the end of the iteration space, a `Block` statement for the body, and some metadata.
 - `Cond` statement: these statements represent if-s: they consist of a condition expression and two `Block` statements for true and false branches (both are allowed to be null).
 - `Block` statement: these statements represent a linear sequence of other statements.

An example of a root statement:
```
for (int m = 0; m < 100; m++) {
  for (int n = 0; n < 200; n++) {
    c[m * 200 + n] = a[m * 200 + n + 1] + a[m * 200 + n];
  }
}
for (int i = 0; i < W; i++) {
  q[i] = i + 1
}
```

## Step 3: Apply loop transformations
One can apply various loop transformations on a loop nest. The transformations mutate statements in the loop nest and loop nest can record the history of the transformations.

## Step 4: Prepare loop nest for codegen
After all desired loop transformations are applied, a final transformation is carried out on the loop nest's root statement. A result of this transformation is also a statement, but it now can include a lower-level statements like `Allocate` and `Free`, which are not allowed to exist during the loop transformation phase.

## Step 5: Pass the final statement for codegen (LLVM/CUDA/IREval)
Codegen is implemented as an IR visitor over the statements produced in the previous step.

# Tensor Expressions Language
There are several core concepts in the Tensor Expression engine, this section defines them and shows how they connect to each other.

## Expr
Expr represents a node in the abstract syntax tree of a tensor expression. Leaf nodes in such tree are either a symbolic variable (`Var`), a constant (`IntImm` or `FloatImm`), `Buffer`, or a `Tensor`. Non-leaf nodes refer to other expressions and represent various operations. E.g. `Add` has two operands: `lhs` and `rhs`, both of which are also `Expr`.

## Tensor
`Tensor` is a bundle of
1) a variable `Var` defining which tensor this `Tensor` expression is describing
2) a list of indices `args` (each of them is `Var`)
3) a list of expressions for dimensions `dims` (each of them is `Expr`)
4) a computational expression `body` (of `Expr` type)

## Buffer
`Buffer`s are essentially `Tensor`s without a `body` - they represent an indexed access to "tensors" that is outsied the tensor-expression system.
`Buffer` is a bundle of
1) a `Var` defining which buffer this `Buffer` expression is defining
2) a list of indices `args` (each of them is `Var`)
3) a list of expressions for dimensions `dims` (each of them is `Expr`)

## Example
Suppose we'd like to represent the following expression:
```
A[i,j] = B[i,j] + 7
```
where both `A` and `B` are 100x100 tensors.
On the top level we would have a single `Tensor` expression with:
1) a variable referring to "A"
2) list of two indices referring to "i" and "j"
3) list of two `IntImm` constants describing sizes (both of them would carry the value of 100)
4) a body expression which is an `Add` with two operands: `Buffer` describing `B[i,j]` access and an `IntImm` constant `7`.

The buffer expression describing `B[i,j]` would have similar properties:
1) a variable referring to "B"
2) list of two indices referring to "i" and "j"
3) list of two `IntImm` constants describing sizes (both of them would carry the value of 100)

In contrast to the tensor expression, the buffer expression would not have a body - it represents a symbolic access.

The code for constructing such an expression could look like this:

```
    Buffer B("B", kFloat32, {100, 100});
    Tensor* A = Compute(
        "A",
        {{100, "i"}, {100, "j"}},
        [&](const VarHandle& i, const VarHandle& j) {
          return B(i, j) + 7;
        });
```

## Function
`Function` represents several tensor computations bundled together. In fact, `Tensor`s are implemented via `Function`s. A function allows us to specify that several different tensor expressions operate over the same set of indices and dimensions.

# Memory model
TBD

# Integartion with PyTorch JIT
TBD
