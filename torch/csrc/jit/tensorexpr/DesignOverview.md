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

## Step 2: Create schedule for the tensor expressions:
```
   Schedule s(tensors_to_compute);
```
This constructs a tree-like data structure (`TensorExprNode`) representing loop nests for the given tensor computation.
A node in this IR is either a loop-axis(LoopAxis) or a tensor expression (`TensorExprOp`).
If it is a loop-axis, it also contains children that again might be either a loop-axes or a tensor expression, and so on.
If it is a tensor-expression, it is lowered to a statement (`Stmt`). Currently, it just means that we're creating a `Store` for every tensor-expression. We also keep a pointer to the original tensor expression.
It could look like this:
```
loop-axis i
  loop-axis j
    Store(to: a[i, j], what: x[i] + y[j])
loop-axis k
  loop-axis l
    Store(to: b[k, l], what: a[i, j] + 1)
    loop-axis m
      Store(to: c[k,l,m], what: b[k,l] + z[m])
```

## Step 3: Apply scheduling primitives
Scheduling primitives mutate the tree structure: they can create or remove loop-axis, replace statements with other statements (updates `element_stmt` for each affected tensor expression) or remove them. The transformations also record the history.
The output of this step is a modified tree-like structure (same format as in step 2).

## Step 4: Lower the tree structure to statements.
This step creates a `For` statement for each loop-axis and emits `element_stmt` for bodies of the loops.

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

## Stmt
`Stmt`s are what tensor expressions are lowered to before the codegen. They represent the computation in a less abstract way, compared to pure tensor expressions. Statements are built upon expressions, i.e. they can contain expressions as operands. Statement is a unit that a codegen works with, it is incorrect to try to pass an expression to a codegen.
An example of statements are `Store` and `For`.
TODO: provide more detailed example/description for the stmt.

# Memory model
TBD
