Semantic Checking
=================

The semantic checking logic in the Slang compiler is located in `source/slang/slang-check*`.
Semantic checking is applied in the front end after parsing, and before lowering of code to the IR.

The main job of the semantic checking stage is to detect and forbid code that has errors in it.
The errors and other diagnostics reported are intended to be of benefit to the user, but semantic checking is also important for the overall function of the compiler.
Stages of compilation after semantic checking (e.g., lowering to the IR) are allowed to *assume* that the code they operate on is semantically valid, and may assert-fail or even crash on invalid code.
Semantic checking is thus not an optional step, and there is no meaningful way to turn it off.

Semantic Checking can be broken into three main kinds of work, and we will discuss how each is implemented in the following sections:

* Checking of "terms" which include expressions and type expressions

* Checking of statements

* Checking of declarations

Checking Terms
--------------

### Some Terminology for Terms

We use the word "term" to refer generically to something that can be evaluated to produce a result, but where we do not yet know if the result will be a type or a value. For example, `Texture2D` might be a term that results in a type, while `main` might be a term that results in a value (of function type), but both start out as a `NameExpr` in the AST. Thus the AST uses the class hierarchy under `Expr` to represent terms, whether they evaluate to values or types.

There is also the `Type` hierarchy, but it is important to understand that `Type` represents types as their logical immutable selves, while `Expr`s that evaluate to types are *type expressions* which can be concretely pointed to in the user's code. Type expressions have source locations, because they represent something the user wrote in their code, while `Type`s don't have singular locations by default.

The codebase uses the notion of a `TypeRepr` for those `Expr`s that should only ever evaluate to types, and there is also a `TypeExp` type that is meant to package up a `Type` with an optional `Expr` for a type expression that produced it. The names of these implementation types aren't great, and should probably not be spread further.

A value-bearing `Expr` will eventually be given a `Type` that describes the type of value it produces.
An `Expr` that evaluates to a type will eventually be given a `Type` that uses the `TypeType` subclass to indicate the specific type it evaluated to.
The `TypeType` idea is kind of kludge to represent "kinds" (the "types of types") in our system.
More correctly, we should say that every `Expr` gets a *classifier*, with the classifiers for value expressions being `Type`s and the classifiers for type expressions being kinds, but we haven't had time or inclination to fix the model yet.

### The Big Picture

Checking of terms is largely done as an ad hoc postorder traversal of the AST.
That is, in order to check a compound expression like `f(a)` we first need to check `f` and `a` before we can check the function call.

When checking an expression there are four main things that have to be done:

1. Recursively check all sub-expressions.

2. Detect and diagnose any errors (or warnings) in the current expression.

3. Optionally construct a new expression to replace the current expression (or one of its sub-expressions) in cases where the syntactic form of the input doesn't match the desired semantics (e.g., make an implicit type conversion explicit in the AST).

4. Determine the correct type for the result expression, and store it so that it can be used by subsequent checking.

Those steps may end up being interleaved in practice.

### Handling Errors Gracefully

If an error is detected in a sub-expression, then there are a few issues that need to be dealt with:

* We need to ensure that an erroneous sub-expression can't crash the compiler when it goes on to check a parent expression. For example, leaving the type of an expression as null when it has errors is asking for trouble.

* We ideally want to continue to diagnose other unrelated errors in the same expression, statement, function, or file. That means that we shouldn't just bail out of semantic checking entirely (e.g., by throwing an exception).

* We don't want to produce "cascading" errors where, e.g., an error in `a` causes us to also report an error in `a + b` because no suitable operator overload was found.

We tackle all of these problems by introducing the `ErrorType` and `ErrorExpr` classes.
If we can't determine a correct type for an expression (say, because it has an error) then we will assign it the type `ErrorType`.
If we can't reasonably form an expression to return *at all* then we will return an `ErrorExpr` (which has type `ErrorType`).

These classes are designed to make sure that subsequent code won't crash on them (since we have non-null objects), but to help avoid cascading errors.
Some semantic checking logic will detect `ErrorType`s on sub-expressions and skip its own checking logic (e.g., this happens for function overload resolution), producing an `ErrorType` further up.
In other cases, expressions with `ErrorType` can be silently consumed.
For example, an erroneous expression is implicitly convertible to *any* type, which means that assignment of an error expression to a local variable will always succeed, regardless of variable's type.

### Overload Resolution

One of the most involved parts of expression checking is overload resolution, which occurs when there is an expression of the form `f(...)` where `f` could refer to multiple function declarations.

Our basic approach to overload resolution is to iterate over all the candidates and add them to an `OverloadResolveContext`.
The context is responsible for keeping track of the "best" candidate(s) seen so far.

Traditionally a language defines rules for which overloads are "better" than others that focus only on candidates that actually apply to the call site.
This is the right way to define language semantics, but it can produce sub-optimal diagnostics when *no* candidate was actually applicable.

For example, suppose the user wrote `f(a,b)` and there are 100 functions names `f`, but none works for the argument types of `a` and `b`.
A naive approach might just say "no overload applicable to arguments with such-and-such types."
A more advanced compiler might try to list all 100 candidates, but that wouldn't be helpful.
If it turns out that of the 100 candidates, only 10 of them have two parameters, then it might be much more helpful to list only the 10 candidates that were even remotely applicable at the call site.

The Slang compiler strives to provide better diagnostics on overload resolution by breaking the checking of a candidate callee into multiple phases, and recording the earliest phase at which a problem was detected (if any).
Candidates that made it through more phases of checking without errors are considered "better" than other candidates, even if they ultimately aren't applicable.

### Type Conversions

Conversion of values from one type to another can occur both explicitly (e.g., `(int) foo`) and implicitly (e.g., `while(foo)` implicitly converts `foo` to a `bool`).

Type conversion also tied into overload resolution, since some conversions get ranked as "better" than others when deciding between candidates (e.g., converting an `int` to a `float` is preferred over converting it to a `double`).

We try to bottleneck all kinds of type conversion through a single code path so that the various kinds of conversion can be handled equivalently.

### L-Values

An *l-value* is an expression that can be used as the destination of an assignment, or for read-modify-write operations.

We track the l-value-ness of expressions using `QualType` which basically represents a `Type` plus a bit to note whether something is an l-value or not.
(This type could eventually be compressed down to be stored as a single pointer, but we haven't gotten to that yet)
We do not currently have a concept like the `const` qualifier in C/C++, that would be visible to the language user.

Propagation of l-value-ness is handled in an ad hoc fashion in the small number of expression cases that can ever produce l-values.
The default behavior is that expressions are not l-values and the implicit conversion from `Type` to `QualType` reflects this.

Checking Statements
-------------------

Checking of statements is relatively simpler than checking expressions.
Statements do not produce values, so they don't get assigned types/classifiers.
We do not currently have cases where a statement needs to be transformed into an elaborated form as part of checking (e.g., to make implicit behavior explicit), so statement checking operates "in place" rather than optionally producing new AST nodes.

The most interesting part of statement checking is that it requires information about the lexical context.
Checking a `return` statement requires knowing the surrounding function and its declared result type.
Checking a `break` statement requires knowing about any surrounding loop or `switch` statements.

We represent the surrounding function explicitly on the `SemanticsStmtVisitor` type, and also use a linked list of `OuterStmtInfo` threaded up through the stack to track lexically enclosing statements.

Note that semantic checking of statements at the AST level does *not* encompass certain flow-sensitive checks.
For example, the logic in `slang-check-stmt.cpp` does not check for or diagnose any of:

* Functions that fail to `return` a value along some control flow paths

* Unreachable code

* Variables used without being initialized first

All of the above are instead intended to be handled at the IR level (where dataflow analysis is easier) during the "mandatory" optimization passes that follow IR lowering.

Checking Declarations
---------------------

Checking of declarations is the most complicated and involved part of semantic checking.

### The Problem

Simple approaches to semantic checking of declarations fall into two camps:

1. One can define a total ordering on declarations (usually textual order in the source file) and only allow dependencies to follow that order, so that checking can follow the same order. This is the style of C/C++, which is inherited from the legacy of traditional single-pass compilers.

2. One can define a total ordering on *phases* of semantic checking, so that every declaration in the file is checked at phase N before any is checked at phase N+1. E.g., the types of all variables and functions must be determined before any expressions that use those variables/functions can be checked. This is the style of, e.g., Java and C#, which put a premium on defining context-free languages that don't dictate order of declaration.

Slang tries to bridge these two worlds: it has inherited features from HLSL that were inspired by C/C++, while it also strives to support out-of-order declarations like Java/C#.
Unsurprisingly, this leads to unique challenges.

Supporting out-of-order declarations means that there is no simple total order on declarations (we can have mutually recursive function or type declarations), and supporting generics with value parameters means there is no simple total order on phases.
For that last part observe that:

* Resolving an overloaded function call requires knowing the types of the parameters for candidate functions.

* Determining the type of a parameter requires checking type expressions.

* Type expressions may contain value arguments to generics, so checking type expressions requires checking value expressions.

* Value expressions can include function calls (e.g., operator invocations), which then require overload resolution to type-check.

### The Solution

Our declaration checking logic takes the idea of phase-based checking as a starting point, but instead of a global ordering on phases we use a per-declaration order.

Each declaration in the Slang AST will have a `DeclCheckState` that represents "how checked" that declaration is.
We can apply semantic checking logic to a declaration `D` to raise its state to some desired state `S`.

By default, the logic in `slang-check-decl.cpp` will do a kind of "breadth-first" checking strategy where it will try to raise all declarations to the one state before moving on to the next.
In many cases this will reproduce the behavior of a Java or C#-style compiler with strict phases.

The main difference for Slang is that whenever, during the checking of some declaration `D`, we discover that we need information from some other declaration `E` that would depend on `E` being in state `S`, we manually call a routine `ensureDecl(E,S)` whose job is to ensure that `E` has been checked enough for us to proceed.

The `ensureDecl` operation will often be a no-op, if the declaration has already been checked previously, but in cases where the declaration *hasn't* been checked yet it will cause the compiler to recursively re-enter semantic checking and try to check `E` until it reached the desired state.

In pathological cases, this method can result in unbounded recursion in the type checker. The breadth-first strategy helps to make such cases less likely, and introducing more phases to semantic checking can also help reduce problems.
In the long run we may need to investigate options that don't rely on unbounded recursion.

### The Rules

As a programmer contributing to the semantic checking infrastructure, the declaration-checking strategy requires following a few rules:

* If a piece of code is about to rely on some property of a declaration that might be null/absent/wrong if checking hasn't been applied, it should use `ensureDecl` to make sure the declaration in question has been checked enough for that property to be available.

* If adding some `ensureDecl`s leads to an internal compiler error because of circularity in semantic checking, then either the `ensureDecl`s were misplaced, or they were too strong (you asked for more checking than was necessary), or in the worse case we need to add more phases (more `DeclCheckState`s) to separate out the checking steps and break the apparent cycle.

* In very rare cases, semantic checking for a declaration may want to use `SetCheckState` to update the state of the declaration itself before recursively `ensureDecl`ing its child declarations, but this must be done carefully because it means you are claiming that the declaration is in some state `S`, while not having complete the checking that is associated with state `S`.

* It should *never* be necessary to modify `checkModuleDecl` so that it performs certain kinds of semantic analysis on certain declarations before others (e.g., iterate over all the `AggTypeDecl`s before all the `FuncDecl`s). If you find yourself tempted to modify it in such a way, then add more `DeclCheckState`s to reflect the desired ordering. It is okay to have phases of checking that only apply to a subset of declarations.

* Every statement and expression/term should be checked once and only once. If something is being checked twice and leading to failures, the right thing is to fix the source of the problem in declaration checking, rather than make the expression/statement checking be defensive against this case.

Name Lookup
-----------

Lookup is the processing of resolving the contextual meaning of names either in a lexical scope (e.g., the user wrote `foo` in a function body - what does it refer to?) or in the scope of some type (e.g., the user wrote `obj.foo` for some value `obj` of type `T` - what does it refer to?).

Lookup can be tied to semantic analysis quite deeply.
In order to know what a member reference like `obj.foo` refers to, we not only need to know the type of `obj`, but we may also need to know what interfaces that type conforms to (e.g., it might be a type parameter `T` with a constraint `T : IFoo`).
In order to support lookup in the presence of our declaration-checking strategy described above, the lookup logic may be passed a `SemanticsVisitor` that it can use to `ensureDecl()` declarations before it relies on their properties.

However, lookup also currently gets used during parsing, and in those cases it may need to be applied without access to the semantics-checking infrastructure (since we currently separate parsing and semantic analysis).
In those cases a null `SemanticsVisitor` is passed in, and the lookup process will avoid using lookup approaches that rely on derived semantic information.
This is fine in practice because the main thing that gets looked up during parsing are names of `SyntaxDecl`s (which are all global) and also global type/function/variable names.


Known Issues
------------

The largest known issue for the semantic checking logic is that there are currently dependencies between parsing and semantic checking.
Just like a C/C++ parser, the Slang parser sometimes needs to disambiguate whether an identifier refers to a type or value to make forward progress, and that would in general require semantic analysis.

Ideally the way forward is some combination of the following two strategies:

* We should strive to make parsing at the "global scope" fully context-insensitive (e.g., by using similar lookahead heuristics to C#). We are already close to this goal today, but will need to be careful that we do not introduce regressions compared to the old parser (perhaps a "compatibility" mode for legacy HLSL code is needed?)

* We should delay the parsing of nested scopes (both function and type bodies bracketed with `{}`) until later steps of the compiler. Ideally, parsing of function bodies can be done in a context-sensitive manner that interleaves with semantic checking, closer to the traditional C/C++ model (since we don't care about out-of-order declarations in function bodies).

