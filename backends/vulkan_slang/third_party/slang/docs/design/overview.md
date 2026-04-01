An overview of the Slang Compiler
=================================

This document will attempt to walk through the overall flow of the Slang compiler, as an aid to developers who are trying to get familiar with the codebase and its design.
More emphasis will be given to places where the compiler design is nontraditional, or might surprise newcomers; things that are straightforward won't get much detail.

High-Level Concepts
-------------------

Compilation is always performed in the context of a *compile request*, which bundles together the options, input files, and request for code generation.
Inside the code, there is a type `CompileRequest` to represent this.

The user specifies some number of *translation units* (represented in the code as a `TranslationUnitRequest`) which comprise some number of *sources* (files or strings).
HLSL follows the traditional C model where a "translation unit" is more or less synonymous with a source file, so when compiling HLSL code the command-line `slangc` will treat each source file as its own translation unit.
For Slang code, the command-line tool will by default put all source files into a single translation unit (so that they represent a shared namespace0).

The user can also specify some number of *entry points* in each translation unit (`EntryPointRequest`), which combines the name of a function to compile with the pipeline stage to compile for.

In a single compile request, we can generate code for zero or more *targets* (represented with `TargetRequest`) a target defines both the format for output code (e.g., DXIL or SPIR-V) and a *profile* that specifies the capability level to assume (e.g., "Shader Model 5.1").

It might not be immediately clear why we have such fine-grained concepts as this, but it ends up being quite important to decide which pieces of the compiler are allowed to depend on which pieces of information (e.g., whether or not a phase of compilation gets to depend on the chosen target).

The "Front End"
---------------

The job of the Slang front-end is to turn textual source code into a combination of code in our custom intermediate representation (IR) plus layout and binding information for shader parameters.

### Lexing

The first step in the compiler (after a source file has been loaded into memory) is to *lex* it.
The `Lexer` type is implement in `lexer.{h,cpp}` and produces `Token`s that represent the contents of the file on-demand as requested by the next phase of compilation.

Each token stores a `TokenCode` that indicates the kind of token, the raw text of the token, and the location in the source code where it is located.
Source locations use a somewhat clever encoding to avoid being bloated (they are a single integer rather than separate file, line, and column fields).

We don't make any attempt in the lexer to extract the actual value of integer and floating-point literals; we just store the raw text.
We also don't try to distinguish keywords from identifiers; keywords show up as ordinary identifier tokens.

Much of the complexity (and inefficiency) in the current lexer is derived from the need to support C-isms like backspace line continuation, and special case rules like allowing `<>` to delimit a file name string after a `#include`.

### Preprocessing

The preprocessor (`Preprocessor`) in `preprocessor.{h,cpp}` deals with `#include` constructs, macro expansions, etc.
It pulls tokens from the lexer as needed (making sure to set flags to control the lexer behavior when required) and uses a limited lookahead to decide what to do with each token.

The preprocessor maintains a stack of input streams, with the original source file at the bottom, and pushes entries for `#include`d files, macros to expand etc.

Macro definitions store a sequence of already-lexed tokens, and expansion simply "replays" these tokens.
Expansion keeps a notion of an "environment" for looking up identifiers and mapping them to macro definitions.
Calling through to a function-style macro creates a fresh environment that maps the macro parameter names to pseudo-macros for the arguments.

We still tokenize code in inactive preprocessor conditionals, but don't evaluate preprocessor directives inside inactive blocks (except those that may change the active/inactive state).
Preprocessor directives are each handled as a callback on the preprocessor state and are looked up by name; adding a new directive (if we ever had a reason to) is a fairly simple task.

One important detail of the preprocessor is that it runs over a full source file at once and produces a flat array of `Token`s, so that there is no direct interaction between the parser and preprocessor.

### Parsing

The parser (`Parser` in `parser.{h,cpp}`) is mostly a straightforward recursive-descent parser.
Because the input is already tokenized before we start, we can use arbitrary lookahead, although we seldom look ahead more than one token.

Traditionally, parsing of C-like languages requires context-sensitive parsing techniques to distinguish types from values, and deal with stuff like the C++ "most vexing parse."
Slang instead uses heuristic approaches: for example, when we encounter an `<` after an identifier, we first try parsing a generic argument list with a closing `>` and then look at the next token to determine if this looks like a generic application (in which case we continue from there) or not (in which case we backtrack).

There are still some cases where we use lookup in the current environment to see if something is a type or a value, but officially we strive to support out-of-order declarations like most modern languages.
In order to achieve that goal we will eventually move to a model where we parse the bodies of declarations and functions in a later pass, after we have resolved names in the global scope.

One important choice in the parser is that we strive to avoid hard-coding keywords as much as possible.
We already track an environment for C-like parsing, and we simply extend that so that we also look up declaration and statement keywords in the environment.
This means that most of the language "keywords" in Slang aren't keywords at all, and instead are just identifiers that happen to be bound to syntax in the default environment.
Syntax declarations are associated with a callback that is invoked to parse the construct they name.

The design of treating syntax as ordinary declarations has a long-term motivation (we'd like to support a flexible macro system) but it also has short-term practical benefits.
It is easy for us to add new modifier keywords to the language without touching the lexer or parser (just adding them to the core module), and we also don't have to worry about any of Slang's extended construct (e.g., `import`) breaking existing HLSL code that just happens to use one of those new keywords as a local variable name.

What the parser produces is an abstract syntax tree (AST).
The AST currently uses a strongly-typed C++ class hierarchy with a "visitor" API generated via some ugly macro magic.
Dynamic casting using C++ RTTI is used in many places to check the class of an AST node; we aren't happy with this but also haven't had time to implement a better/faster solution.

In the parsed AST, both types and expressions use the same representation (because in an expression like `A(B)` it is possible that `A` will resolve to a type, or to a function, and we don't know which yet).

One slightly odd design choice in the parser is that it attaching lexical scoping information to the syntax nodes for identifiers, and any other AST node that need access to the scope/environment where it was defined. This is a choice we will probably change at some point, but it is deeply ingrained right now.

### Semantic Checking

The semantic checking step (`check.{h,cpp}`) is, not surprisingly, the most complicated and messiest bit of the compiler today.
The basic premise is simple: recursively walk the entire AST and apply semantic checking to each construct.

Semantic checking applies to one translation unit at a time.
It has access to the list of entry points for the translation unit (so it can validate them), but it *not* allowed to depend on the compilation target(s) the user might have selected.

Semantic checking of an expression or type term can yield the same AST node, with type information added, or it can return newly constructed AST needs (e.g., when an implicit cast needs to be inserted).
Unchecked identifiers or member references are always resolved to have a pointer to the exact declaration node they are referencing.

Types are represented with a distinct class hierarchy from AST nodes, which is also used for a general notion of compile-time values which can be used to instantiate generic types/functions/etc.
An expression that ends up referring to a type will have a `TypeType` as its type, which will hold the actual type that the expression represents.

The most complicated thing about semantic checking is that we strive to support out-of-order declarations, which means we may need to check a function declaration later in the file before checking a function body early in the file.
In turn, that function declaration might depend on a reference to a nested type declared somewhere else, etc.
We currently solve this issue by doing some amount of on-demand checking; when we have a reference to a function declaration and we need to know its type, we will first check if the function has been through semantic checking yet, and if not we will go ahead and recursively type check that function before we proceed.

This kind of unfounded recursion can lead to real problems (especially when the user might write code with circular dependencies), so we have made some attempts to more strictly "phase" the semantic checking, but those efforts have not yet been done systematically.

When code involved generics and/or interfaces, the semantic checking phase is responsible for ensuring that when a type claims to implement an interface it provides all of the requirements of that interface, and it records the mapping from requirements to their implementations for later use. Similarly, the body of a generic is checked to make sure it uses type parameters in ways that are consistent with their constraints, and the AST is amended to make it explicit when an interface requirement is being employed.

### Lowering and Mandatory Optimizations

The lowering step (`lower-to-ir.{h,cpp}`) is responsible for converting semantically valid ASTs into an intermediate representation that is more suitable for specialization, optimization, and code generation.
The main thing that happens at this step is that a lot of the "sugar" in a high-level language gets baked out. For example:

- A "member function" in a type will turn into an ordinary function that takes an initial `this` parameter
- A `struct` type nested in another `struct` will turn into an ordinary top-level `struct`
- Compound expressions will turn into sequences of instructions that bake the order of evaluation
- High-level control-flow statements will get resolved to a control-flow graph (CFG) of basic blocks

The lowering step is done once for each translation unit, and like semantic checking it does *not* depend on any particular compilation target.
During this step we attach "mangled" names to any imported or exported symbols, so that each function overload, etc. has a unique name.

After IR code has been generated for a translation unit (now called a "module") we next perform a set of "mandatory" optimizations, including SSA promotion and simple copy propagation and elimination of dead control-flow paths.
These optimizations are not primarily motivated by a desire to speed up code, but rather to ensure that certain "obvious" simplifications have been performed before the next step of validation.

After the IR has been "optimized" we perform certain validation/checking tasks that would have been difficult or impossible to perform on the AST.
For example, we can validate that control flow never reached the end of a non-`void` function, and issue an error otherwise.
There are other validation tasks that can/should be performed at this step, although not all of them are currently implemented:

- We should check that any `[unroll]` loops can actually be unrolled, by ensuring that their termination conditions can be resolved to a compile-time constant (even if we don't know the constant yet)

- We should check that any resource types are being used in ways that can be statically resolved (e.g., that the code never conditionally computes a resource to reference), since this is a requirement for all our current targets

- We should check that the operands to any operation that requires a compile-time constant (e.g., the texel offset argument to certain `Sample()` calls) are passed values that end up being compile-time constants

The goal is to eliminate any possible sources of failure in low-level code generation, without needing to have a global view of all the code in a program.
Any error conditions we have to push off until later starts to limit the value of our separate compilation support.

### Parameter Binding and Type Layout

The next phase of parameter binding (`parameter-binding.{h,cpp}`) is independent of IR generation, and proceeds based on the AST that came out of semantic checking.
Parameter binding is the task of figuring out what locations/bindings/offsets should be given to all shader parameters referenced by the user's code.

Parameter binding is done once for each target (because, e.g., Vulkan may bind parameters differently than D3D12), and it is done for the whole compile request (all translation units) rather than one at a time.
This is because when users compile something like HLSL vertex and fragment shaders in distinct translation units, they will often share the "same" parameter via a header, and we need to ensure that it gets just one location.

At a high level, parameter binding starts by computing the *type layout* of each shader parameter.
A type layout describes the amount of registers/bindings/bytes/etc. that a type consumes, and also encodes the information needed to compute offsets/registers for individual `struct` fields or array elements.

Once we know how much space each parameter consumes, we then inspect an explicit binding information (e.g., `register` modifiers) that are relevant for the target, and build a data structure to record what binding ranges are already consumed.
Finally, we go through any parameters without explicit binding information and assign them the next available range of the appropriate size (in a first-fit fashion).

The parameter binding/layout information is what the Slang reflection API exposes. It is layered directly over the Slang AST so that it accurately reflects the program as the user wrote it, and not the result of lowering that program to our IR.

This document describes parameter binding as a "front end" activity, but in practice it is something that could be done in the front-end, the back-end or both.
When shader code involves generic type parameters, complete layout information cannot be generated until the values of these parameters are fully known, and in practice that might not happen until the back end.

### Serialization

It is not yet fully implemented, but our intention is that the last thing the front-end does is to serialize the following information:

- A stripped-down version of the checked AST for each translation unit including declarations/types, but not function bodies

- The IR code for each translation unit

- The binding/layout information for each target

The above information is enough to type-check a subsequent module that `import`s code compile in the front-end, to link against its IR code, or to load and reflect type and binding information.


The "Back End"
--------------

The Slang back end logically starts with the user specifying:

- An IR module, plus any necessary modules to link in and provide its dependencies

- An entry point in that module, plus arguments for any generic parameters that entry point needs

- A compilation target (e.g., SPIR-V for Vulkan)

- Parameter binding/layout information for that module and entry point, computed for the chosen target

We eventually want to support compiling multiple entry points in one pass of the back end, but for now it assumes a single entry point at a time

### Linking and Target Specialization

The first step we perform is to copy the chosen entry point and anything it depends on, recursively into a "fresh" IR module.
We make a copy of things so that any optimization/transformation passes we do for one target don't alter the code the front-end produced in ways that affect other targets.

While copying IR code into the fresh module, we have cases where there might be multiple definitions of the same function or other entity.
In those cases, we apply "target specialization" to pick the definition that is the best for the chosen target.
This step is where we can select between, say, a built-in definition of the `saturate` function for D3D targets, vs. a hand-written one in a Slang standard module to use for GLSL-based targets.

### API Legalization

If we are targeting a GLSL-based platform, we need to translate "varying" shader entry point parameters into global variables used for cross-stage data passing.
We also need to translate any "system value" semantics into uses of the special built-in `gl_*` variables.

We currently handle this kind of API-specific legalization quite early in the process, performing it right after linking.

### Generic Specialization

Once the concrete values for generic parameters are know we can set about specializing code to the known types.
We do this by cloning a function/type/whatever and substituting in the concrete arguments for the parameters.
This process can be continued as specializing one function may reveal opportunities to specialize others.

During this step we also specialize away lookup of interface requirements through their witness tables, once generic witness-table parameters have been replaced with concrete witness tables.

At the end of specialization, we should have code that makes no use of user-defined generics or interfaces.

### Type Legalization

While HLSL and Slang allow a single `struct` type to contain both "ordinary" data like a `float3` and "resources" like a `Texture2D`, the rules for GLSL and SPIR-V are more restrictive.
There are some additional wrinkles that arise for such "mixed" types, so we prefer to always "legalize" the types in the users code by replacing an aggregate type like:

```hlsl
struct Material { float4 baseColor; Texture2D detailMap; };
Material gMaterial;
```

with separate declarations for ordinary and resource fields:

```hlsl
struct Material { float4 baseColor; }

Material gMaterial;
Texture2D gMaterial_detailMap;
```

Changing the "shape" of a type like this (so that a single variable becomes more than one) needs to be done consistently across all declarations/functions in the program (hence why we do it after specialization, so that all concrete types are known).

### Other Optimizations

We dont' currently apply many other optimizations on the IR code in the back-end, under the assumption that the lower-level compilers below Slang will do some of the "heavy lifting."

That said, there are certain optimizations that Slang must do eventually, for semantic completeness. One of the most important examples of these is implementing the semantics of the `[unroll]` attribute, since we can't always rely on downstream compilers to have a capable unrolling implementation.

We expect that over time it will be valuable for Slang to support a wider array of optimization passes, as long as they are ones that are considered "safe" to do above the driver interface, because they won't interfere with downstream optimization opportunities.

### Emission

Once we have transformed the IR code into something that should be legal for the chosen target, we emit high-level source code in either HLSL or GLSL.

The emit logic is mostly just a scan over the IR code to emit a high-level declaration for each item: an IR structure type becomes a `struct` declaration, and IR function becomes a function definition, etc.

In order to make the generated code a bit more readable, the Slang compiler currently does *not* emit declarations using their mangled names and instead tries to emit everything using a name based on how it was originally declared.

To improve the readability of function bodies, the emit logic tries to find consecutive sequences of IR instructions that it can emit as a single high-level language expression. This reduces the number of temporaries in the output code, but we need to be careful about inserting parentheses to respect operator precedence, and also to not accidentally change the order of evaluation of code.

When emitting a function body, we need to get from the low-level control flow graph (CFG) to high-level structured control-flow statements like `if`s and loops. We currently do this on a per-function basis during code emission, using an ad hoc algorithm based on control-flow structured information we stored in the IR.
A future version of the compiler might implement something more complete like the "Relooper" algorithm used by Emscripten.

### Downstream Compiler Execution

Once we have source code, we can invoke downstream compilers like fxc, dxc, and glslang to generate binary code (and optionally to disassemble that code for console output).

The Slang compiler also supports a "pass through" mode where it skips most of the steps outlined so far and just passes text along to these downstream compilers directly. This is primarily intended as a debugging aid for developers working on Slang, since it lets you use the same command-line arguments to invoke both Slang compilation and compilation with these other compilers.

Conclusion
----------

Hopefully this whirlwind introduction to the flow of the Slang compiler gives some idea of how the project fits together, and makes it easier to dive into the code and start being productive.
