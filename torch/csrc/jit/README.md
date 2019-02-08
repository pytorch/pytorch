JIT Technical Overview
======================

The JIT can run and optimize PyTorch programs separate from the Python interpreter. This overview is organized into sections that go over different independent components:

1. Core Program Representation -  The JIT executes TorchScript, a subset of python. This section describes how TorchScript programs are represented in the JIT, and serves as the interchange format between components of the JIT.
2. Generating Programs - TorchScript programs can be created either through tracing python code or through directly writing TorchScript. This section describes how Models are created from these frontends.
3. Executing Programs - Once created, TorchScript models are optimized and run. Since this is a just-in-time compiler, programs are optimized as they are executed, so this section describes both how programs are optimized and how they get run.
4. Saving Programs - TorchScript is often created in Python and then used from C++. This section describes how the save and load process works.
5. Python Bindings - TorchScript code is normally created and used from Python, so this section describes how the Python components interact with the code in this directory.

For concepts that are actual classes in the JIT, we use capitalized words, e.g. Graph or Value.

Sections start with a reference to the file they are talking about.

Core Program Representation
---------------------------


### Modules ###

[script/module.h](script/module.h)

At the top level, all TorchScript programs are represented as a Module. Modules contain named Parameters (tensors used in training such as `weight` or `bias`), named Methods (functions that can be run on the module such as `forward`), and named submodules used for code organization. This mirrors the `nn.Module` objects used in Python. All TorchScript code is a member of some module. This includes pure functions such as those created by annotating a Python function with `@torch.jit.script`, which are represented internally as a Module that has a single method `forward` that contains the implementation of the function.

### Parameters ###

[script/module.h](script/module.h)

Modules contain Parameter objects, which simply hold a "slot" where a Tensor can be placed. These tensors are accessible by the Methods of the Module or the parent Module.

### Method ###

[script/module.h](script/module.h)

A Method is a piece of TorchScript code that takes a number of arguments and produces an output value. Methods have several subcomponents. A FunctionSchema describes the types and names of the input arguments and return value. A list of member_inputs describes which Parameters are accessed by the method (this is blank for pure functions). A Graph object describes the actual code inside the method. The Method also maintains a GraphExecutor which is used to actually execute the Graph that defines the method.

The Graph inside the Method is a pure function. The Parameters used by the Method are added as additional inputs to this graph before it is run. This allows the GraphExecutor to treat method inputs and method parameters the same for the purposes of optimization and execution, simplifying the process for executing programs.

Methods also contain helper functions for inserting calls to the Method from other Method objects.

### FunctionSchema ###

[aten/src/ATen/core/function_schema.h](../../../aten/src/ATen/core/function_schema.h)

Each Method has a FunctionSchema that describes the Types of the arguments and return values of a function. Operators (builtin primitives that are called by the Interpreter) also have FunctionSchema. FunctionSchema are analogous to a function _declaration_ in C++. They describe how to call the function but do not provide an implementation.

### Graph ###

[ir.h](ir.h)

Graphs are the root of the intermediate representation (IR) used to define the implementation of TorchScript functions. If you are familiar with [LLVM](llvm.org), they are analogous to an `llvm::Function` object. A Graph is composed of Nodes, Blocks, and Values. Nodes are instructions (e.g. do a matrix multiply). Nodes are organized into Blocks of sequentially executed Nodes. Each Node produces a list of output Values, and also consumes a list of input Values. Graphs in the JIT are in single-static assignment (SSA) form, meaning that each Value has precisely one defining Node that can be looked up directly from the Value (`Node* n = v.node()`).

Blocks, Nodes, and Values are _owned_ by the Graph they appear in and may only appear in a single Graph. This is enforced by assertions in the API. Creation and deletion of Block, Node, and Value objects is done via methods on Graph objects (e.g. `Graph::create`,  `Node::addOutput`, or `Node::addBlock`). This API also enforces certain consistency properties. For instance, `Node::destroy` removes a Node, but it is only valid to call this function if the Values produced by this node are no longer used, which can be accomplished using other functions such as `Value::replaceAllUsesWith`.

Because Graph owns all its Nodes, Values, and Blocks, these values are always passed around by raw pointer. Generally developers should not write code that holds Value, Node, or Block objects indefinitely without also holding a shared_ptr to their owning Graph.

### Node ###

[ir.h](ir.h)

A node represents a single built-in instruction such as a matrix multiply or a convolution. Each node has a `kind()` method that determines which builtin instruction the node represents. Different nodes (e.g. conv vs matrix-multiply) are represented using different kinds and not via subclassing of Node, as one would find in LLVM. A `kind()` is a `Symbol` object, which is just an "interned" string inside some namespace. Symbols can be created from strings, e.g. through `Symbol::fromQualString("aten::add")`, so there is not a closed set of `kind()` values that a Node might have. This design was chosen to allow the open registration of new operators and user-defined operators.

*Code in the JIT should always assume the universe of valid Node kinds is open and subject to be expanded.*

This reflects the reality of the PyTorch operator library where there are already several hundred valid operators.

Nodes produces output Values and take input Values as arguments. For instance, a matrix-multiply will take two input tensors and produce one output tensor. Nodes can produce multiple outputs. For instance `prim::UnpackTuple` splits a tuple into its components, so it has a number of outputs equal to the number of members of the tuple. Though Nodes may have multiple outputs, the number of outputs is _statically known_ for each Node. Operations which may produce a dynamic amount of results, e.g. splitting a tensor into chunks of size 2, will be represented as a operator that results a list object.

Because Nodes are not subclassed per-operator, it is very easy to construct invalid Nodes, e.g. by forgetting an input or an output, or by passing Values of the wrong Type. To help avoid this, Graph provides the method (`Graph::insert`) for constructing Nodes that guarantees Nodes have the correct setup. This method uses the database of registered Operators and their FunctionSchema to construct Nodes using that schema.

For Nodes representing built-in Operators, the method `Node::schema` can also look up the FunctionSchema registered for that Operator.

Each node also has a set of of attributes which are named integers, strings, floats, Tensors, and subgraphs, or lists of these types. These are used by special primitive operators to encode additional data in the Node. For instance `prim::Constant` defines a compile-time constant value. For Tensor constants, it will have a single Tensor attribute with the name `attr::value` which contains the value of the constant.

Attributes are _rarely used_. Operators like convolution or matrix-multiply have no attributes and take of their arguments through the input list. This includes things that might be typically through of as constants, like the stride of the convolution. In PyTorch, any of this information is potentially a dynamic property of the program so Nodes are always encoded in a way that allows these values to be dynamically determined. However, we recognize that many inputs are almost always constants, so we make it easy to quickly check if an input is constant and get its value with `c10::optional<IValue> Node::get(Symbol name)`, which returns an IValue (a concrete value for the input) in the case the node is constant and `nullopt` otherwise.

### Block ###

[ir.h](ir.h)

Nodes are organized into sequentially executed lists inside a Block. A Node is a member of precisely one Block. The Graph itself has a top-level `graph.block()`, and control-flow nodes (`prim::If` and `prim::Loop`) also also have sub-blocks. While it is possible to design a Graph representation that does not have a sequential order for nodes (i.e. a sea-of-nodes representation), we find it is much easier to debug and understand Blocks when there is a specific canonical order for all of the nodes. This does not preclude optimization passes from changing the order when it would improve performance, and the interpreter is potentially allowed to execute the block out-of-order if the re-ordering preserves the semantics much like an out-of-order processor. Having the ordering ensure that graphs can always be easily printed, and that we can easily step through the execution of a graph.

Control-flow is represented with using sub-blocks rather than a control-flow graph representation. A `prim::If` has one block for the true branch and one block for the else.
A `prim:Loop` has a block for the loop body (there is no condition block, instead the end of the loop body computes whether to re-enter the loop body). This representation ensures we have structured control-flow. Currently TorchScript does not allow for early returns, breaking out of loops early. This limitation makes a lot of optimizations easier and is true for the vast majority of networks. Our frontend permits certain forms of syntax sugar that allow a limited amount of re-writing of if statements to avoid needing to support early returns. A Node can lookup what Block it is in, and a Block and can look up its parent (either the Node that has it as a subblock, or `nullptr` for the main Block).

Values are Block-scoped. A Value is in scope for the remainder of the Block it is defined in, including in the sub-blocks of any Node defined after it. Values go out of scope at the end of the block in which they are defined.

Block also contain a list if input and output values. The meaning of these values depends on where the block is used. For the Graph's top-level block, these are inputs and outputs to the Graph, and line up with the FunctionSchema associated with a Method. For if-statements, the Blocks have no inputs, and the outputs are the new values of variables in the outer block whose values were altered in an if-statement. For example, the following code:

```
  b = x
  if cond:
    a = y + 1
    b = y
  else:
    a = z + 2
  return b + a
```
will turn into IR that looks like:

```
  a, b_next = prim::If(cond)
     block0:
       t = y + 1
       -> t, y
     block1:
       t2 = z + 2
       -> t2, b
  t3 = b_next + a
  return t3
  # when true: a, b_next = t, y
  # when false: a, b_next = t2, b
```

The outputs of the if-statement serve a role similar to "phi" nodes in traditional SSA control-flow graphs. A similar encoding is used to define how loop-carried dependencies are defined inside `prim::Loop` statements.

When Nodes are inserted into a Graph, they are inserted at a special "insertion point" that is part of the state of the Graph. On construction, this will go to the end of the Graph.

Each block has two dummy nodes that are not included in the list of nodes in the block. The `prim::Param` node represents the inputs to block and does have a `prev()` or `next()` node. The `prim::Return` node represents the outputs of a block.
The list of Nodes in a block is implemented as a circular linked list with the `prim::Return` Node serving as the beginning/end sentinel.  Inserting and deleting at arbitrary places is efficient. Developers may also encounter implementations inside of IR objects that use this fact (e.g. appending to a block is equivalent to putting the node before the `prim::Return` node).

Iterators for the `nodes()` list are invalided when the current Node they point to is moved or deleted. Otherwise iterators remain valid.

### Value ###

[ir.h](ir.h)

A Value represents data flowing through the operations in the program, e.g. the output of a matrix-multiply op. Value objects are always defined by a single Node (`v.node()`) due to single-static assignment form. For inputs to a Block/Graph, this node is a special `prim::Param` node that does not appear anywhere in the block's list of nodes. Value objects also have a Type (e.g. is it a tensor? a list? a tuple?) that provides a static guarantee that its value will be of that Type.

Value objects have methods on them to from the Value to its definition (`v.node()`) and to all of its uses `v.uses()`, which is a list of Nodes whose input list includes the value. Be careful when iterating over `v.uses()` while changing how `v` is used because each change to `v` will invalidate the `v.uses()` iterator.

Values are abstract representation of data in the program. When executing, the actual tensors, list, tuples, etc. are stored in IValues (_interpreter_ values), which are tagged unions of all possible values in TorchScript. In retrospect the name Value is a bit confusing because it seems like it should be the tagged union, but it originally came from analogy to `llvm::Value`, which serves the same purpose as `jit::Value`.


### Type ###

[aten/src/ATen/core/jit_type.h](../../../aten/src/ATen/core/jit_type.h)

TorchScript, unlike Python, is statically typed, so every Value has a Type associated with it, and every FunctionSchema has a list of argument types and a return type for a function. Type is the base class of a hierarchy of C++ objects that represent the built-in types of TorchScript. Types provide methods such as `Type::isSubtypeOf` that describe the typing relationships. Common type are:

* TensorType - the root type of all Tensors in the system.
* DimensionedTensorType - a tensor with a particular number of dimension and backend (e.g. a 4 dimensional cuda Tensor), a subtype of TensorType that only appears after shape analysis.
* CompleteTensorType - A subtype of DimensionedTensorType that adds fixed sizes (e.g. a [3 x 4] cuda tensor). This only appears from tracing at the moment.
* Tuples - e.g. Tuple[Tensor, Int]. Each member of the tuple is statically typed and the length of the tuple is statically known.
* List[T] - e.g. List[Tensor]. Mutable lists of a particular type.
* Optional[T] - e.g. Optional[Tensor], either the Tensor value or None.
* Dict[K, V] - e.g. Dict[String, Tensor], dictionaries

If type S is a subtype of P, then we can substitute an IValue that has type S anywhere something of type P is expected. This means that all subtyping relationships also require the representation of the IValue for subtypes to be compatible with the representation for the base type.


Generating Programs
-------------------

JIT programs are created using either the tracing frontend (`torch.jit.trace`) or the scripting frontend (`torch.jit.script`). In both cases, the result of these frontends is a complete Module that contains all the code in Methods, and all the model weights in the Parameters of the Module. However, each frontend goes through a different pathway for generating those Modules.

### Tracer ###


[tracer.h](tracer.h)
[tracer_state.h](tracer_state.h)

The tracer produces graphs by recording what actual operations are done on tensors.
The entrypoint from Python into C++ for tracing using `torch.jit.trace` is `_create_method_from_trace`.

A thread local instance of the TracingState object maintains a mapping between actual data being computing during the trace (e.g. Tensors) stored in IValues, and the abstract `Value*` in the Graph that would compute that value. The functions `void setValueTrace(const IValue&, Value*)` and `Value* getValueTrace(const IValue&)` are used by the tracer to maintain this mapping.

An initial IValue to Value mapping is setup up between the inputs to the function being traced and symbolic Value inputs to the Graph being constructed. If we are tracing a `torch.nn.Module`, the tracer also adds Parameters and sub-Modules to the Module being constructed that correspond to the Python `torch.nn.Module` being traced.  These values are also added as mapping so that uses of the Parameters in the trace will create uses of the Parameters in the Graph.

As the trace runs, individual operators create Nodes in the Graph being traced to record what happens. This code is currently generated per operator in /data/users/zdevito/pytorch/tools/autograd/gen_variable_type.py. It results in code that looks like the following:

```
torch::jit::Node* node = nullptr;
std::shared_ptr<jit::tracer::TracingState> tracer_state;
if (jit::tracer::isTracing()) {
	tracer_state = jit::tracer::getTracingState();
	at::Symbol op_name;
	op_name = jit::Symbol::fromQualString("aten::__ilshift__");
	node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
	jit::tracer::recordSourceLocation(node);
	jit::tracer::addInputs(node, "self", self);
	jit::tracer::addInputs(node, "other", other);
	tracer_state->graph->insertNode(node);

	jit::tracer::setTracingState(nullptr);
}
TypeDefault::__ilshift__(self, other);
if (tracer_state) {
	jit::tracer::setTracingState(std::move(tracer_state));
	jit::tracer::addOutput(node, self);
}
```

The functions `addInputs` and `addOutput` are overloaded to handle the different data types that operators use.

Currently set/getValueTrace only works on Tensors and Futures. Other types are not natively traced. Instead aggregates like tuples or lists are often flattened into tensors at the end of a trace and explicitly constructed from individual tensors at the beginning of this trace.

The tracer has special behavior when tracing calls to other TorchScript functions. This behavior is implemented in the GraphExecutor right before a Graph is about to be run. If tracing is enabled while running the graph, the GraphExecutor will disable tracing, run the graph as normal, and then inline the Graph into the trace. It then hooks up the IValues computed by running the Graph to inlined Graph's out Values in the inlined graph.

*When a trace calls a TorchScript function, that function is preserved as is, meaning that control-flow is preserved.* This makes it possible to "fix" tracing issues by writing the subset of the program that cannot be traced in script and having the trace invoke it.

The resulting Graph created by tracing is installed as the 'forward' method of the Module being created. A Module is produced regardless of whether the thing being traced was a function or a `torch.nn.Module`. In the function case, the Module produced will simply have a single `forward` function, no Parameters, and no sub-Modules.

### Script ###

The script frontend directly converts Python syntax into Modules. Like many compilers this happens in two phases. First, we generate an abstract syntax tree (AST), which is constructed out of Tree objects. The compiler (misnamed, but that is the name of the file) then does semantic analysis on the Tree and lowers it into a Module. We can generate Trees in two ways: (1) using frontend.py, which takes the Python AST and transliterates it into Tree objects, or (2) via the Lexer and Parser which parse python syntax directly.  The Lexer/Parser path may seem redundant but it is crucially important. We need to define builtin functions () when Python is not linked. We allow users to load TorchScript programs directly from strings without Python (). We also use this Python syntax as the serialization format for TorchScript, since it allows us to make changes to our IR without breaking backward compatibility. Furthermore, the Lexer is reused to implement the FunctionSchema parser (), which turns FunctionSchema declarations from strings into FunctionSchema objects.

The following sections look into each the stages in the script frontend in detail.

### Tree ###

[script/tree.h](script/tree.h)

Our frontends produce ASTs in the form of Tree objects. Trees are similar to [s-expressions](https://en.wikipedia.org/wiki/S-expression). Leafs (i.e. Atoms) are always strings. Compound trees have a `kind` (e.g `TK_CONST` or `TK_IDENT` defined in lexer.h) and a list of sub-trees.  For instance, the Tree for `z.sigmoid() - (x + y)` is:

```
 (-
	(+
	  (variable (ident x))
	  (variable (ident y)))
	(apply
	  (.
		(variable (ident z))
		(ident sigmoid))
	  (list)
	  (list))))
```

This is printed in s-expression style with `(kind ...)` representing compound trees and `string_value` representing strings.

We provide utilities to construct, traverse, and print ASTs without a lot of complicated visitor infrastructure and inheritance.

Each tree also has a mandatory SourceRange object that describes the range of text that it came from. These will be used for error reporting in the rest of the code.

### Tree Views ###

[script/tree_views.h](script/tree_views.h)

Trees are easy to construct visualize and traverse, but extracting information from a large compound tree like that of a function definition is unwieldy since it requires numeric indexing. Tree _Views_ are a small layer on top of a tree that make it possible to create and de-structure trees of particular kinds. For example, here is the tree view for the apply node which provides named accessors for its subtrees: the function being called, the inputs, and the attributes (i.e. kwargs):

```
struct Apply : public Expr {
  Expr callee() const {
    return Expr(subtree(0));
  }
  List<Expr> inputs() const {
    return List<Expr>(subtree(1));
  }
  List<Attribute> attributes() const {
    return List<Attribute>(subtree(2));
  ...
};
```

The typical way to traverse a tree is to `switch` on the kind and then construct the appropriate Treeview:

```
switch (tree.kind()) {
  case TK_VAR:
  	auto var = Var(tree); // construct tree-view
	return environment_stack->getSugaredVar(var.name());
  case '.': {
	auto select = Select(tree); // construct tree-view
	auto sv = emitSugaredExpr(select.value(), 1);
	return sv->attr(select.range(), method, select.selector().name());
  }
  case TK_APPLY: {
	auto apply = Apply(tree); // construct tree-view
	return emitApplyExpr(apply, n_binders);
  } break;

```

### frontend.py ###

[torch/jit/frontend.py](../../jit/frontend.py)

One way we construct Tree objects is directly from Python ASTs. This logic is contained inside frontend.py and is intentionally very minimal. We endeavor to keep most of the JIT code written in C++, because most of the JIT functionality still needs to work without Python installed. So this code simply constructs the Tree, filtering out the AST nodes of Python that we do not support.

### Lexer ###

[script/lexer.h](script/lexer.h)

When loading TorchScript code directly from a string, we using a standard Lexer/Parser combo. The Lexer takes an initial string and then exposes a stateful interface for walking the Tokens of the string, providing a standard set of functions:

* `next()` advances the lexer, returning the current token
* `cur()` provides the current token
* `lookahead()` provides the token coming after the current token
* `nextIf(int token_kind)` advances the token if it matches token kind.

Similar to Python, the Lexer handles the white-space sensitive nature of Python blocks. The Tokens `TK_INDENT`, `TK_DEDENT`, and `TK_NEWLINE` are injected into the token stream when code first becomes indented, when it dedents, and at the end of a statement. For instance for this stream:

```
if
  .
  .
```

We would get a token stream `TK_IF TK_NEWLINE TK_INDENT . TK_NEWLINE . TK_NEWLINE TK_DEDENT`. Unmatched opening brackets disabled the injection of these tokens. The result is that the Parser can simply treat `TK_IDENT`, `TK_DEDENT` and `TK_NEWLINE` like C's `{`, `}`, and `;`.

### Tokens ###

[script/lexer.h](script/lexer.h)

Tokens are either keywords (`def`), operators (`+`), literals (`3.4`), or identifiers (`foo`). A token_kind identifies what it is and is the exact same type as the `kind` of a Tree. For single-character Tokens (e.g. `+`), the kind is the same as the character, enable statements like:

```
if (lexer.nextIf('+')) {
	// handle + ...
}
```

Multi-character token kinds are defined in a list, `TC_FORALL_TOKEN_KINDS`. Tokens also have a `text()` field that records the actual string producing the token and is used by identifiers and literals to construct the actual values (e.g. the numeric value of a floating point literal).

### Parser ###

[script/parser.h](script/parser.h)

The Parser uses the Lexer to build a the AST for function definitions. `parseFunction` is the entrypoint for parsing a single `def ...` and will return a `Def` tree view.

The Parser is written as a [top-down precedence parser](https://eli.thegreenplace.net/2010/01/02/top-down-operator-precedence-parsing), or "Pratt" parser.  They are simpler and easier to understand than typical parser generators, while still being flexible enough to parse programming languages. For the most part parsing is done by recursive decent. To resolve operator precedence issues, the function to parse an expression is augmented with a precedent _p_ such that calling the function means _parse an expression whose operators all have precedence higher than p_.

### Compiler ###

[script/compiler.h](script/compiler.h)

The file compiler.cpp translates Trees into Modules. Its name is slightly ambiguous because there are other compiler-like things in the system such as the FusionCompiler. The main entrypoint is `defineMethodsInModule` which takes a list of Def Tree Views representing function definitions and adds them as Methods to the module. During the lowering processing _semantic checking_ occurs. The compiler checks that all used variables are defined (sometimes called scope checking), and that all values have compatible types (type-checking). During this process it also emits the graph nodes corresponding to each statment in the Tree and generates a FunctionSchema for the whole definition.

A few helper objects exist in the lowering process.  SugaredValues are special values that represent objects that can appear during compilation but that are not first class values. For instance, in TorchScript methods `self` refers to the module, and `self.weight` refers to a Parameter of the module. Neither are first-class Types and have no corresponding Value in a graph. Resolver objects are std::functions that resolve externally-defined variables to SugaredValues. For instance, the identifier `torch` which contains most of our built-in ops is looked up through Resolver objects which interact with the python state of the program.

The Environment tracks the mapping between variable names and the SugaredValues they refer to.

### SugaredValue ###

[script/sugared_value.h](script/sugared_value.h)

SugaredValues are how the compiler represents non-first class values during Graph creation. These values are things like the Module or a python function call that do not have corresponding Value objects in the Graph. The compiler _desugars_ the SugaredValue objects to instructions in the graph based on how they are used.  The SugaredValue class has a number of abstract methods on it such as `attr` or `call`. Consider the expression `self.foo`. For methods, `self` will resolve to a special SugaredValue subclass,  ModuleValue. When the compiler sees `self.foo`, it will then call the ModuleValue function `sv.attr("foo")`, asking the ModuleValue how it should desugar itself when the attribute `"foo"` accessed. If `foo` is a parameter, it would then ensure that the parameter was added to the Method being compiled, and return a `SimpleValue` sugared value that contains the Value object representing the parameter as an input. If `foo` were a sub-Module then it would return another SugaredModule. The method `call` is invoked when the compiler sees the value used as a function call.

SugaredValues are also how we interact with Python runtime during the compilation process. For instance, `math.pi` is resolved to 3.1415... by first resolving `math` to a SugaredValue representing accesses to Python modules (PythonModuleValue) whose `attr` function turns python numbers into  `prim::Constant` Nodes in the graph.

Finally, normal Values are also represented by the SimpleValue SugaredValue in places where it is valid either a SugaredValue or a normal Value to appear.

### Resolver ###

[script/compiler.h](script/compiler.h)

Any undefined variable during compilation is resolved with a call to an externally-provided Resolver. When called from Python (e.g `torch.jit.script`) this resolver interacts with the Python runtime via pybind11 to resolve symbols like `torch` and `math` to their Python equivalents.

*The combination of SugaredValue and Resolver decouples the implementation of the compiler from the pybind11 Python bindings that enable its interaction with the Python state.*

This makes it possible to use most of the compiler functionality when python is not present.

### Environment ###

[script/compiler.cpp](script/compiler.cpp)

The Environment object tracks the assignment of variable names to SugaredValues during compilation. It is local to the compiler file. A stack of environments exist, with a new environment being created for sub-blocks introduced by control flow. The Environment also handles turning the AST representation into SSA-form by tracking which variables were modified inside a sub-block and inserting the correct inputs/outputs to the Blocks of if-statements and loops.

### Python-Compiler Interaction ###

[script/init.cpp](script/init.cpp)

A set of special SugaredValues are used to translate between objects in the Python environment and Values in the Graph during the compilation process. The entry-point for this behavior is `toSugaredValue(py::object obj, ...)` which takes a pybind11 Python value and figures out how to turn it into an appropriate SugaredValue. Values exist to represent Python functions, Python modules, and ScriptModule objects.


Executing Programs
------------------

TODO: Graph executor, optimization passes, specialization, differentiation, symbolic autograd,
fusion, operators


Saving Programs
---------------

TODO: python_print, serialization format

Python Bindings
---------------

TODO: Script Module, torch.jit.trace, __constant__ handling, weak script modules


# Older Readme, incorporate content into the correct sections above.

The jit directory contains infrastructure for a just-in-time compiler for
PyTorch and associated 'script' subset of python it can execute directly.

The JIT compiler has several phases.

1. Parsing - An AST (defined in tree_views.h) is generated either by parsing a string of python-like code (`jit/script/parser.h`) or by translation from the Python AST (`jit/frontend.py`). This phase only checks for syntactic correctness and for use of the syntactic subset of python that
the script supports.

2. Semantic Checking/Specialization - We lower the AST into an IR Graph object. In this
phase we check that variables are in scope and resolve any free variables to python objects.
When we find free variables that are python objects, or references to non-first-class values
such as modules, we temporarily represent them as `SugaredValue` objects. This phase then
de-sugars these values by e.g. inserting a `PythonOp` into the graph to call a python function.

3. Optimizations - A `GraphExecutor` works on an initial `Graph` object, performing optimizations,
possibly differentiating it, and possibly specializing it to a particular size.

4. Translation to Instructions - to execute a graph, it is lowered by the interpreter
into a linear list of Instruction objects.

5. Execution - the interpreter reads the instruction stream, executing ATen operations and
any generated code fragments.

## Well-known functions

Ordinarily, when defining a compiler you want the set of functions to be user
extensible; e.g., a user can add to the set of defined functions by defining an
appropriate autograd Function.  However, there are some functions where we want
to make assumptions about their semantics, because we are going to write
optimizations over them or insert them into the program.  Such functions are
"well-known" functions, because the JIT compiler knows about them, and a user
implementation must abide by the contract (sometimes implicitly) specified by
the compiler.

A well-known function is usually implemented in several parts:

* First, we pre-intern the string (`interned_strings.h`) that identifies
  the node.  This allows us to more conveniently refer to these operators
  without having to first do a lookup through the intern table.

* If we generate this operator during optimizations, we will often have
  a helper function in `Graph` (`ir.h`) for creating the operator.  This is
  the easiest way to find out, in code, what attributes we assume for an
  operator.

* There is a runtime interpretation of the operator in
  `torch/csrc/autograd/functions/interpreter.cpp`, which specifies how we
  actually interpret programs that contain such an operator.

So, whence the specifications!  For the most part, we are following
the [ONNX operator specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md)
to determine the semantics of our operators.  However, there are a few
other well-known functions which are specific to PyTorch.

* **FusionGroup**

  A fusion group takes some number of input tensors, applies a graph `Subgraph`
  to them, producing the returned tensors of the subgraph.  Operationally,
  operators inside a FusionGroup are fused into a single kernel, so that their
  intermediate results are never materialized.  Not all operators support
  fusion:

  * **attribute**:
    <dl>
      <dt>Subgraph</dt>
      <dd>The graph of fused operators.  Its inputs and outputs should match
      the number of inputs and outputs to the FusionGroup operator.</dd>
    </dl>
  * **input**: 1 - ∞ (same as inputs of Subgraph)
  * **output**: 1 - ∞ (same as outputs of Subgraph)
