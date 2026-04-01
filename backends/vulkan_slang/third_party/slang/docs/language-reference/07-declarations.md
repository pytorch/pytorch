> Note: This document is a work in progress. It is both incomplete and, in many cases, inaccurate.

Declarations
============

Modules
-------

A module consists of one or more source units that are compiled together.
The global declarations in those source units comprise the body of the module.

In general, the order of declarations within a source unit does not matter; declarations can refer to other declarations (of types, functions, variables, etc.) later in the same source unit.
Declarations (other than `import` declarations) may freely be defined in any source unit in a module; declarations in one source unit of a module may freely refer to declarations in other source units.

Imports
-------

An import declaration is introduced with the keyword `import`:

```hlsl
import Shadowing;
```

An import declaration searches for a module matching the name given in the declaration, and brings the declarations in that module into scope in the current source unit.

> Note: an `import` declaration only applies to the scope of the current source unit, and does *not* import the chosen module so that it is visible to other source units of the current module.

The name of the module being imported may use a compound name:

```hlsl
import MyApp.Shadowing;
```

The mechanism used to search for a module is implementation-specific.

> Note: The current Slang implementation searches for a module by translating the specified module name into a file path by:
>
> * Replacing any dot (`.`) separators in a compound name with path separators (e.g., `/`)
>
> * Replacing any underscores (`_`) in the name with hyphens (`-`)
>
> * Appending the extension `.slang`
>
> The implementation then looks for a file matching this path on any of its configured search paths.
> If such a file is found it is loaded as a module comprising a single source unit.

The declarations of an imported module become visible to the current module, but they are not made visible to code that later imports the current module.

> Note: An experimental feature exists for an "exported" import declaration:
>
> ```hlsl
> // inside A.slang
> __exported import Shadowing;
> ```
>
> This example imports the declarations from `Shadowing` into the current module (module `A`),
> and also sets up information so that if other code declares `import A` then it can see
> both the declarations in `A` and those in `Shadowing`.

> Note: Mixing `import` declarations and traditional preprocessor-based (`#include`) modularity
> in a codebase can lead to surprising results.
>
> Some things to be aware of:
>
> * Preprocessor definitions in your module do *not* affect the code of modules you `import`.
>
> * Preprocessor definitions in a module you `import` do *not* affect your code
>
> * The above caveats also apply to "include guards" and `#pragma once`, since they operate at the granularity of a source unit (not across modules)
>
> * If you `import` two modules, and then both `#include`  the same file, then those two modules may end up with duplicate declarations with the same name.
>
> As a general rule, be wary of preprocessor use inside of code meant to be an `import`able module.

Variables
---------

Variables are declared using the keywords `let` and `var`:

```hlsl
let x = 7;
var y = 9.0;
```

A `let` declaration introduces an immutable variable, which may not be assigned to or used as the argument for an `in out` or `out` parameter.
A `var` declaration introduces a mutable variable.

An explicit type may be given for a variable by placing it after the variable name and a colon (`:`):

```hlsl
let x : int = 7;
var y : float = 9.0;
```

If no type is specified for a variable, then a type will be inferred from the initial-value expression.
It is an error to declare a variable that has neither a type specifier or an initial-value expression.
It is an error to declare a variable with `let` without an initial-value expression.

A variable declared with `var` may be declared without an initial-value expression if it has an explicit type specifier:

```
var y : float;
```

In this case the variable is _uninitialized_ at the point of declaration, and must be explicitly initialized by assigning to it.
Code that uses the value of an uninitialized variable may produce arbitrary results, or even exhibit undefined behavior depending on the type of the variable.
Implementations *may* issue an error or warning for code that might make use of an uninitialized variable.

### Traditional Syntax

Variables may also be declared with traditional C-style syntax:

```hlsl
const int x = 7;
float y = 9.0;
```

For traditional variable declarations a type must be specified.

> Note: Slang does not support an `auto` type specifier like C++.

Traditional variable declarations are immutable if they are declared with the `const` modifier, and are otherwise mutable.

### Variables at Global Scope

Variables declared at global scope may be either a global constant, a static global variables, or a global shader parameters.

#### Global Constants

A variable declared at global scope and marked with `static` and `const` is a _global constant_.

A global constant must have an initial-value expression, and that initial-value expression must be a compile-time constant expression.

#### Static Global Variables

A variable declared at global scope and marked with `static` (but not with `const`) is a _static global variable_.

A static global variable provides storage for each invocation executing an entry point.
Assignments to a static global variable from one invocation do not affect the value seen by other invocations.

> Note: the semantics of static global variable are similar to a "thread-local" variable in other programming models.

A static global variable may include an initial-value expression; if an initial-value expression is included it is guaranteed to be evaluated and assigned to the variable before any other expression that references the variable is evaluated.
There is no guarantee that the initial-value expression for a static global variable is evaluated before entry point execution begins, or even that the initial-value expression is evaluated at all (in cases where the variable might not be referenced at runtime).

> Note: the above rules mean that an implementation may perform dead code elimination on static global variables, and may choose between eager and lazy initialization of those variables at its discretion.

#### Global Shader Parameters

A variable declared at global scope and not marked with `static` (even if marked with `const`) is a _global shader parameter_.

Global shader parameters are used to pass arguments from application code into invocations of an entry point.
The mechanisms for parameter passing are specific to each target platform.

> Note: Currently only global shader parameters of opaque types or arrays of opaque types are supported.

A global shader parameter may include an initial-value epxression, but such an expression does not affect the semantics of the compiled program.

> Note: Initial-value expressions on global shader parameters are only useful to set up "default values" that can be read via reflection information and used by application code.

### Variables at Function Scope

Variables declared at _function scope_ (in the body of a function, initializer, subscript accessor, etc.) may be either a function-scope constant, function-scope static variable, or a local variable.

#### Function-Scope Constants

A variable declared at function scope and marked with both `static` and `const` is a _function-scope constant_.
Semantically, a function-scope constant behaves like a global constant except that is name is only visible in the local scope.

#### Function-Scope Static Variables

A variable declared at function scope and marked with `static` (but not `const`) is a _function-scope static variable_.
Semantically, a function-scope static variable behaves like a global static variable except that its name is only visible in the local scope.

The initial-value expression for a function-scope static variable may refer to non-static variables in the body of the function.
In these cases initialization of the variable is guaranteed not to occur until at least the first time the function body is evaluated for a given invocation.

#### Local Variables

A variable declared at function scope and not marked with `static` (even if marked with `const`) is a _local variable_.
A local variable has unique storage for each _activation_ of a function by an invocation.
When a function is called recursively, each call produces a distinct activation with its own copies of local variables.

Functions
---------

Functions are declared using the `func` keyword:

```hlsl
func add(x: int, y: float) -> float { return float(x) + y; }
```

Parameters
----------

The parameters of the function are declared as `name: type` pairs.

Parameters may be given a _default value_ by including an initial-value-expression clause:

```hlsl
func add(x: int, y: float = 1.0f) { ... }
```

Parameters may be marked with a _direction_ which affects how data is passed between caller and callee:

```hlsl
func add(x: in out int, y : float) { x += ... }
```

The available directions are:

* `in` (the default) indicates typical pass-by-value (copy-in) semantics. The callee receives a *copy* of the argument passed by the caller.

* `out` indicates copy-out semantics. The callee writes to the parameter and then a copy of that value is assigned to the argument of the caller after the call returns.

* `in out` or `inout` indicates pass-by-value-result (copy-in and copy-out) semantics. The callee receives a copy of the argument passed by the caller, it may manipulate the copy, and then when the call returns the final value is copied back to the argument of the caller.

An implementation may assume that at every call site the arguments for `out` or `in out` parameters never alias.
Under those assumptions, the `out` and `inout` cases may be optimized to use pass-by-reference instead of copy-in and copy-out.

> Note: Applications that rely on the precise order in which write-back for `out` and `in out` parameters is performed are already on shaky semantic ground.

Body
----

The _body_ of a function declaration consists of statements enclosed in curly braces `{}`.

In some cases a function declaration does not include a body, and in these cases the declaration must be terminated with a semicolon (`;`):

```hlsl
func getCount() -> int;
```

> Note: Slang does not require "forward declaration" of functions, although
> forward declarations are supported as a compatibility feature.
>
> The only place where a function declaration without a definition should be
> required is in the body of an `interface` declaration.


The result type of a function mayb be specified after the parameter list using a _result type clause_ consisting of an arrow (`->`) followed by a type.
If the function result type is `void`, the result type clause may be elided:

```hlsl
func modify(x: in out int) { x++; }
```


### Traditional Syntax

Functions can also be declared with traditional C-style syntax:

```hlsl
float add(int x, float y) { return float(x) + y; }

void modify(in out int x) { x ++; }
```

> Note: Currently traditional syntax must be used for shader entry point functions,
> because only the traditional syntax currently supports attaching semantics to
> parameters.

### Entry Points

An _entry point_ is a function that will be used as the starting point of execution for one or more invocations of a shader.



Structure Types
---------------

Structure types are declared using the `struct` keyword:

```hlsl
struct Person
{
    var age : int;
    float height;

    int getAge() { return age; }
    func getHeight() -> float { return this.height; }
    static func getPopulation() -> int { ... }
}
```

The body of a structure type declaration may include variable, type, function, and initializer declarations.

### Fields

Variable declarations in the body of a structure type declaration are also referred to as _fields_.

A field that is marked `static` is shared between all instances of the type, and is semantically like a global variable marked `static`.

A non-`static` field is also called an _instance field_.

### Methods

Function declarations in the body of a structure type declaration are also referred to as _methods_.

A method declaration may be marked `static`.
A `static` method must be invoked on the type itself (e.g., `Person.getPopulation()`).

A non-`static` method is also referred to as an _instance method_.
Instance methods must be invoked on an instance of the type (e.g., `somePerson.getAge()`).
The body of an instance method has access to an implicit `this` parameter which refers to the instance on which the method was invoked.

By default the `this` parameter of an instance method acts as an immutable variable.
An instance method with the `[mutating]` attribute receives a mutable `this` parameter, and can only be invoked on a mutable value of the structure type.

### Inheritance

A structure type declaration may include an _inheritance clause_ that consists of a colon (`:`) followed by a comma-separated list of types that the structure type inherits from:

```
struct Person : IHasAge, IHasName
{ .... }
```

When a structure type declares that it inherits from an interface, the programmer asserts that the structure type implements the required members of the interface.

### Syntax Details

A structure declaration does *not* need to be terminated with a semicolon:

```hlsl
// A terminating semicolon is allowed
struct Stuff { ... };

// The semicolon is not required
struct Things { ... }
```

When a structure declarations ends without a semicolon, the closing curly brace (`}`) must be the last non-comment, non-whitespace token on its line.

For compatibility with C-style code, a structure type declaration may be used as the type specifier in a traditional-style variable declaration:

```hlsl
struct Association
{
    int from;
    int to;
} associations[] =
{
    { 1, 1 },
    { 2, 4 },
    { 3, 9 },
};
```

If a structure type declaration will be used as part of a variable declaration, then the next token of the variable declaration must appear on the same line as the closing curly brace (`}`) of the structure type declaration.
The whole variable declaration must be terminated with a semicolon (`;`) as normal.


Enumeration Types
-----------------

Enumeration type declarations are introduced with the `enum` keyword:

```hlsl
enum Color
{
    Red,
    Green = 3,
    Blue,
}
```

### Cases

The body of an enumeration type declaration consists of a comma-separated list of case declarations.
An optional trailing comma may terminate the lis of cases.

A _case declaration_ consists of the name of the case, along with an optional initial-value expression that specifies the _tag value_ for that case.
If the first case declaration in the body elides an initial-value expression, the value `0` is used for the tag value.
If any other case declaration elides an initial-value expressions, its tag value is one greater than the tag value of the immediately preceding case declaration.

An enumeration case is referred to as if it were a `static` member of the enumeration type (e.g., `Color.Red`).

### Inheritance

An enumeration type declaration may include an inheritance clause:

```hlsl
enum Color : uint
{ ... }
```

The inheritance clause of an enumeration declaration may currently only be used to specify a single type to be used as the _tag type_ of the enumeration type.
The tag type of an enumeration must be a built-in scalar integer type.
The tag value of each enumeration case will be a value of the tag type.

If no explicit tag type is specified, the type `int` is used instead.

> Note: The current Slang implementation has bugs that prevent explicit tag types from working correctly.

### Conversions

A value of an enumeration type can be implicitly converted to a value of its tag type:

```hlsl
int r = Color.Red;
```

Values of the tag type can be explicitly converted to the enumeration type:

```hlsl
Color red = Color(r);
```

Type Aliases
------------

A type alias is declared using the `typealias` keyword:

```hlsl
typealias Height  = int;
```

A type alias defines a name that will be equivalent to the type to the right of `=`.

### Traditional Syntax

Type aliases can also be declared with traditional C-style syntax:

```hlsl
typedef int Height;
```

Constant Buffers and Texture Buffers
------------------------------------

As a compatibility feature, the `cbuffer` and `tbuffer` keywords can be used to introduce variable declarations.

A declaration of the form:

```hlsl
cbuffer Name
{
    F field;
    // ...
}
```

is equivalent to a declaration of the form:

```hlsl
struct AnonType
{
    F field;
    // ...
}
__transparent ConstantBuffer<AnonType> anonVar;
```

In this expansion, `AnonType` and `anonVar` are fresh names generated for the expansion that cannot collide with any name in user code, and the modifier `__transparent` makes it so that an unqualified reference to `field` can implicitly resolve to `anonVar.field`.

The keyword `tbuffer` uses an equivalent expansion, but with `TextureBuffer<T>` used instead of `ConstantBuffer<T>`.

Interfaces
----------

An interface is declared using the `interface` keyword:

```hlsl
interface IRandom
{
    uint next();
}
```

The body of an interface declaration may contain function, initializer, subscript, and associated type declarations.
Each declaration in the body of an interface introduces a _requirement_ of the interface.
Types that declare conformance to the interface must provide matching implementations of the requirements.

Functions, initializers, and subscripts declared inside an interface must not have bodies; default implementations of interface requirements are not currently supported.

An interface declaration may have an inheritance clause:

```hlsl
interface IBase
{
    int getBase();
}

interface IDerived : IBase
{
    int getDerived();
}
```

The inheritance clause for an interface must only list other interfaces.
If an interface `I` lists another interface `J` in its inheritance clause, then `J` is a _base interface_ of `I`.
In order to conform to `I`, a type must also conform to `J`.

Associated Types
----------------

An associated type declaration is introduced with `associatedtype`:

```hlsl
associatedtype Iterator;
```

An associated type declaration introduces a type into the signature of an interface, without specifying the exact concrete type to use.
An associated type is an interface requirement, and different implementations of an interface may provide different types that satisfy the same associated type interface requirement:

```
interface IContainer
{
    associatedtype Iterator;
    ...
}

struct MyArray : IContainer
{
    typealias Iterator = Int;
    ...
}

struct MyLinkedList : IContainer
{
    struct Iterator { ... }
    ...
}
```

It is an error to declare an associated type anywhere other than the body of an interface declaration.

An associated type declaration may have an inheritance clause.
The inheritance clause of an associated type may only list interfaces; these are the _required interfaces_ for the associated type.
A concrete type that is used to satisfy an associated type requirement must conform to all of the required interfaces of the associated type.

Initializers
------------

An initializer declaration is introduced with the `__init` keyword:

```hlsl
struct MyVector
{
    float x, float y;

    __init(float s)
    {
        x = s;
        y = s;
    }
}
```

> Note: Initializer declarations are a non-finalized and unstable feature, as indicated by the double-underscore (`__`) prefix on the keyword.
> Arbitrary changes to the syntax and semantics of initializers may be introduced in future versions of Slang.

An initializer declaration may only appear in the body of an interface or a structure type.
An initializer defines a method for initializing an instance of the enclosing type.

> Note: A C++ programmer might think of an initializer declaration as similar to a C++ _constructor_.

An initializer has a parameter list and body just like a function declaration.
An initializer must not include a result type clause; the result type of an initializer is always the enclosing type.

An initializer is invoked by calling the enclosing type as if it were a function.
E.g., in the example above, the initializer in `MyVector` can be invoked as `MyVector(1.0f)`.


An initializer has access to an implicit `this` variable that is the instance being initialized; an initializer must not be marked `static`.
The `this` variable of an initializer is always mutable; an initializer need not, and must not, be marked `[mutating]`.

> Note: Slang currently does not enforce that a type with an initializer can only be initialized using its initializers.
> It is possible for user code to declare a variable of type `MyVector` above, and explicitly write to the `x` and `y` fields to initialize it.
> A future version of the language may close up this loophole.

> Note: Slang does not provide any equivalent to C++ _destructors_ which run automatically when an instance goes out of scope.

Subscripts
----------

A subscript declaration is introduced with the `__subscript` keyword:

```hlsl
struct MyVector
{
    ...

    __subscript(int index) -> float
    {
        get { return index == 0 ? x : y; }
    }
}
```

> Note: subscript declarations are a non-finalized and unstable feature, as indicated by the double-underscore (`__`) prefix on the keyword.
> Arbitrary changes to the syntax and semantics of subscript declarations may be introduced in future versions of Slang.

A subscript declaration introduces a way for a user-defined type to support subscripting with the `[]` braces:

```hlsl
MyVector v = ...;
float f = v[0];
```

A subscript declaration lists one or more parameters inside parentheses, followed by a result type clause starting with `->`.
The result type clause of a subscript declaration cannot be elided.

The body of a subscript declaration consists of _accessor declarations_.
Currently only `get` accessor declarations are supported for user code.

A `get` accessor declaration introduces a _getter_ for the subscript.
The body of a getter is a code block like a function body, and must return the appropriate value for a subcript operation.
The body of a getter can access the parameters of the enclosing subscript, as a well as an implicit `this` parameter of the type that encloses the accessor.
The `this` parameter of a getter is immutable; `[mutating]` getters are not currently supported.

Extensions
----------

An extension declaration is introduced with the `extension` keyword:

```hlsl
extension MyVector
{
    float getLength() { return sqrt(x*x + y*y); }
    static int getDimensionality() { return 2; }
}
```

An extension declaration adds behavior to an existing type.
In the example above, the `MyVector` type is extended with an instance method `getLength()`, and a static method `getDimensionality()`.

An extension declaration names the type being extended after the `extension` keyword.
The body of an extension declaration may include type declarations, functions, initializers, and subscripts.

> Note: The body of an extension may *not* include variable declarations.
> An extension cannot introduce members that would change the in-memory layout of the type being extended.

The members of an extension are accessed through the type that is being extended.
For example, for the above extension of `MyVector`, the introduced methods are accessed as follows:

```hlsl
MyVector v = ...;

float f = v.getLength();
int n = MyVector.getDimensionality();
```

An extension declaration need not be placed in the same module as the type being extended; it is possible to extend a type from third-party or standard module code.
The members of an extension are only visible inside of modules that `import` the module declaring the extension;
extension members are *not* automatically visible wherever the type being extended is visible.

An extension declaration may include an inheritance clause:

```hlsl
extension MyVector : IPrintable
{
    ...
}
```

The inheritance clause of an extension declaration may only include interfaces.
When an extension declaration lists an interface in its inheritance clause, it asserts that the extension introduces a new conformance, such that the type being extended now conforms to the given interface.
The extension must ensure that the type being extended satisfies all the requirements of the interface.
Interface requirements may be satisfied by the members of the extension, members of the original type, or members introduced through other extensions visible at the point where the conformance was declared.

It is an error for overlapping conformances (that is, of the same type to the same interface) to be visible at the same point.
This includes cases where two extensions declare the same conformance, as well as those where the original type and an extension both declare the same conformance.
The conflicting conformances may come from the same module or difference modules.

In order to avoid problems with conflicting conformances, when a module `M` introduces a conformance of type `T` to interface `I`, one of the following should be true:

* the type `T` is declared in module `M`, or
* the type `I` is declared in module `M`

Any conformance that does not follow these rules (that is, where both `T` and `I` are imported into module `M`) is called a _retroactive_ conformance, and there is no way to guarantee that another module `N` will not introduce the same conformance.
The runtime behavior of programs that include overlapping retroactive conformances is currently undefined.

Currently, extension declarations can only apply to structure types; extensions cannot apply to enumeration types or interfaces.

Generics
--------

Many kinds of declarations can be made _generic_: structure types, interfaces, extensions, functions, initializers, and subscripts.

A generic declaration introduces a _generic parameter list_ enclosed in angle brackets `<>`:

```hlsl
T myFunction<T>(T left, T right, bool condition)
{
    return condition ? left : right;
}
```

### Generic Parameters

A generic parameter list can include one or more parameters separated by commas.
The allowed forms for generic parameters are:

* A single identifier like `T` is used to declare a _generic type parameter_ with no constraints.

* A clause like `T : IFoo` is used to introduce a generic type parameter `T` where the parameter is _constrained_ so that it must conform to the `IFoo` interface.

* A clause like `let N : int` is used to introduce a generic value parameter `N`, which takes on values of type `int`.

> Note: The syntax for generic value parameters is provisional and subject to possible change in the future.

Generic parameters may declare a default value with `=`:

```hlsl
T anotherFunction<T = float, let N : int = 4>(vector<T,N> v);
```

For generic type parameters, the default value is a type to use if no argument is specified.
For generic value parameters, the default value is a value of the same type to use if no argument is specified.

### Explicit Specialization

A generic is _specialized_ by applying it to _generic arguments_ listed inside angle brackets `<>`:

```hlsl
anotherFunction<int, 3>
```

Specialization produces a reference to the declaration with all generic parameters bound to concrete arguments.

When specializing a generic, generic type parameters must be matched with type arguments that conform to the constraints on the parameter, if any.
Generic value parameters must be matched with value arguments of the appropriate type, and that are specialization-time constants.

An explicitly specialized function, type, etc. may be used wherever a non-generic function, type, etc. is expected:

```hlsl
int i = anotherFunction<int,3>( int3(99) );
```

### Implicit Specialization

If a generic function/type/etc. is used where a non-generic function/type/etc. is expected, the compiler attempts _implicit specialization_.
Implicit specialization infers generic arguments from the context at the use site, as well as any default values specified for generic parameters.

For example, if a programmer writes:

```hlsl
int i = anotherFunction( int3(99) );
```

The compiler will infer the generic arguments `<int, 3>` from the way that `anotherFunction` was applied to a value of type `int3`.

> Note: Inference for generic arguments currently only takes the types of value arguments into account.
> The expected result type does not currently affect inference.

### Syntax Details

The following examples show how generic declarations of different kinds are written:

```
T genericFunction<T>(T value);
funct genericFunction<T>(value: T) -> T;

__init<T>(T value);

__subscript<T>(T value) -> X { ... }

struct GenericType<T>
{
    T field;
}

interface IGenericInterface<T> : IBase<T>
{
}
```

> Note: Currently there is no user-exposed syntax for writing a generic extension.
