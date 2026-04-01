> Note: This document is a work in progress. It is both incomplete and, in many cases, inaccurate.

Expressions
===========

Expressions are terms that can be _evaluated_ to produce values.
This section provides a list of the kinds of expressions that may be used in a Slang program.

In general, the order of evaluation of a Slang expression proceeds from left to right.
Where specific expressions do not follow this order of evaluation, it will be noted.

Some expressions can yield _l-values_, which allows them to be used on the left-hand-side of assignment, or as arguments for `out` or `in out` parameters.

Literal Expressions
-------------------

Literal expressions are never l-values.

### Integer Literal Expressions

An integer literal expression consists of a single integer literal token:

```hlsl
123
```

An unsuffixed integer literal expression always has type `int`.

### Floating-Point Literal Expressions

A floating-point literal expression consists of a single floating-point literal token:

```hlsl
1.23
```

A unsuffixed floating-point literal expression always has type `float`.

### Boolean Literal Expressions

Boolean literal expressions use the keywords `true` and `false`.

### String Literal Expressions

A string literal expressions consists of one or more string literal tokens in a row:

```hlsl
"This" "is one" "string"
```

Identifier Expression
---------------------

An _identifier expression_ consists of a single identifier:

```hlsl
someName
```

When evaluated, this expression looks up `someName` in the environment of the expression and yields the value of a declaration with a matching name.

An identifier expression is an l-value if the declaration it refers to is mutable.

### Overloading

It is possible for an identifier expression to be _overloaded_, such that it refers to one or more candidate declarations with the same name.
If the expression appears in a context where the correct declaration to use can be disambiguated, then that declaration is used as the result of  the name expression; otherwise use of an overloaded name is an error at the use site.

### Implicit Lookup

It is possible for a name expression to refer to nested declarations in two ways:

* In the body of a method, a reference to `someName` may resolve to `this.someName`, using the implicit `this` parameter of the method

* When a global-scope `cbuffer` or `tbuffer` declaration is used, `someName` may refer to a field declared inside the `cbuffer` or `tbuffer`

Member Expression
-----------------

A _member expression_ consists of a base expression followed by a dot (`.`) and an identifier naming a member to be accessed:

```hlsl
base.m
```

When `base` is a structure type, this expression looks up the field or other member named by `m`.
Just as for an identifier expression, the result of a member expression may be overloaded, and might be disambiguated based on how it is used.

A member expression is an l-value if the base expression is an l-value and the member it refers to is mutable.

### Implicit Dereference

If the base expression of a member reference is a _pointer-like type_ such as `ConstantBuffer<T>`, then a member reference expression will implicitly dereference the base expression to refer to the pointed-to value (e.g., in the case of `ConstantBuffer<T>` this is the buffer contents of type `T`).

### Vector Swizzles

When the base expression of a member expression is of a vector type `vector<T,N>` then a member expression is a _vector swizzle expression_.
The member name must conform to these constraints:

* The member name must comprise between one and four ASCII characters
* The characters must be come either from the set (`x`, `y`, `z`, `w`) or (`r`, `g`, `b`, `a`), corresponding to element indics of (0, 1, 2, 3)
* The element index corresponding to each character must be less than `N`

If the member name of a swizzle consists of a single character, then the expression has type `T` and is equivalent to a subscript expression with the corresponding element index.

If the member name of a swizzle consists of `M` characters, then the result is a `vector<T,M>` built from the elements of the base vector with the corresponding indices.

A vector swizzle expression is an l-value if the base expression was an l-value and the list of indices corresponding to the characters of the member name contains no duplicates.

### Matrix Swizzles

> Note: The Slang implementation currently doesn't support matrix swizzles.

### Static Member Expressions

When the base expression of a member expression is a type instead of a value, the result is a _static member expression_.
A static member expression can refer to a static field or static method of a structure type.
A static member expression can also refer to a case of an enumeration type.

A static member expression (but not a member expression in general) may use the token `::` instead of `.` to separate the base and member name:

```hlsl
// These are equivalent
Color.Red
Color::Red
```

This Expression
---------------

A _this expression_ consists of the keyword `this` and refers to the implicit instance of the enclosing type that is being operated on in instance methods, subscripts, and initializers.

The type of `this` is `This`.

Parenthesized Expression
----------------------

An expression wrapped in parentheses `()` is a _parenthesized expression_ and evaluates to the same value as the wrapped expression.

Call Expression
---------------

A _call expression_ consists of a base expression and a list of argument expressions, separated by commas and enclosed in `()`:

```hlsl
myFunction( 1.0f, 20 )
```

When the base expression (e.g., `myFunction`) is overloaded, a call expression can disambiguate the overloaded expression based on the number and type or arguments present.

The base expression of a call may be a member reference expression:

```hlsl
myObject.myFunc( 1.0f )
```

In this case the base expression of the member reference (e.g., `myObject` in this case) is used as the argument for the implicit `this` parameter of the callee.

### Mutability

If a `[mutating]` instance is being called, the argument for the implicit `this` parameter must be an l-value.

The argument expressions corresponding to any `out` or `in out` parameters of the callee must be l-values.

A call expression is never an l-value.

### Initializer Expressions

When the base expression of a call is a type instead of a value, the expression is an initializer expression:

```hlsl
float2(1.0f, 2.0f)
```

An initializer expression initialized an instance of the specified type using the given arguments.

An initializer expression with only a single argument is treated as a cast expression:

```hlsl
// these are equivalent
int(1.0f)
(int) 1.0f
```

Subscript Expression
--------------------

A _subscript expression_ consists of a base expression and a list of argument expressions, separated by commas and enclosed in `[]`:

```hlsl
myVector[someIndex]
```

A subscript expression invokes one of the subscript declarations in the type of the base expression. Which subscript declaration is invoked is resolved based on the number and types of the arguments.

A subscript expression is an l-value if the base expression is an l-value and if the subscript declaration it refers to has a setter or by-reference accessor.

Subscripts may be formed on the built-in vector, matrix, and array types.


Initializer List Expression
---------------------------

An _initializer list expression_ comprises zero or more expressions, separated by commas, enclosed in `{}`:

```
{ 1, "hello", 2.0f }
```

An initialier list expression may only be used directly as the initial-value expression of a variable or parameter declaration; initializer lists are not allowed as arbitrary sub-expressions.

> Note: This section will need to be updated with the detailed rules for how expressions in the initializer list are used to initialize values of each kind of type.

Cast Expression
---------------

A _cast expression_ attempt to coerce a single value (the base expression) to a desired type (the target type):

```hlsl
(int) 1.0f
```

A cast expression can perform both built-in type conversions and invoke any single-argument initializers of the target type.

### Compatibility Feature

As a compatibility feature for older code, Slang supports using a cast where the base expression is an integer literal zero and the target type is a user-defined structure type:

```hlsl
MyStruct s = (MyStruct) 0;
```

The semantics of such a cast are equivalent to initialization from an empty initializer list:

```hlsl
MyStruct s = {};
```

Assignment Expression
---------------------

An _assignment expression_ consists of a left-hand side expression, an equals sign (`=`), and a right-hand-side expressions:

```hlsl
myVar = someValue
```

The semantics of an assignment expression are to:

* Evaluate the left-hand side to produce an l-value,
* Evaluate the right-hand side to produce a value
* Store the value of the right-hand side to the l-value of the left-hand side
* Yield the l-value of the left-hand-side

Operator Expressions
--------------------

### Prefix Operator Expressions

The following prefix operators are supported:

| Operator 	| Description |
|-----------|-------------|
| `+`		| identity |
| `-`		| arithmetic negation |
| `~` 		| bit-wise Boolean negation |
| `!`		| Boolean negation |
| `++`		| increment in place |
| `--`		| decrement in place |

A prefix operator expression like `+val` is equivalent to a call expression to a function of the matching name `operator+(val)`, except that lookup for the function only considers functions marked with the `__prefix` keyword.

The built-in prefix `++` and `--` operators require that their operand is an l-value, and work as follows:

* Evaluate the operand to produce an l-value
* Read from the l-value to yield an _old value_
* Increment or decrement the value to yield a _new value_
* Write the new value to the l-value
* Yield the new value

### Postfix Operator Expressions

The following postfix operators are supported:

| Operator 	| Description |
|-----------|-------------|
| `++`		| increment in place |
| `--`		| decrement in place |

A postfix operator expression like `val++` is equivalent to a call expression to a function of the matching name `operator++(val)`, except that lookup for the function only considers functions marked with the `__postfix` keyword.

The built-in prefix `++` and `--` operators require that their operand is an l-value, and work as follows:

* Evaluate the operand to produce an l-value
* Read from the l-value to yield an _old value_
* Increment or decrement the value to yield a _new value_
* Write the new value to the l-value
* Yield the old value

### Infix Operator Expressions

The follow infix binary operators are supported:

| Operator 	| Kind        | Description |
|-----------|-------------|-------------|
| `*`		| Multiplicative 	| multiplication |
| `/`		| Multiplicative 	| division |
| `%`		| Multiplicative 	| remainder of division |
| `+`		| Additive 			| addition |
| `-`		| Additive 			| subtraction |
| `<<`		| Shift 			| left shift |
| `>>`		| Shift 			| right shift |
| `<` 		| Relational 		| less than |
| `>`		| Relational 		| greater than |
| `<=`		| Relational 		| less than or equal to |
| `>=`		| Relational 		| greater than or equal to |
| `==`		| Equality 			| equal to |
| `!=`		| Equality 			| not equal to |
| `&`		| BitAnd 			| bitwise and |
| `^`		| BitXor			| bitwise exclusive or |
| `\|`		| BitOr 			| bitwise or |
| `&&`		| And 				| logical and |
| `\|\|`	| Or 				| logical or |
| `+=`		| Assignment  		| compound add/assign |
| `-=`      | Assignment  		| compound subtract/assign |
| `*=`      | Assignment  		| compound multiply/assign |
| `/=`      | Assignment  		| compound divide/assign |
| `%=`      | Assignment  		| compound remainder/assign |
| `<<=`     | Assignment  		| compound left shift/assign |
| `>>=`     | Assignment  		| compound right shift/assign |
| `&=`      | Assignment  		| compound bitwise and/assign |
| `\|=`     | Assignment  		| compound bitwise or/assign |
| `^=`      | Assignment  		| compound bitwise xor/assign |
| `=`       | Assignment  		| assignment |
| `,`		| Sequencing  		| sequence |

With the exception of the assignment operator (`=`), an infix operator expression like `left + right` is equivalent to a call expression to a function of the matching name `operator+(left, right)`.

### Conditional Expression

The conditional operator, `?:`, is used to select between two expressions based on the value of a condition:

```hlsl
useNegative ? -1.0f : 1.0f
```

The condition may be either a single value of type `bool`, or a vector of `bool`.
When a vector of `bool` is used, the two values being selected between must be vectors, and selection is performed component-wise.

> Note: Unlike C, C++, GLSL, and most other C-family languages, Slang currently follows the precedent of HLSL where `?:` does not short-circuit.
>
> This decision may change (for the scalar case) in a future version of the language.
> Programmer are encouraged to write code that does not depend on whether or not `?:` short-circuits.
