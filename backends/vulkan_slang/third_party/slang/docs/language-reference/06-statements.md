> Note: This document is a work in progress. It is both incomplete and, in many cases, inaccurate.

Statements
==========

Statements are used to define the bodies of functions and determine order of evaluation and control flow for an entire program.
Statements are distinct from expressions in that statements do not yield results and do not have types.

This section lists the kinds of statements supported by Slang.

Expression Statement
--------------------

An expression statement consists of an expression followed by a semicolon:

```hlsl
doSomething();
a[10] = b + 1;
```

An implementation may warn on an expression statement that has to effect on the results of execution.

Declaration Statement
---------------------

A declaration may be used as a statement:

```hlsl
let x = 10;
var y = x + 1;
int z = y - x;
```

> Note: Currently only variable declarations are allowed in statement contexts, but other kinds of declarations may be enabled in the future.

Block Statement
---------------

A block statement consists of zero or more statements wrapped in curly braces `{}`:

```hlsl
{
	int x = 10;
	doSomething(x);
}
```

A block statement provides local scoping to declarations.
Declarations in a block are visible to later statements in the same block, but not to statements or expressions outside of the block.

Empty Statement
---------------

A single semicolon (`;`) may be used as an empty statement equivalent to an empty block statement `{}`.

Conditional Statements
----------------------

### If Statement

An _if statement_ consists of the `if` keyword and a conditional expression in parentheses, followed by a statement to execute if the condition is true:

```hlsl
if(somethingShouldHappen)
    doSomething();
```

An if statement may optionally include an _else clause_ consisting of the keyword `else` followed by a statement to execute if the condition is false:

```hlsl
if(somethingShouldHappen)
 	doSomething();
else
	doNothing();
```

### Switch Statement

A _switch statement_ consists of the `switch` keyword followed by an expression wrapped in parentheses and a _body statement_:

```hlsl
switch(someValue)
{
	...
}
```

The body of a switch statement must be a block statement, and its body must consist of switch case clauses.
A _switch case clause_ consists of one or more case labels or default labels, followed by one or more statements:

```hlsl
// this is a switch case clause
case 0:
case 1:
    doBasicThing();
    break;

// this is another switch case clause
default:
    doAnotherThing();
    break;
```

A _case label_ consists of the keyword `case` followed by an expressions and a colon (`:`).
The expression must evaluate to a compile-time constant integer.

A _default label_ consists of the keyword `default` followed by a colon (`:`).

It is an error for a case label or default label to appear anywhere other than the body of a `switch` statement.
It is an error for a statement to appear inside the body of a `switch` statement that is no part of a switch case clause.

Each switch case clause must exit the `switch` statement via a `break` or other control transfer statement.
"Fall-through" from one switch case clause to another is not allowed.

Loop Statements
---------------

### For Statement

A _for statement_ uses the following form:

```hlsl
for( <initial statement> ; <condition expression> ; <side effect expression> ) <body statement>
```

The _initial statement_ is optional, but may declare a variable whose scope is limited to the for statement.

The _condition expression_ is optional. If present it must be an expression that can be coerced to type `bool`. If absent, a true value is used as the condition.

The _side effect expression_ is optional. If present it will executed for its effects before each testing the condition for every loop iteration after the first.

The _body statement_ is a statement that will be executed for each iteration of the loop.

### While Statement

A _while statement_ uses the following form:

```hlsl
while( <condition expression> ) <body statement>
```

and is equivalent to a `for` loop of the form:

```hlsl
for( ; <condition expression> ; ) <body statement>
```

### Do-While Statement

A _do-while statement_ uses the following form:

```hlsl
do <body statement> while( <condition expression> )
```

and is equivalent to a `for` loop of the form:

```hlsl
for(;;)
{
	<body statement>
	if(<condition expression>) continue; else break;
}
```

Control Transfer Statements
---------------------------

### Break Statement

A `break` statement transfers control to after the end of the closest lexically enclosing switch statement or loop statement:

```hlsl
break;
```

### Continue Statement

A `continue` statement transfers control to the start of the next iteration of a loop statement.
In a for statement with a side effect expression, the side effect expression is evaluated when `continue` is used:

```hlsl
break;
```

### Return Statement

A `return` statement transfers control out of the current function.

In the body of a function with a `void` result type, the `return` keyword may be followed immediately by a semicolon:

```hlsl
return;
```

Otherwise, the `return` keyword must be followed by an expression to use as the value to return to the caller:

```hlsl
return someValue;
```

The value returned must be able to coerce to the result type of the lexically enclosing function.

### Discard Statement

A `discard` statement can only be used in the context of a fragment shader, in which case it causes the current invocation to terminate and the graphics system to discard the corresponding fragment so that it does not get combined with the framebuffer pixel at its coordinates.

Operations with side effects that were executed by the invocation before a `discard` will still be performed and their results will become visible according to the rules of the platform.

Compile-Time For Statement
--------------------------

A _compile-time for statement_ is used as an alternative to preprocessor techniques for loop unrolling.
It looks like:

```hlsl
$for( <name> in Range(<initial-value>, <upper-bound>)) <body statement>
```

The _initial value_ and _upper bound_ expressions must be compile-time constant integers.
The semantics of a compile-time for statement are as if it were expanded into:

```hlsl
{
	let <name> = <initial-value>;
	<body statement>
}
{
	let <name> = <initial-value> + 1;
	<body statement>
}
...
{
	let <name> = <upper-bound> - 1;
	<body statement>
}
```
