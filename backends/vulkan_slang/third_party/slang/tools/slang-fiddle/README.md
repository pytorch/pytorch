Fiddle
======
> "Fiddle writes code, so you don't have to"

The `slang-fiddle` tool bundles together a few different pieces of functionality related to source-code generation, in a way that is intended to fit the workflow needed for building the Slang compiler.

Invoking Fiddle
---------------

Fiddle gets invoked from the command line with a command line like:

```
slang-fiddle -i source/ -o generated/ a.cpp b.h c.h d.cpp ...
```

This will run Fiddle on each of the files `a.cpp`, `b.h`, and so forth.

The `-i` option gives a prefix (here `source/`) that gets prepended on each of the file names to produce the input path for that file, and the `-o` options gives a prefix (here `generated/`) that gets prepended on each of the file names as part of the output path for that file. The output path for each file is *also* suffixed with `.fiddle`. Thus, the above command line will process `source/a.cpp` to generate `generated/a.cpp.fiddle`, and so on.

In addition to generating an output file corresponding to each input, Fiddle may *also* overwrite the input file, in order to inject a little bit of code (more on that later).

For the most part, you shouldn't have to think about how Fiddle gets invoked, because the build system should be doing it for you.

Overview of Steps
-----------------

Fiddle does its work across a few different steps:

* First, each of the input files is read in and parsed for two kinds of constructs:

    * C++ declarations that have been marked with the `FIDDLE()` macro are *scraped* to extract (minimal) information about them.

    * Lines that have `FIDDLE TEMPLATE` markers on them are used to identify the *text templates* for which output should be generated.

* Second, a minimal amount of semantic checking is performed on the C++ declarations that were scraped (basically just building an inheritance hierarchy). This step happens after *all* of the input files have been scraped, so it can detect relationships between types in different files.

* Third, code is generated into the output file, for each of the two kinds of constructs:

  * For each `FIDDLE()` macro invocation site, a specialized macro definition is generated that provides the output to match that site

  * For each text template, the template is evaluated (using Lua) to generate the desired code.

* Fourth, each input file that contained any text templates is carefully overwritten so that the template definition site will `#include` the corresponding generated code from the output file.

The remaining sections here will go into more detail about the two kinds of constructs.

Scraping with `FIDDLE()`
------------------------

During the scraping step, Fiddle will run the Slang lexer on each of the input files (relying on the fact that the Slang lexer can handle a reasonable subset of C++), and then scan through the token stream produced by the lexer looking for invocations of the `FIDDLE()` macro.

Putting `FIDDLE()` in front of an ordinary C++ `class`, `struct`, or `namespace` declaration tells Fiddle that it should include that C++ construct in the model of the program that it scrapes. For example, given this input:

```
FIDDLE()
namespace MyProgram
{
    FIDDLE()
    struct A { /* ... */ };

    struct B { /* ... */ };
}
```

Fiddle will include the `MyProgram` namespace and the `MyProgram::A` type in its model, but will *not* include `B`.

> Note that because the scraping step does *not* try to understand anything in the C++ code that is not preceded by an invocation of `FIDDLE()`.

A programmer can place Fiddle-specific modifiers inside the `FIDDLE()` invocation before a type, to apply those modifiers to the model of that type:

```
FIDDLE(abstract)
class Thing { /* ... */ };

FIDDLE()
class ConcreteThing : public Thing { /* ... */ }
```

One important constraint is that any `struct` or `class` type marked with `FIDDLE()` *must* have an invocation of the form `FIDDLE(...)` (that is, `FIDDLE` applied to an actual ellipsis `...`) as the first item after the opening curly brace for its body:

```
FIDDLE()
struct A
{
    FIDDLE(...)

    /* rest of declaration */
};
```

Fiddle will generate macro definitions that each of the different `FIDDLE()` invocations will expand to (using `__LINE__` so that each can expand differently), and the `FIDDLE(...)` within a type is used to inject additional members into that type.

> The ability to inject additional declarations in a type is currently mostly used to add all the boilerplate that is required by AST node classes.

Fields within a type can also be marked with `FIDDLE()` so that they will be available in the data model that Fiddle builds while scraping.
Note again that Fiddle will *ignore* any fields not marked with `FIDDLE()`, so be sure to mark all the fields that are relevant to your purpose.

In order for Fiddle to provide the macros that each `FIDDLE()` invocation expands into, any input file that includes invocations of `FIDDLE()` *must* also `#include` the corresponding generated file.
For example, the input file `a.cpp` should `#include "a.cpp.fiddle"`.
The `#include` of the generated output file should come *after* any other `#include`s, to make sure that any `.h` files that also use `FIDDLE()` don't cause confusion.

Text Templates
--------------

The real meat of what Fiddle can do is around its support for text templates. These allow your C++ code files (or any text files, really) to embed Lua script code that generates additional C++ source.

The start of a text template is identified by a line that contains the exact text `FIDDLE TEMPLATE`.
Every text template must follow this sequence:

* A line containing `FIDDLE TEMPLATE`

* Zero or more lines of template code

* A line containing `FIDDLE OUTPUT`

* Zero or more lines of (generated) output code

* A line containing `FIDDLE END`

Fiddle doesn't care what else is on the three marker lines it uses, so you can construct your code to conveniently place the markers and the template in comments or other code that the C++ compiler won't see.
An idiomatic approach would be something like:

```
// my-class-forward-decls.h
#pragma once

// Generate a bunch of forward declarations:
//
#if 0 // FIDDLE TEMPLATE:
%for _,T in ipairs(MyNamespace.MyClass.subclasses) do
    class $T;
%end
#else // FIDDLE OUTPUT:
#endif
```

For the template part of things, you can write lines of more-or-less ordinary C++ code, interspersed with two kinds of script code:

* Lines in a template where `%` is the first non-whitespace character are assumed to be Lua statements

* Otherwise, any `$` in a line marks the start of a *splice*. The `$` can be followed by a single identifier, or a Lua expression enclosed in `()` (the nested expression *must* have balanced `()`s).

Statement lines (using `%`) are executed for their effect, and don't directly produce output, while splices (using `$`) evaluate a Lua expression and then apply `tostring()` to it to yield text to be spliced in.

Rather than directly writing the generated code for a template back into the input file, Fiddle writes the code for each template out in the generated output file, and then injects a simple `#include` at each text template site that pulls in the corresponding text.
For example, given the input above, the generated output for the template might be:

```
#if 0 // FIDDLE TEMPLATE:
...
#else // FIDDLE OUTPUT:
#define FIDDLE_TEMPLATE_OUTPUT_ID 0
#include "my-class-forward-decls.h.fiddle"
#endif
```
