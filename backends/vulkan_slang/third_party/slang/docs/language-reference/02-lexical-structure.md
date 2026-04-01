> Note: This document is a work in progress. It is both incomplete and, in many cases, inaccurate.

Lexical Structure
=================

Source Units
------------

A _source unit_ comprises a sequence of zero or more _characters_ which for purposes of this document are defined as Unicode scalars (code points).

Encoding
--------

Implementations *may* accept source units stored as files on disk, buffers in memory, or any appropriate implementation-specified means.
When source units are stored as byte sequences, they *should* be encoded using UTF-8.
Implementations *may* support additional implemented-specified encodings.

Whitespace
----------

_Horizontal whitespace_ consists of space (U+0020) and horizontal tab (U+0009).

A _line break_ consists of a line feed (U+000A), carriage return (U+000D) or a carriage return followed by a line feed (U+000D, U+000A).
Line breaks are used as line separators rather than terminators; it is not necessary for a source unit to end with a line break.

Escaped Line Breaks
-------------------

An _escaped line break_ comprises a backslack (`\`, U+005C) follow immediately by a line break.

Comments
--------

A _comment_ is either a line comment or a block comment:

```hlsl
// a line comment
/* a block comment */
```

A _line comment_ comprises two forward slashes (`/`, U+002F) followed by zero or more characters that do not contain a line break.
A line comment extends up to, but does not include, a subsequent line break or the end of the source unit.

A _block comment_ begins with a forward slash (`/`, U+002F) followed by an asterisk (`*`, U+0052). 
A block comment is terminated by the next instance of an asterisk followed by a forward slash (`*/`).
A block comment contains all characters between where it begins and where it terminates, including any line breaks.
Block comments do not nest.
It is an error if a block comment that begins in a source unit is not terminated in that source unit.

Phases
------

Compilation of a source unit proceeds _as if_ the following steps are executed in order:

1. Line numbering (for subsequent diagnostic messages) is noted based on the locations of line breaks

2. Escaped line breaks are eliminated. No new characters are inserted to replace them. Any new escaped line breaks introduced by this step are not eliminated.

3. Each comments is replaced with a single space (U+0020)

4. The source unit is _lexed_ into a sequence of tokens according the lexical grammar in this chapter

5. The lexed sequence of tokens is _preprocessed_ to produce a new sequence of tokens (Chapter 3)

6. Subsequent processing is performed on the preprocessed sequence of tokens

Identifiers
-----------

An _identifier_ begins with an uppercase or lowercase ASCII letter (`A` through `Z`, `a` through `z`), or an underscore (`_`).
After the first character, ASCII digits (`0` through `9`) may also be used in an identifier.

The identifier consistent of a single underscore (`_`) is reserved by the language and must not be used by programs.
Otherwise, there are no fixed keywords or reserved words.
Words that name a built-in language construct can also be used as user-defined identifiers and will shadow the built-in definitions in the scope of their definition.

Literals
--------

### Integer Literals

An _integer literal_ consists of an optional radix specifier followed by digits and an optional suffix.

The _radix specifier_ may be:

* `0x` or `0X` to specify a hexadecimal literal (radix 16)
* `0b` or `0B` to specify a binary literal (radix 2)

When no radix specifier is present a radix of 10 is used.

Octal literals (radix 8) are not supported.
A `0` prefix on an integer literal does *not* specify an octal literal as it does in C.
Implementations *may* warn on integer literals with a `0` prefix in case users expect C behavior.

The _digits_ of an integer literal may include ASCII `0` through `9`.
In the case of a hexadecimal literal, digits may include the letters `A` through `F` (and `a` through `f`) which represent digit values of 10 through 15.
It is an error for an integer literal to include a digit with a value greater than or equal to the radix.
The digits of an integer literal may also include underscore (`_`) characters, which are ignored and have no semantic impact.

The _suffix_ on an integer literal may be used to indicate the desired type of the literal:

* A `u` suffix indicates the `uint` type
* An `l` or `ll` suffix indicates the `int64_t` type
* A `ul` or `ull` suffix indicates the `uint64_t` type

### Floating-Point Literals

> Note: This section is not yet complete.

### String Literals

> Note: This section is not yet complete.

### Character Literals

> Note: This section is not yet complete.

Operators and Punctuation
-------------------------

> Note: This section is not yet complete.
