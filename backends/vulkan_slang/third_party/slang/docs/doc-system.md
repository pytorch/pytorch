Slang Doc System
================

Slang contains a rudimentary documentation generation system. The mechanism used to mark up source is similar to [doxygen](https://www.doxygen.nl/manual/docblocks.html). Namely

```
/**
 ... text ... (JavaDoc style)
 */
void someFunctionA() {}

/*!
 .. text .. (QT style)
 another line
 */
void someFunctionB() {}

/// ... text ... (Multi line)
/// another line
void someFunctionC() {}

//!... text ...  (QT Multi line)
//! another line
void someFunctionD() {}

```

All of the above examples will add the documentation for the declaration that appears after them. Also note that this slightly diverges from doxygen in that an empty line before and after in a multi line comment is *not* required.

We can also document the parameters to a function similarly

```
/// My function
void myFunction(
    /// The A parameter
    int a,
    /// The B parameter
    int b);
```

If you just need a single line comment to describe something, you can place the documentation after the parameter as in

```

/// My function
void myFunction(    int a,      //< The A parameter
                    int b)      //< The B parameter
{}
```

This same mechanisms work for other kinds of common situations such as with enums

```
/// An enum
enum AnEnum
{
    Value, ///< A value
    /// Another value
    /// With a multi-line comment
    AnotherValue,
};
```

Like `doxygen` we can also have multi line comments after a declaration for example

```
/// An enum
enum AnEnum
{
    Value, ///< A value
           ///< Some more information about `Value`

    /// Another value
    /// With a multi-line comment
    AnotherValue,
};
```




To actually get Slang to output documentation you can use the `-doc` option from the `slangc` command line, or pass it in as parameter to `spProcessCommandLineArguments` or `processCommandLineArguments`. The documentation is currently output by default to the same `ISlangWriter` stream as diagnostics. So for `slangc` this will generally mean the terminal/stderr.

Currently the Slang doc system does not support any of the 'advanced' doxygen documentation features. If you add documentation to a declaration it is expected to be in [markdown](https://guides.github.com/features/mastering-markdown/).

Currently the only documentation style supported is a single file 'markdown' output. Future versions will support splitting into multiple files and linking between them. Also future versions may also support other documentation formats/standards.

It is possible to generate documentation for the slang core module. This can be achieved with `slangc` via

```
slangc -doc -compile-core-module
```

The documentation will be written to a file `stdlib-doc.md`.

It should be noted that it is not necessary to add markup to a declaration for the documentation system to output documentation for it. Without the markup the documentation is going to be very limited, in essence saying the declaration exists and other aspects that are available from the source. This may not be very helpful. For this reason and other reasons there is a mechanism to control the visibility of items in your source.

There are 3 visibility levels 'public', 'internal' and 'hidden'/'private'. There is a special comment that controls visibility for subsequent lines. The special comment starts with `//@` as shown below.

```
//@ public:

void thisFunctionAppearsInDocs() {}

//@ internal:

void thisFunctionCouldAppearInInternalDocs() {}

//@ hidden:

void thisFunctionWillNotAppearInDocs() {}
```


