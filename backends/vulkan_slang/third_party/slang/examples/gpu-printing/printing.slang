// printing.slang

// This file provides the GPU code for a simple library that
// allows GPU shaders to print values to `stdout`.
//
// The implementation relies on a single buffer that must
// be bound to any shader that uses GPU printing.
//
RWStructuredBuffer<uint> gPrintBuffer;
//
// Encoding
// ========
//
// The print buffer is organized in terms of 32-bit (`uint`) *words*.
//
// The first word in the print buffer is used as an atomic
// counter, and must be initialized to zero before a shader starts.
// By atomically incrementing this counter, GPU threads can allocate
// space for printing commands in the buffer. All printing
// commands are stored after the first word (so, starting at
// an index of 1).
//
// A printing command starts with a single-word header, where
// the high 32 bits specify the *op* for the command, and the
// low 32 bits specify the number of *payload* words in in
// the command. The payload is the words that immediately
// follow the command header.
//
// Note that the header word for a command is *not* included
// in the count of words in the low 16 bits.
//
// The opcode values need to be shared between CPU and GPU
// code, so we use a bit of preprocessor trickery here to
// generate an `enum` type with all the opcodes.
//
enum PrintingOp
{
#define GPU_PRINTING_OP(NAME) NAME ,
#include "gpu-printing-ops.h"
};

// It is critical that when printing something, we allocate
// all the words it requires in the print buffer contiguously.
// For example, if the user writes:
//
//      println("Thread number ", threadID, " has value ", someValue);
//
// It would be very bad if the output from different threads
// got interleaved, such that one cannot determine which value
// goes with which thread.
//
// Allocating individual print *commands* atomically is not necessarily
// enough: instead, we need to allocate the storage for all
// the commands that comprise a `print()` call at once.
//
// The core allocation operation here is `_allocatePrintWords()`

// Allocate space for one or more print commands.
//
uint _allocatePrintWords(uint wordCount)
{
    // We allocate the required number of words with an atomic, and
    // get back the old value of the counter, which tells us the
    // offset at which the words for our printing operation should start.
    //
    uint wordOffset = 0;
    InterlockedAdd(gPrintBuffer[0], wordCount, wordOffset);

    // Because the first word of the buffer is reserved for the counter,
    // and the counter value starts at zero, we need to add one to
    // get to the actual offset for the data to be written.
    //
    return wordOffset + 1;
}

// Java-style `println`
// ====================
//
// We will start by building up a Java-style `println()` function
// that accepts zero or more values to print, and prints them
// atomically (without any other thread being able to interleave
// in the printed output), followed by a newline.
//
// We will define a wrapper around `_allocatePrintWords()`
// that captures the main idiom for `println()`.
//
uint _beginPrintln(uint wordCount)
{
    // The `wordCount` passed in will represent the
    // number of words required for the arguments
    // to `println`, but won't include the terminating
    // newline.
    //
    // Thus we will allocate one extra word to allow
    // us to append a newline to the print command we
    // generate.
    //
    uint wordOffset = _allocatePrintWords(wordCount + 1);
    //
    // We will then initialize the last word of the command
    // that was allocated to a `NewLine` command.
    //
    gPrintBuffer[wordOffset + wordCount] = uint(PrintingOp.NewLine) << 16;
    return wordOffset;
}
//
// With the `_beginPrintLn()` function handling all the heavy-lifting,
// we can define a zero-argument `println()` trivially.
//
void println()
{
    _beginPrintln(0);
}

// We could continue to build a family of overloaded `println()` functions, like:
//
//      void println();
//      void println(int value);
//      void println(float value);
//      void println(uint value);
//      ...
//
// but it should be clear that this approach doesn't scale at all
// to functions with multiple argumenst:
//
//      void println(int a, int b);
//      void println(float a, int b);
//      void println(int a, float b);
//      ...
//
// Using the features of the Slang language, we can build a framework
// for a more scalable solution.
//
// We start by defining an `interface` that captures the essence
// of what a type of printable values needs to support.
//

interface IPrintable
{
    // Every printable value needs to be able to compute the number
    // of words required to write it into the print buffer.
    //
    uint getPrintWordCount();

    // A printable value must also support writing those words into
    // a buffer, once the appropriate offset to write to is known.
    //
    void writePrintWords(RWStructuredBuffer<uint> buffer, uint offset);
};

// With the `IPrintable` interface in place, we can now write
// a generic one-argument `println()` that works with any
// printable value.

void println<T : IPrintable>(T value)
{
    // In order to print a value we first compute the number of words
    // it needs in the print buffer.
    //
    uint wordCount = value.getPrintWordCount();

    // Then we can use `_beginPrint()` to allocate those words and
    // find the starting offset to write to.
    //
    uint wordOffset = _beginPrintln(wordCount);

    // And finally we can ask the value to write itself into the
    // buffer at the given offset.
    //
    value.writePrintWords(gPrintBuffer, wordOffset);
}

// Of course, in order to be able to print things with this `println()`
// operation, we need to have some types that implement `IPrintable`.
//
// In particular, we'd like to be able to print built-in types like
// `uint`, but we don't have access to the declaration of `uint`
// to be able to change it!
//
// It just so happens that another Slang feature, `extension`
// declarations, lets us extend a type with new methods *and*
// allows us to add new interface implementations to it.
//
// We can therefore making the exisint Slang `uint` type be
// printable.

extension uint : IPrintable // <-- Note: we are adding a conformance to `IPrintable here`
{
    // Printing a `uint` uses up two words in the buffer
    //
    uint getPrintWordCount() { return 2; }

    // Writing a command to print a `uint` is straightforward,
    // given knowledge of our encoding.
    //
    void writePrintWords(RWStructuredBuffer<uint> buffer, uint offset)
    {
        buffer[offset++] = (uint(PrintingOp.UInt32) << 16) | 1;
        buffer[offset++] = this;
    }
}

extension String : IPrintable
{
    uint getPrintWordCount() { return 2; }

    void writePrintWords(RWStructuredBuffer<uint> buffer, uint offset)
    {
        buffer[offset++] = (uint(PrintingOp.String) << 16) | 1;
        buffer[offset++] = getStringHash(this);
    }
}

// Where generics and interfaces start to pay off is when we want
// to scale up to a two-argument `println()` function that can
// work for any combination of printable types.

// Print two values, `a` and `b`.
//
// This function ensures that the values of `a` and `b`
// are written out atomically, without values printed
// from other threads spliced in between.
//
void println<A : IPrintable, B : IPrintable>(A a, B b)
{
    // To print two values atomically, we must first
    // allocate the total number of words that are
    // required to print the values.
    //
    uint wordCount = 0;
    uint aCount = a.getPrintWordCount(); wordCount += aCount;
    uint bCount = b.getPrintWordCount(); wordCount += bCount;

    // Then we can allocate those words atomically
    // with a single `_beginPrint()`.
    //
    uint wordOffset = _beginPrintln(wordCount);

    // Finally, we can write the words for each of `a`
    // and `b` to an appropriate offset in the print buffer,
    // without having to worry about other threads inserting
    // print commands between them.
    //
    a.writePrintWords(gPrintBuffer, wordOffset); wordOffset += aCount;
    b.writePrintWords(gPrintBuffer, wordOffset); wordOffset += bCount;
}

// We can then continue to build up to `println()` functions with
// three or more arguments.

void println<A : IPrintable, B : IPrintable, C : IPrintable>(
             A a,            B b,            C c)
{
    uint wordCount = 0;
    uint aCount = a.getPrintWordCount(); wordCount += aCount;
    uint bCount = b.getPrintWordCount(); wordCount += bCount;
    uint cCount = c.getPrintWordCount(); wordCount += cCount;

    uint wordOffset = _beginPrintln(wordCount);

    a.writePrintWords(gPrintBuffer, wordOffset); wordOffset += aCount;
    b.writePrintWords(gPrintBuffer, wordOffset); wordOffset += bCount;
    c.writePrintWords(gPrintBuffer, wordOffset); wordOffset += cCount;
}

// Further generalizing to four or more arguments is straightforward but tedious.
//
// A future version of Slang may support variadic functions, variadic generics,
// or some other facilities to make writing code like this easier.

// An important benefit of the approach we have taken here with an `IPrintable`
// interface is that arbitrary user-defined types can implement `IPrintable`
// and will work correctly with the existing `println()` definitions in
// this file.

// C-style `printf()`
// ==================
//
// Many developers who use C/C++ would prefer to be able to use traditional
// `printf()` with format strings. `printf`-based printing tends to be
// more readable than `println`-style alternatives, but comes at the cost
// of only supported a more restricted set of types for printing.
//
// Similar to the `println()` case, our Slang implementation of `printf()`
// starts with an allocation function that does the behind-the-scenes
// work.
//
// Note: We use the name `printf_` here because `printf` clashes with
// HLSL's printf.
//

uint _beginPrintf(String format, uint wordCount)
{
    // A printf command will start with the usual command header word,
    // along with a word for the (hashed) format string. These
    // two header words will be followed by the user-provided payload
    // words for all the format arguments.
    //
    uint wordOffset = _allocatePrintWords(wordCount + 2);
    gPrintBuffer[wordOffset++] = (uint(PrintingOp.PrintF) << 16) | (wordCount+1);
    gPrintBuffer[wordOffset++] = getStringHash(format);
    return wordOffset;
}

// Now we will define an interface for types that are allowed to
// appear as format arguments to `printf()`.

interface IPrintf
{
    // A `printf()` format argument must know how many words it encodes into
    uint getPrintfWordCount();

    // A `printf()` format argument must know how to encode itself
    void writePrintfWords(RWStructuredBuffer<uint> buffer, uint offset);
};

// The extension to make `uint` compatible with `printf()` is straightforward.

extension uint : IPrintf
{
    // A `uint` only consumes one word in the variadic payload.
    //
    // Note: unlike the case for `IPrintable` above, the encoding
    // for format args for `printf()` doesn't include type information.
    //
    uint getPrintfWordCount() { return 1; }

    // Writing the required data to the payload for `printf()` is simple
    void writePrintfWords(RWStructuredBuffer<uint> buffer, uint offset)
    {
        buffer[offset++] = this;
    }
}

extension String : IPrintf
{
    uint getPrintfWordCount() { return 1; }

    void writePrintfWords(RWStructuredBuffer<uint> buffer, uint offset)
    {
        buffer[offset++] = getStringHash(this);
    }
}


// A `printf()` with no format arguments can just call back to `_beginPrintf()`
void printf_(String format)
{
    _beginPrintf(format, 0);
}

// The `printf()` cases with one or more format arguments are all quite similar.

void printf_<A : IPrintf>(String format, A a)
{
    // We need to compute the words required by each format argument
    // and sum them up.
    //
    uint wordCount = 0;
    uint aCount = a.getPrintfWordCount(); wordCount += aCount;

    // We need to allocate a `printf()` command in the buffer with
    // the required number of words for format argument payload.
    //
    uint wordOffset = _beginPrintf(format, wordCount);

    // We need to write each format argument to the appropriate offset
    // in the payload part of the `printf()` command.
    //
    a.writePrintfWords(gPrintBuffer, wordOffset); wordOffset += aCount;
}

void printf_<A : IPrintf, B : IPrintf>(String format, A a, B b)
{
    uint wordCount = 0;
    uint aCount = a.getPrintfWordCount(); wordCount += aCount;
    uint bCount = b.getPrintfWordCount(); wordCount += bCount;

    uint wordOffset = _beginPrintf(format, wordCount);

    a.writePrintfWords(gPrintBuffer, wordOffset); wordOffset += aCount;
    b.writePrintfWords(gPrintBuffer, wordOffset); wordOffset += bCount;
}

// Extending this `printf()` implementation to handle more format arguments
// is straightforward, but tedious. Future versions of Slang might add
// support for variadic generics, which could make this code more compact.
