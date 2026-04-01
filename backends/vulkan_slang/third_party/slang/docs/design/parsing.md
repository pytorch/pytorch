# Resolving Ambiguity in Slang's Parser

A typical text-book style compiler front-end usually features explicit stages: tokenization, parsing, and semantic checking. Slang's original design follows this pattern, but the design has a drawback that it cannot effectively disambiguate the syntax due to lack of semantic info during parsing.

For example, without knowing what `X` is, it is impossible to tell whether `X<a&&b>(5)` means calling a generic function `X` with argument `5`, or computing the logical `AND` between condition `X < a` and `b > 5`.

Slang initially addresses this problem with a heursitic: if the compiler sees `IDENTIFIER` followed by `<`, it will try to parse the expression as a generic specialization first, and if that succeeds, it checks the token after the closing `>` to see if the following token is one of the possible "generic specialization followers". In this example, the next token is `(`, which is a "generic specialization follower", so the compiler determines that the expression being parsed is very likely a generic function call, and it will parse the expression as such. For reference, the full set of "generic specialization followers" are: `::`, `.`, `(`, `)`, `[`, `]`, `:`, `,`, `?`, `;`, `==`, `!=`, `>` and `>>`.

This simplistic heuristic is originated from the C# compiler, which works well there since C# doesn't allow generic value arguments, therefore things like `X<a&&b>...` or `X<a<y>...` can never be valid generic specializations. This isn't the case for Slang, where generic arguments can be int or boolean values, so `a&&b` and `a<y` are valid as generic arguments. Although using the same heuristic here works most of the time, it is still causing a lot of confusion to the users when the heuristic fails.

The ambiguity problem can be systematically solved if the parser has access to semantic info. If the parser knows that `X` is / isn't a generic, then it can parse the expression accordingly without any guess work. The key challenge is to make such semantic info available while we are still parsing.

## Two-stage Parsing

Slang solves this problem by breaking parsing into two stages: the decl parsing stage, and body parsing stage. Initially, we will parse the user source in the decl parsing stage. In this stage, we parse all decls, such as `struct`s, variables, functions etc. as usual, except that when we are about to parse the body of a function, we will just collect all tokens enclosed by `{` and `}` and store them in a raw list as a `UnparsedStmt` AST node. By deferring the parsing of function bodies, we no longer need to guess whether a `<` token inside a function body means generic specialization or less-than comparison.

After the decl parsing stage, we have the AST that represents the decl structure but not the function bodies. With this initial AST, we can start semantic checking. Once we reached the `UnparsedStmt` nodes, the semantic visitor will spawn a new `Parser` and start to parse the tokens stored in the `UnparsedStmt` node. When we spawn the parser in a semantic visitor, initialize the parser to be in `Body` parsing stage, and pass a pointer to the semantic visitor to the parser. This way, we are triggering the second parsing stage from the semantic visitor.

During the second parsing stage, whenever we see a `<` and need to disambiguate, we will use the semantic visitor to check the expression that has been parsed so far before `<`. If we are able to type check the expression and find it to be a `DeclRefExpr` referencing a generic decl, or an `OverloadedExpr` where one of the candidate is a generic decl, then we know `<` should be parsed as a generic specialization instead of `operator <`. If the expression before `<` checks to be a reference to a variable or a property, we should parse it as the comparison operator. The reason we are still parsing `<` as generic specialization when the expression before it is an non-generic function or type, is to allow us provide better error messages instead of just a "syntax error" somewhere down the line: in this case the user is most likely treating the non-generic type or function as a generic one by mistake, so we should diagnose as such. In the case that we are unable to properly check the preceeding expression or it checks to something else that we don't know, the compiler will fallback to the heuristic based method for disambiguation.

Note that in the second stage, parsing and semantic checking is interleaved organically. We no longer have a clean boundary between parsing and checking. However, the checking that happens in the second stage is on-demand and checks only necessary parts of the code to determine the type of the expression preceeding the `<` token. Any other code irrelevant to disambiguation purposes are left unchecked. Once the function body is fully parsed, the semantic visitor working on the function will make sure every node of the parsed AST is visited.

This two stage parsing technique should work well to correctly disambiguate code inside a function body. However the current implementation is not 100% bulletproof. Expressions at decl level, such as default values for struct members or function parameters, are still fully parsed in the first stage using the heuristic based method. However this should be a lesser problem in practice, because the default values are typically simple expressions and the chances of running into wrongly disambiguated case is much lower than in function bodies.

## Scope of Local Variables

Another issue linked with parsing is to correctly support the scope of local variables. A local variable should only be visible to code after its declaration within the same `{}` block. Consider this example:

```cpp
static int input = 100;
int f()
{
    input = 2; // global `input` is now 2
    int input = input + 1; // local `input` is now 3
    input = input + 2; // local `input` is now 5
    return input; // returns 5.
}
```

In Slang's implementation, we are creating a `ScopeDecl` container node for each `BlockStatement`, and variable declarations inside the block are added to the same `ScopeDecl`. This creates a problem for two stage parsing: to allow any expression to check during disambiguation, we need to insert variables into the scope as soon as they are parsed, but this means that when we are doing the "full checking" after the entire body is parsed, all variables are already registered in scope and discoverable when we are checking the earlier statements in the block. This means that the compiler cannot report an error if the user attempts to use a variable that is defined later in the block. In the example above, it means that when we are checking the first statement `input = 2`, the lookup logic for `input` will find the local variable instead of the global variable, thus generating the wrong code.

One way to solve this problem is instead of registering all local variables to the same scope owned by the containing `BlockStmt`, we make each local variable declaration own its own scope, that is ended at the end of the owning block. This way, all statements following the local variable declaration become the children of the local variable `DeclStmt`, effectively parsing the above example as:

```cpp
static int input = 100;
int f()
{
    input = 2; // global `input` is now 2
    {
        int input = input + 1; // local `input` is now 3
        input = input + 2; // local `input` is now 5
        return input; // returns 5.
    }
}

```

This will ensure the scope data-structure matches the semantic scope of the variable, and allow the compiler to produce the correct diagnostics.

However, expressing scope this way creates long nested chains in the AST, and leads to inefficient lookup and deep ASTs that risk overflowing the stack. Instead, Slang stays with the design to put all variables in the same block registered to the same `ScopeDecl`, but uses a separate state on each `VarDecl` called `hiddenFromLookup` to track whether or not the decl should be visible to lookup. During parsing, all decls are set to visible by default, so they can be used for disambiguation purpose. Once parsing is fully done and we are about to check a `BlockStmt`, we will first visit all `DeclStmt`s in the block, mark it as `invisible`, then continue checking the children statements. When checking encounters a `DeclStmt`, it will then mark the decl as `visible`, allowing it to be found by lookup logic for code after the declaration side. This solution allows us to respect the semantic scope of local variables without actually forming a long chain of scopes for a sequence of statements.

## Future Work: Extend Staged Parsing to Decl Scopes

We can further extend this to properly support expressions in global/decl scopes, such as default value expressions for struct members, or the type expressions for functions and global/member variables. To do so, we will use a different strategy for parsing expressions in the first parsing stage. Instead of parsing the expression directly, we should identify the token boundary of an expression without detailed understanding of the syntax. We will parse all expressions into `UnparsedExpr` nodes, which contain unparsed tokens for each expression. By doing so, the first parsing stage will give us an AST that is detailed enough to identify the names of types and functions, and whether or not they are generic. Then we can perform the semantic checking on the intial AST, and use the semantic checking to drive the parsing and checking of any `UnparsedExpr` and `UnparsedStmt`s.

## Future Work: ScopeRef

We can get rid of the `hiddenFromLookup` flag and use a more immutable representation of AST nodes if we introduce the concept of a `ScopeRef` that is a `Scope*` + `endIndex` to mark the boundary of the referenced scope. This way, different statements in a block can have different `ScopeRef` to the same scope but different ending member index. If we are looking up through a `ScopeRef` and find a variable in the scope that has an index greater than `endIndex`, we should treat the variable as invisible and report an error. This is cleaner, allowing better error messages, and avoids having to maintain mutable state flags on Decls.