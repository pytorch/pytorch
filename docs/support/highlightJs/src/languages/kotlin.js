/*
 Language: Kotlin
 Author: Sergey Mashkov <cy6erGn0m@gmail.com>
 Category: misc
 */


function (hljs) {
  var KEYWORDS = 'val var get set class trait object public open private protected ' +
    'final enum if else do while for when break continue throw try catch finally ' +
    'import package is as in return fun override default companion reified inline volatile transient native';

  return {
    keywords: {
      typename: 'Byte Short Char Int Long Boolean Float Double Void Unit Nothing',
      literal: 'true false null',
      keyword: KEYWORDS
    },
    contains : [
      hljs.COMMENT(
        '/\\*\\*',
        '\\*/',
        {
          relevance : 0,
          contains : [{
            className : 'doctag',
            begin : '@[A-Za-z]+'
          }]
        }
      ),
      hljs.C_LINE_COMMENT_MODE,
      hljs.C_BLOCK_COMMENT_MODE,
      {
        className: 'type',
        begin: /</, end: />/,
        returnBegin: true,
        excludeEnd: false,
        relevance: 0
      },
      {
        className: 'function',
        beginKeywords: 'fun', end: '[(]|$',
        returnBegin: true,
        excludeEnd: true,
        keywords: KEYWORDS,
        illegal: /fun\s+(<.*>)?[^\s\(]+(\s+[^\s\(]+)\s*=/,
        relevance: 5,
        contains: [
          {
            begin: hljs.UNDERSCORE_IDENT_RE + '\\s*\\(', returnBegin: true,
            relevance: 0,
            contains: [hljs.UNDERSCORE_TITLE_MODE]
          },
          {
            className: 'type',
            begin: /</, end: />/, keywords: 'reified',
            relevance: 0
          },
          {
            className: 'params',
            begin: /\(/, end: /\)/,
            keywords: KEYWORDS,
            relevance: 0,
            illegal: /\([^\(,\s:]+,/,
            contains: [
              {
                className: 'typename',
                begin: /:\s*/, end: /\s*[=\)]/, excludeBegin: true, returnEnd: true,
                relevance: 0
              }
            ]
          },
          hljs.C_LINE_COMMENT_MODE,
          hljs.C_BLOCK_COMMENT_MODE
        ]
      },
      {
        className: 'class',
        beginKeywords: 'class trait', end: /[:\{(]|$/,
        excludeEnd: true,
        illegal: 'extends implements',
        contains: [
          hljs.UNDERSCORE_TITLE_MODE,
          {
            className: 'type',
            begin: /</, end: />/, excludeBegin: true, excludeEnd: true,
            relevance: 0
          },
          {
            className: 'typename',
            begin: /[,:]\s*/, end: /[<\(,]|$/, excludeBegin: true, returnEnd: true
          }
        ]
      },
      {
        className: 'variable', beginKeywords: 'var val', end: /\s*[=:$]/, excludeEnd: true
      },
      hljs.QUOTE_STRING_MODE,
      {
        className: 'shebang',
        begin: "^#!/usr/bin/env", end: '$',
        illegal: '\n'
      },
      hljs.C_NUMBER_MODE
    ]
  };
}
