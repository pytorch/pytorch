/*
 Language: GAMS
 Author: Stefan Bechert <stefan.bechert@gmx.net>
 Contributors: Oleg Efimov <efimovov@gmail.com>
 Description: The General Algebraic Modeling System language
 Category: scientific
 */

function (hljs) {
  var KEYWORDS =
    'abort acronym acronyms alias all and assign binary card diag display else1 eps eq equation equations file files ' +
    'for1 free ge gt if inf integer le loop lt maximizing minimizing model models na ne negative no not option ' +
    'options or ord parameter parameters positive prod putpage puttl repeat sameas scalar scalars semicont semiint ' +
    'set1 sets smax smin solve sos1 sos2 sum system table then until using variable variables while1 xor yes';

  return {
    aliases: ['gms'],
    case_insensitive: true,
    keywords: KEYWORDS,
    contains: [
      {
        className: 'section',
        beginKeywords: 'sets parameters variables equations',
        end: ';',
        contains: [
          {
            begin: '/',
            end: '/',
            contains: [hljs.NUMBER_MODE]
          }
        ]
      },
      {
        className: 'string',
        begin: '\\*{3}', end: '\\*{3}'
      },
      hljs.NUMBER_MODE,
      {
        className: 'number',
        begin: '\\$[a-zA-Z0-9]+'
      }
    ]
  };
}

