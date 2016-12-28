/*
Language: Ini
Contributors: Guillaume Gomez <guillaume1.gomez@gmail.com>
Category: common, config
*/

function(hljs) {
  var STRING = {
    className: "string",
    contains: [hljs.BACKSLASH_ESCAPE],
    variants: [
      {
        begin: "'''", end: "'''",
        relevance: 10
      }, {
        begin: '"""', end: '"""',
        relevance: 10
      }, {
        begin: '"', end: '"'
      }, {
        begin: "'", end: "'"
      }
    ]
  };
  return {
    aliases: ['toml'],
    case_insensitive: true,
    illegal: /\S/,
    contains: [
      hljs.COMMENT(';', '$'),
      hljs.HASH_COMMENT_MODE,
      {
        className: 'title',
        begin: /^\s*\[+/, end: /\]+/
      },
      {
        className: 'setting',
        begin: /^[a-z0-9\[\]_-]+\s*=\s*/, end: '$',
        contains: [
          {
            className: 'value',
            endsWithParent: true,
            keywords: 'on off true false yes no',
            contains: [
              {
                className: 'variable',
                variants: [
                  {begin: /\$[\w\d"][\w\d_]*/},
                  {begin: /\$\{(.*?)}/}
                ]
              },
              STRING,
              {
                className: 'number',
                begin: /([\+\-]+)?[\d]+_[\d_]+/
              },
              hljs.NUMBER_MODE
            ],
            relevance: 0
          }
        ]
      }
    ]
  };
}
