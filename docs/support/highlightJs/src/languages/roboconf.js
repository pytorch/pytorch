/*
Language: Roboconf
Author: Vincent Zurczak <vzurczak@linagora.com>
Website: http://roboconf.net
Description: Syntax highlighting for Roboconf's DSL
Category: config
*/

function(hljs) {
  var IDENTIFIER = '[a-zA-Z-_][^\n{\r\n]+\\{';

  return {
    aliases: ['graph', 'instances'],
    case_insensitive: true,
    keywords: 'import',
    contains: [
      // Facet sections
      {
        className: 'facet',
        begin: '^facet ' + IDENTIFIER,
        end: '}',
        keywords: 'facet installer exports children extends',
        contains: [
          hljs.HASH_COMMENT_MODE
        ]
      },

      // Instance sections
      {
        className: 'instance-of',
        begin: '^instance of ' + IDENTIFIER,
        end: '}',
        keywords: 'name count channels instance-data instance-state instance of',
        contains: [
          // Instance overridden properties
          {
            className: 'keyword',
            begin: '[a-zA-Z-_]+( |\t)*:'
          },
          hljs.HASH_COMMENT_MODE
        ]
      },

      // Component sections
      {
        className: 'component',
        begin: '^' + IDENTIFIER,
        end: '}',
        lexemes: '\\(?[a-zA-Z]+\\)?',
        keywords: 'installer exports children extends imports facets alias (optional)',
        contains: [
          // Imported component variables
          {
            className: 'string',
            begin: '\\.[a-zA-Z-_]+',
            end: '\\s|,|;',
            excludeEnd: true
          },
          hljs.HASH_COMMENT_MODE
        ]
      },

      // Comments
      hljs.HASH_COMMENT_MODE
    ]
  };
}
