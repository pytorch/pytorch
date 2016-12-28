/*
Language: Dockerfile
Requires: bash.js
Author: Alexis Hénaut <alexis@henaut.net>
Description: language definition for Dockerfile files
Category: config
*/

function(hljs) {
  return {
    aliases: ['docker'],
    case_insensitive: true,
    keywords: {
      built_ins: 'from maintainer cmd expose add copy entrypoint volume user workdir onbuild run env'
    },
    contains: [
      hljs.HASH_COMMENT_MODE,
      {
        keywords : {
          built_in: 'run cmd entrypoint volume add copy workdir onbuild'
        },
        begin: /^ *(onbuild +)?(run|cmd|entrypoint|volume|add|copy|workdir) +/,
        starts: {
          end: /[^\\]\n/,
          subLanguage: 'bash'
        }
      },
      {
        keywords: {
          built_in: 'from maintainer expose env user onbuild'
        },
        begin: /^ *(onbuild +)?(from|maintainer|expose|env|user|onbuild) +/, end: /[^\\]\n/,
        contains: [
          hljs.APOS_STRING_MODE,
          hljs.QUOTE_STRING_MODE,
          hljs.NUMBER_MODE,
          hljs.HASH_COMMENT_MODE
        ]
      }
    ]
  }
}
