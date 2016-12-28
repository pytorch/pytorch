/*
Language: Golo
Author: Philippe Charriere <ph.charriere@gmail.com>
Description: a lightweight dynamic language for the JVM, see http://golo-lang.org/
*/

function(hljs) {
    return {
      keywords: {
        keyword:
          'println readln print import module function local return let var ' +
          'while for foreach times in case when match with break continue ' +
          'augment augmentation each find filter reduce ' +
          'if then else otherwise try catch finally raise throw orIfNull',
        typename:
          'DynamicObject|10 DynamicVariable struct Observable map set vector list array',
        literal:
          'true false null'
      },
      contains: [
        hljs.HASH_COMMENT_MODE,
        hljs.QUOTE_STRING_MODE,
        hljs.C_NUMBER_MODE,
        {
          className: 'annotation', begin: '@[A-Za-z]+'
        }
      ]
    }
}
