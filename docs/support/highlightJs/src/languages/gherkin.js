/*
 Language: Gherkin
 Author: Sam Pikesley (@pikesley) <sam.pikesley@theodi.org>
 Description: Gherkin (Cucumber etc)
 */

function (hljs) {
  return {
    aliases: ['feature'],
    keywords: 'Feature Background Ability Business\ Need Scenario Scenarios Scenario\ Outline Scenario\ Template Examples Given And Then But When',
    contains: [
      {
        className: 'keyword',
        begin: '\\*'
      },
      hljs.COMMENT('@[^@\r\n\t ]+', '$'),
      {
        begin: '\\|', end: '\\|\\w*$',
        contains: [
          {
            className: 'string',
            begin: '[^|]+'
          }
        ]
      },
      {
        className: 'variable',
        begin: '<', end: '>'
      },
      hljs.HASH_COMMENT_MODE,
      {
        className: 'string',
        begin: '"""', end: '"""'
      },
      hljs.QUOTE_STRING_MODE
    ]
  };
}
