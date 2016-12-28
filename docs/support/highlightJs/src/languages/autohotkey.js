/*
Language: AutoHotkey
Author: Seongwon Lee <dlimpid@gmail.com>
Description: AutoHotkey language definition
Category: scripting
*/

function(hljs) {
  var BACKTICK_ESCAPE = {
    className: 'escape',
    begin: '`[\\s\\S]'
  };
  var COMMENTS = hljs.COMMENT(
    ';',
    '$',
    {
      relevance: 0
    }
  );
  var BUILT_IN = [
    {
      className: 'built_in',
      begin: 'A_[a-zA-Z0-9]+'
    },
    {
      className: 'built_in',
      beginKeywords: 'ComSpec Clipboard ClipboardAll ErrorLevel'
    }
  ];

  return {
    case_insensitive: true,
    keywords: {
      keyword: 'Break Continue Else Gosub If Loop Return While',
      literal: 'A true false NOT AND OR'
    },
    contains: BUILT_IN.concat([
      BACKTICK_ESCAPE,
      hljs.inherit(hljs.QUOTE_STRING_MODE, {contains: [BACKTICK_ESCAPE]}),
      COMMENTS,
      {
        className: 'number',
        begin: hljs.NUMBER_RE,
        relevance: 0
      },
      {
        className: 'var_expand', // FIXME
        begin: '%', end: '%',
        illegal: '\\n',
        contains: [BACKTICK_ESCAPE]
      },
      {
        className: 'label',
        contains: [BACKTICK_ESCAPE],
        variants: [
          {begin: '^[^\\n";]+::(?!=)'},
          {begin: '^[^\\n";]+:(?!=)', relevance: 0} // zero relevance as it catches a lot of things
                                                    // followed by a single ':' in many languages
        ]
      },
      {
        // consecutive commas, not for highlighting but just for relevance
        begin: ',\\s*,',
        relevance: 10
      }
    ])
  }
}
