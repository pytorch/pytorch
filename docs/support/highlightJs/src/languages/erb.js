/*
Language: ERB (Embedded Ruby)
Requires: xml.js, ruby.js
Author: Lucas Mazza <lucastmazza@gmail.com>
Contributors: Kassio Borges <kassioborgesm@gmail.com>
Description: "Bridge" language defining fragments of Ruby in HTML within <% .. %>
Category: template
*/

function(hljs) {
  return {
    subLanguage: 'xml',
    contains: [
      hljs.COMMENT('<%#', '%>'),
      {
        begin: '<%[%=-]?', end: '[%-]?%>',
        subLanguage: 'ruby',
        excludeBegin: true,
        excludeEnd: true
      }
    ]
  };
}
