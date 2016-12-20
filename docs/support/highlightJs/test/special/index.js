'use strict';

var _       = require('lodash');
var fs      = require('fs');
var hljs    = require('../../build');
var jsdom   = require('jsdom').jsdom;
var utility = require('../utility');

var blocks,
    filename = utility.buildPath('index.html'),
    page     = fs.readFileSync(filename, 'utf-8');

// Allows hljs to use document
global.document = jsdom(page);

// Setup hljs environment
hljs.configure({ tabReplace: '    ' });
hljs.initHighlighting();

// Setup hljs for non-`<pre><code>` tests
hljs.configure({ useBR: true });

blocks = document.querySelectorAll('.code');
_.each(blocks, hljs.highlightBlock);

describe('special cases test', function() {
  require('./explicitLanguage');
  require('./customMarkup');
  require('./languageAlias');
  require('./noHighlight');
  require('./subLanguages');
  require('./buildClassName');
  require('./useBr');
});
