'use strict';

var _       = require('lodash');
var utility = require('../utility');

describe('language alias', function() {
  before(function() {
    var testHTML = document.querySelectorAll('#language-alias .hljs');

    this.blocks = _.map(testHTML, 'innerHTML');
  });

  it('should highlight as aliased language', function(done) {
    var filename = utility.buildPath('expect', 'languagealias.txt'),
        actual   = this.blocks[0];

    utility.expectedFile(filename, 'utf-8', actual, done);
  });
});
