'use strict';

var _       = require('lodash');
var utility = require('../utility');

describe('custom markup', function() {
  before(function() {
    var testHTML = document.querySelectorAll('#custom-markup .hljs');

    this.blocks = _.map(testHTML, 'innerHTML');
  });

  it('should replace tabs', function(done) {
    var filename = utility.buildPath('expect', 'tabreplace.txt'),
        actual   = this.blocks[0];

    utility.expectedFile(filename, 'utf-8', actual, done);
  });

  it('should keep custom markup', function(done) {
    var filename = utility.buildPath('expect', 'custommarkup.txt'),
        actual   = this.blocks[1];

    utility.expectedFile(filename, 'utf-8', actual, done);
  });

  it('should keep custom markup and replace tabs', function(done) {
    var filename = utility.buildPath('expect', 'customtabreplace.txt'),
        actual   = this.blocks[2];

    utility.expectedFile(filename, 'utf-8', actual, done);
  });

  it('should keep the same amount of void elements (<br>, <hr>, ...)', function(done) {
    var filename = utility.buildPath('expect', 'brInPre.txt'),
        actual   = this.blocks[3];

    utility.expectedFile(filename, 'utf-8', actual, done);
  });
});
