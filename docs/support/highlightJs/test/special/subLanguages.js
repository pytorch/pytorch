'use strict';

var utility = require('../utility');

describe('sub-languages', function() {
  before(function() {
    this.block = document.querySelector('#sublanguages');
  });

  it('should highlight XML with PHP and JavaScript', function(done) {
    var filename = utility.buildPath('expect', 'sublanguages.txt'),
        actual   = this.block.innerHTML;

    utility.expectedFile(filename, 'utf-8', actual, done);
  });
});
