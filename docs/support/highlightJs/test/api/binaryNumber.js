'use strict';

var hljs = require('../../build');

var pattern = new RegExp(hljs.BINARY_NUMBER_RE + '$');

describe('.BINARY_NUMBER_RE', function() {
  it('should match binary numbers', function() {
    var numbers = [ '0b0101', '0b1100', '0b1001'
                  , '0b11110101', '0b11001111'
                  , '0b1010111111000001'
                  ];

    numbers.should.matchEach(pattern);
  });

  it('should not match binary numbers greater than 2', function() {
    var numbers = [ '0b2101', '0b1130', '0b1041'
                  , '0b11150101', '0b11061111'
                  , '0b1010111117000001'
                  ];

    numbers.should.not.matchEach(pattern);
  });
});
