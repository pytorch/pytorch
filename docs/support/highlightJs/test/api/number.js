'use strict';

var _       = require('lodash');
var hljs    = require('../../build');
var utility = require('../utility');

var pattern        = new RegExp(hljs.NUMBER_RE + '$');
var numberToString = utility.numberToString;

describe('.NUMBER_RE', function() {
  it('should match regular numbers and decimals', function() {
    var number        = _.map(_.range(0, 1001), numberToString);
    var decimal       = _.map(_.range(0, 1.001, 0.001), numberToString);
    var noLeadingZero = ['.1234', '.5206', '.0002', '.9998'];

    var numbers = [].concat(number, decimal, noLeadingZero);

    numbers.should.matchEach(pattern);
  });

  it('should not match hex or binary numbers', function() {
    var numbers = [ '0xbada55', '0xfa1755', '0x45362e'
                  , '0b0101'  , '0b1100'  , '0b1001'
                  ];

    numbers.should.not.matchEach(pattern);
  });
});
