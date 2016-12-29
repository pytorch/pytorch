'use strict';

var _       = require('lodash');
var hljs    = require('../../build');
var utility = require('../utility');

var pattern        = new RegExp(hljs.C_NUMBER_RE + '$');
var numberToString = utility.numberToString;

describe('.C_NUMBER_RE', function() {
  it('should match regular numbers', function() {
    var numbers = _.map(_.range(0, 1001), numberToString);

    numbers.should.matchEach(pattern);
  });

  it('should match decimals', function() {
    var decimal       = _.map(_.range(0, 1.001, 0.001), numberToString);
    var noLeadingZero = ['.1234', '.5206', '.0002', '.9998'];

    var numbers = [].concat(decimal, noLeadingZero);

    numbers.should.matchEach(pattern);
  });

  it('should match hex numbers', function() {
    var numbers = [ '0xbada55', '0xfa1755', '0x45362e', '0xfedcba'
                  , '0x123456', '0x00000f', '0xfff000', '0xf0e1d2'
                  ];

    numbers.should.matchEach(pattern);
  });

  it('should not match hex numbers greater than "f"', function() {
    var numbers = ['0xgada55', '0xfh1755', '0x45i62e'];

    numbers.should.not.matchEach(pattern);
  });

  it('should not match binary numbers', function() {
    var numbers = ['0b0101', '0b1100', '0b1001'];

    numbers.should.not.matchEach(pattern);
  });
});
