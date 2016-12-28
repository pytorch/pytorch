'use strict';

var _    = require('lodash');
var fs   = require('fs');
var path = require('path');

// Build a path relative to `test/`
exports.buildPath = function() {
  var args  = _.slice(arguments, 0),
      paths = [__dirname].concat(args);

  return path.join.apply(this, paths);
};

exports.numberToString = _.method('toString');

exports.expectedFile = function(filename, encoding, actual, done) {
  fs.readFile(filename, encoding, function(error, expected) {
    if(error) return done(error);

    actual.should.equal(expected);
    done();
  });
};

exports.setupFile = function(filename, encoding, that, testHTML, done) {
  fs.readFile(filename, encoding, function(error, expected) {
    if(error) return done(error);

    that.expected = expected.trim();
    that.blocks   = _.map(testHTML, 'innerHTML');
    done();
  });
};
