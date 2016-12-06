'use strict';

var _        = require('lodash');
var bluebird = require('bluebird');
var fs       = bluebird.promisifyAll(require('fs'));
var glob     = require('glob');
var hljs     = require('../../build');
var path     = require('path');
var utility  = require('../utility');

function testLanguage(language) {
  describe(language, function() {
    var filePath  = utility.buildPath('markup', language, '*.expect.txt'),
        filenames = glob.sync(filePath);

    _.each(filenames, function(filename) {
      var testName   = path.basename(filename, '.expect.txt'),
          sourceName = filename.replace(/\.expect/, '');

      it('should markup ' + testName, function(done) {
        var sourceFile   = fs.readFileAsync(sourceName, 'utf-8'),
            expectedFile = fs.readFileAsync(filename, 'utf-8');

        bluebird.join(sourceFile, expectedFile, function(source, expected) {
          var actual = hljs.highlight(language, source).value;

          actual.should.equal(expected);
          done();
        });
      });
    });
  });
}

describe('markup generation test', function() {
  var languages = fs.readdirSync(utility.buildPath('markup'));

  _.each(languages, testLanguage, this);
});
