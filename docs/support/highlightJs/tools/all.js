'use strict';

var _       = require('lodash');
var path    = require('path');
var cdn     = require('./cdn');
var node    = require('./node');
var browser = require('./browser');

function newBuildDirectory(dir, subdir) {
  var build = path.join(dir.build, subdir);

  return { build: build };
}

module.exports = function(commander, dir) {
  var data = {};

  _.each(['cdn', 'node', 'browser'], function(target) {
    var newDirectory = newBuildDirectory(dir, target),
        directory    = _.defaults(newDirectory, dir),
        options      = _.defaults({ target: target }, commander);

    data[target] = {
      directory: directory,
      commander: options
    };
  });

  return [].concat(
    cdn(data.cdn.commander, data.cdn.directory),
    node(data.node.commander, data.node.directory),
    browser(data.browser.commander, data.browser.directory)
  );
};
