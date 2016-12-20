'use strict';

var path = require('path');

var browserBuild = require('./browser');
var registry     = require('./tasks');
var utility      = require('./utility');

var directory;

function moveLanguages() {
  var input   = path.join(directory.root, 'src', 'languages', '*.js'),
      output  = path.join(directory.build, 'languages'),
      regex   = utility.regex,
      replace = utility.replace,

      replaceArgs = replace(regex.header, ''),
      template    = 'hljs.registerLanguage(\'<%= name %>\','+
                    ' <%= content %>);\n';

  return {
    startLog: { task: ['log', 'Building language files.'] },
    read: {
      requires: 'startLog',
      task: ['glob', utility.glob(input)]
    },
    replace: { requires: 'read', task: ['replace', replaceArgs] },
    template: { requires: 'replace', task: ['template', template] },
    replace2: {
      requires: 'template',
      task: [ 'replaceSkippingStrings'
            , replace(regex.replaces, utility.replaceClassNames)
            ]
    },
    replace3: {
      requires: 'replace2',
      task: ['replace', replace(regex.classname, '$1.className')]
    },
    compressLog: {
      requires: 'replace3',
      task: ['log', 'Compressing languages files.']
    },
    minify: { requires: 'compressLog', task: 'jsminify' },
    rename: { requires: 'minify', task: ['rename', { extname: '.min.js' }] },
    writeLog: {
      requires: 'rename',
      task: ['log', 'Writing language files.']
    },
    write: { requires: 'writeLog', task: ['dest', output] }
  };
}

function moveStyles() {
  var css     = path.join(directory.root, 'src', 'styles', '*.css'),
      images  = path.join(directory.root, 'src', 'styles', '*.{jpg,png}'),
      output  = path.join(directory.build, 'styles'),
      options = { dir: output, encoding: 'binary' };

  return {
    startLog: { task: ['log', 'Building style files.'] },
    readCSS: { requires: 'startLog', task: ['glob', utility.glob(css)] },
    readImages: {
      requires: 'startLog',
      task: ['glob', utility.glob(images, 'binary')]
    },
    compressLog: {
      requires: 'readCSS',
      task: ['log', 'Compressing style files.']
    },
    minify: { requires: 'compressLog', task: 'cssminify' },
    rename: {
      requires: 'minify',
      task: ['rename', { extname: '.min.css' }]
    },
    writeLog: {
      requires: ['rename', 'readImages'],
      task: ['log', 'Writing style files.']
    },
    write: { requires: 'writeLog', task: ['dest', options] }
  };
}

module.exports = function(commander, dir) {
  directory = dir;

  return utility.toQueue([moveLanguages(), moveStyles()], registry)
    .concat(browserBuild(commander, dir));
};
