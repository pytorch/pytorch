'use strict';

var _        = require('lodash');
var bluebird = require('bluebird');
var fs       = bluebird.promisifyAll(require('fs'));
var path     = require('path');

var registry = require('./tasks');
var utility  = require('./utility');

var directory;

function templateAllFunc(blobs) {
  var name = path.join('demo', 'index.html');

  return bluebird.join(
    fs.readFileAsync(name),
    utility.getStyleNames(),
    function(template, styles) {
      return {
        template: template,
        path: path,
        blobs: _.compact(blobs),
        styles: styles
      };
    });
}

function copyDocs() {
  var input  = path.join(directory.root, 'docs', '*.rst'),
      output = path.join(directory.build, 'docs');

  return {
    startLog: { task: ['log', 'Copying documentation.'] },
    read: { requires: 'startLog', task: ['glob', utility.glob(input)] },
    writeLog: { requires: 'read', task: ['log', 'Writing documentation.'] },
    write: { requires: 'writeLog', task: ['dest', output] }
  };
}

function generateDemo(filterCB, readArgs) {
  var staticArgs   = utility.glob(path.join('demo', '*.{js,css}')),
      stylesArgs   = utility.glob(path.join('src', 'styles', '*'), 'binary'),
      demoRoot     = path.join(directory.build, 'demo'),
      templateArgs = { callback: templateAllFunc },
      destArgs     = {
        dir: path.join(demoRoot, 'styles'),
        encoding: 'binary'
      };

  return {
    logStart: { task: ['log', 'Generating demo.'] },
    readLanguages: { requires: 'logStart', task: ['glob', readArgs] },
    filterSnippets: { requires: 'readLanguages', task: ['filter', filterCB] },
    readSnippet: { requires: 'filterSnippets', task: 'readSnippet' },
    template: {
      requires: 'readSnippet',
      task: ['templateAll', templateArgs]
    },
    write: {
      requires: 'template',
      task: ['write', path.join(demoRoot, 'index.html')]
    },
    readStatic: { requires: 'logStart', task: ['glob', staticArgs] },
    writeStatic: { requires: 'readStatic', task: ['dest', demoRoot] },
    readStyles: { requires: 'logStart', task: ['glob', stylesArgs] },
    writeStyles: { requires: 'readStyles', task: ['dest', destArgs] }
  };
}

module.exports = function(commander, dir) {
  directory = dir;

  var hljsExt, output, requiresTask, tasks,
      replace           = utility.replace,
      regex             = utility.regex,
      replaceClassNames = utility.replaceClassNames,

      coreFile     = path.join('src', 'highlight.js'),
      languages    = utility.glob(path.join('src', 'languages', '*.js')),
      filterCB     = utility.buildFilterCallback(commander.args),
      replaceArgs  = replace(regex.header, ''),
      templateArgs =
        'hljs.registerLanguage(\'<%= name %>\', <%= content %>);\n';

  tasks = {
    startLog: { task: ['log', 'Building highlight.js pack file.'] },
    readCore: { requires: 'startLog', task: ['read', coreFile] },
    read: { requires: 'startLog', task: ['glob', languages] },
    filter: { requires: 'read', task: ['filter', filterCB] },
    reorder: { requires: 'filter', task: 'reorderDeps' },
    replace: { requires: 'reorder', task: ['replace', replaceArgs] },
    template: { requires: 'replace', task: ['template', templateArgs] },
    packageFiles: {
      requires: ['readCore', 'template'],
      task: 'packageFiles'
    }
  };
  requiresTask = 'packageFiles';

  if(commander.compress || commander.target === 'cdn') {
    tasks.compresslog = {
      requires: requiresTask,
      task: ['log', 'Compressing highlight.js pack file.']
    };

    tasks.replace2 = {
      requires: 'compresslog',
      task: [ 'replaceSkippingStrings'
            , replace(regex.replaces, replaceClassNames)
            ]
    };

    tasks.replace3 = {
      requires: 'replace2',
      task: ['replace', replace(regex.classname, '$1.className')]
    };

    tasks.minify  = { requires: 'replace3', task: 'jsminify' };
    requiresTask  = 'minify';
  }

  tasks.writelog = {
    requires: requiresTask,
    task: ['log', 'Writing highlight.js pack file.']
  };

  hljsExt = commander.target === 'cdn' ? 'min' : 'pack';
  output  = path.join(directory.build, 'highlight.' + hljsExt + '.js');

  tasks.write = {
    requires: 'writelog',
    task: ['write', output]
  };

  tasks = (commander.target === 'browser')
        ? [copyDocs(), generateDemo(filterCB, languages), tasks]
        : [tasks];

  return utility.toQueue(tasks, registry);
};
