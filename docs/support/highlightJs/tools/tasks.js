'use strict';

var _        = require('lodash');
var del      = require('del');
var gear     = require('gear');
var path     = require('path');
var utility  = require('./utility');

var parseHeader = utility.parseHeader;
var tasks       = require('gear-lib');

tasks.clean = function(directories, blobs, done) {
  directories = _.isString(directories) ? [directories] : directories;

  return del(directories, _.partial(done, _, blobs));
};
tasks.clean.type = 'collect';

// Depending on the languages required for the current language being
// processed, this task reorders it's dependencies first then include the
// language.
tasks.reorderDeps = function(options, blobs, done) {
  var buffer       = {},
      newBlobOrder = [];

  _.each(blobs, function(blob) {
    var basename = path.basename(blob.name),
        fileInfo = parseHeader(blob.result),
        extra = { blob: blob, processed: false };

    buffer[basename] = _.merge(extra, fileInfo || {});
  });

  function pushInBlob(object) {
    if(!object.processed) {
      object.processed = true;
      newBlobOrder.push(object.blob);
    }
  }

  _.each(buffer, function(buf) {
    var object;

    if(buf.Requires) {
      _.each(buf.Requires, function(language) {
        object = buffer[language];
        pushInBlob(object);
      });
    }

    pushInBlob(buf);
  });

  done(null, newBlobOrder);
};
tasks.reorderDeps.type = 'collect';

tasks.template = function(template, blob, done) {
  template = template || '';

  var filename = path.basename(blob.name),
      basename = path.basename(filename, '.js'),
      content  = _.template(template)({
        name: basename,
        filename: filename,
        content: blob.result.trim()
      });

  return done(null, new gear.Blob(content, blob));
};

tasks.templateAll = function(options, blobs, done) {
  return options.callback(blobs)
    .then(function(data) {
      var template = options.template || data.template,
          content  = _.template(template)(data);

      return done(null, [new gear.Blob(content)]);
    })
    .catch(done);
};
tasks.templateAll.type = 'collect';

tasks.rename = function(options, blob, done) {
  options = options || {};

  var name = blob.name,
      ext  = new RegExp(path.extname(name) + '$');

  name = name.replace(ext, options.extname);

  return done(null, new gear.Blob(blob.result, { name: name }));
};

// Adds the contributors from `AUTHORS.en.txt` onto the `package.json` file
// and moves the result into the `build` directory.
tasks.buildPackage = function(json, blob, done) {
  var result,
      lines = blob.result.split(/\r?\n/),
      regex = /^- (.*) <(.*)>$/;

  json.contributors = _.transform(lines, function(result, line) {
    var matches = line.match(regex);

    if(matches) {
      result.push({
        name: matches[1],
        email: matches[2]
      });
    }
  }, []);

  result = JSON.stringify(json, null, '  ');

  return done(null, new gear.Blob(result, blob));
};

// Mainly for replacing the keys of `utility.REPLACES` for it's values while
// skipping over strings, regular expressions, or comments. However, this is
// pretty generic so long as you use the `utility.replace` function, you can
// replace a regular expression with a string.
tasks.replaceSkippingStrings = function(params, blob, done) {
  var content = blob.result,
      length  = content.length,
      offset  = 0,

      replace = params.replace || '',
      regex   = params.regex,
      starts  = /\/\/|['"\/]/,

      result  = [],
      chunk, end, match, start, terminator;

  while(offset < length) {
    chunk = content.slice(offset);
    match = chunk.match(starts);
    end   = match ? match.index : length;

    chunk = content.slice(offset, end + offset);
    result.push(chunk.replace(regex, replace));

    offset += end;

    if(match) {
      // We found a starter sequence: either a `//` or a "quote"
      // In the case of `//` our terminator is the end of line.
      // Otherwise it's either a matching quote or an escape symbol.
      terminator = match[0] !== '//' ? new RegExp('[' + match[0] + '\\\\]')
                                     : /$/m;
      start      = offset;
      offset    += 1;

      while(true) {
        chunk = content.slice(offset);
        match = chunk.match(terminator);

        if(!match) {
          return done('Unmatched quote');
        }

        if(match[0] === '\\') {
          offset += match.index + 2;
        } else {
          offset += match.index + 1;
          result.push(content.slice(start, offset));
          break;
        }
      }
    }
  }

  return done(null, new gear.Blob(result.join(''), blob));
};

tasks.filter = function(callback, blobs, done) {
  var filteredBlobs = _.filter(blobs, callback);

  // Re-add in blobs required from header definition
  _.each(filteredBlobs, function(blob) {
    var dirname  = path.dirname(blob.name),
        content  = blob.result,
        fileInfo = parseHeader(content);

    if(fileInfo && fileInfo.Requires) {
      _.each(fileInfo.Requires, function(language) {
        var filename  = dirname + '/' + language,
            fileFound = _.find(filteredBlobs, { name: filename });

        if(!fileFound) {
          filteredBlobs.push(
            _.find(blobs, { name: filename }));
        }
      });
    }
  });

  return done(null, filteredBlobs);
};
tasks.filter.type = 'collect';

tasks.readSnippet = function(options, blob, done) {
  var name        = path.basename(blob.name, '.js'),
      fileInfo    = parseHeader(blob.result),
      snippetName = path.join('test', 'detect', name, 'default.txt');

  function onRead(error, blob) {
    if(error) return done(error); // ignore missing snippets

    var meta = { name: name + '.js', fileInfo: fileInfo };

    return done(null, new gear.Blob(blob.result, meta));
  }

  gear.Blob.readFile(snippetName, 'utf8', onRead, false);
};

// Packages up included languages into the core `highlight.js` and moves the
// result into the `build` directory.
tasks.packageFiles = function(options, blobs, done) {
  var content,
      coreFile  = _.head(blobs),
      languages = _.tail(blobs),

      lines     = coreFile.result
                    .replace(utility.regex.header, '')
                    .split('\n\n'),
      lastLine  = _.last(lines),
      langStr   = _.foldl(languages, function(str, language) {
                    return str + language.result + '\n';
                  }, '');

  lines[lines.length - 1] = langStr.trim();

  lines   = lines.concat(lastLine);
  content = lines.join('\n\n');

  return done(null, [new gear.Blob(content)]);
};
tasks.packageFiles.type = 'collect';

module.exports = new gear.Registry({ tasks: tasks });
