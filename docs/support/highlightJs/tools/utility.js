'use strict';

var _        = require('lodash');
var bluebird = require('bluebird');
var glob     = bluebird.promisifyAll(require('glob')).globAsync;
var path     = require('path');

var Queue = require('gear').Queue;

var REPLACES,
    regex       = {},
    headerRegex = /^\s*\/\*((.|\r?\n)*?)\*/;

REPLACES = {
  'case_insensitive': 'cI',
  'lexemes': 'l',
  'contains': 'c',
  'keywords': 'k',
  'subLanguage': 'sL',
  'className': 'cN',
  'begin': 'b',
  'beginKeywords': 'bK',
  'end': 'e',
  'endsWithParent': 'eW',
  'illegal': 'i',
  'excludeBegin': 'eB',
  'excludeEnd': 'eE',
  'returnBegin': 'rB',
  'returnEnd': 'rE',
  'relevance': 'r',
  'variants': 'v',

  'IDENT_RE': 'IR',
  'UNDERSCORE_IDENT_RE': 'UIR',
  'NUMBER_RE': 'NR',
  'C_NUMBER_RE': 'CNR',
  'BINARY_NUMBER_RE': 'BNR',
  'RE_STARTERS_RE': 'RSR',
  'BACKSLASH_ESCAPE': 'BE',
  'APOS_STRING_MODE': 'ASM',
  'QUOTE_STRING_MODE': 'QSM',
  'PHRASAL_WORDS_MODE': 'PWM',
  'C_LINE_COMMENT_MODE': 'CLCM',
  'C_BLOCK_COMMENT_MODE': 'CBCM',
  'HASH_COMMENT_MODE': 'HCM',
  'NUMBER_MODE': 'NM',
  'C_NUMBER_MODE': 'CNM',
  'BINARY_NUMBER_MODE': 'BNM',
  'CSS_NUMBER_MODE': 'CSSNM',
  'REGEXP_MODE': 'RM',
  'TITLE_MODE': 'TM',
  'UNDERSCORE_TITLE_MODE': 'UTM',
  'COMMENT': 'C',

  'beginRe': 'bR',
  'endRe': 'eR',
  'illegalRe': 'iR',
  'lexemesRe': 'lR',
  'terminators': 't',
  'terminator_end': 'tE'
};

regex.replaces = new RegExp(
  '\\b(' + Object.keys(REPLACES).join('|') + ')\\b', 'g');

regex.classname = /(block|parentNode)\.cN/g;

regex.header = /^\s*(\/\*((.|\r?\n)*?)\*\/)?\s*/;

function replace(from, to) {
  return { regex: from, replace: to };
}

function replaceClassNames(match) {
  return REPLACES[match];
}

// All meta data, for each language definition, it store within the headers
// of each file in `src/languages`. `parseHeader` extracts that data and
// turns it into a useful object -- mainly for categories and what language
// this definition requires.
function parseHeader(content) {
  var headers,
      match = content.match(headerRegex);

  if (!match) {
    return null;
  }

  headers = _.compact(match[1].split('\n'));

  return _.foldl(headers, function(result, header) {
    var keyVal = header.trim().split(': '),
        key    = keyVal[0],
        value  = keyVal[1] || '';

    if(key !== 'Description' && key !== 'Language') {
      value = value.split(/\s*,\s*/);
    }

    result[key] = value;

    return result;
  }, {});
}

function filterByQualifiers(blob, languages, categories) {
  if(_.isEmpty(languages) && _.isEmpty(categories)) return true;

  var language         = path.basename(blob.name, '.js'),
      fileInfo         = parseHeader(blob.result),
      fileCategories   = fileInfo.Category || [],
      containsCategory = _.partial(_.contains, categories);

  if(!fileInfo) return false;

  return _.contains(languages, language) ||
         _.any(fileCategories, containsCategory);
}

// For the filter task in `tools/tasks.js`, this function will look for
// categories and languages specificed from the CLI.
function buildFilterCallback(qualifiers) {
  var result     = _.partition(qualifiers, { 0: ':' }),
      languages  = result[1],
      categories = _.map(result[0], function(category) {
                     return category.slice(1);
                   });

  return _.partial(filterByQualifiers, _, languages, categories);
}

function globDefaults(pattern, encoding) {
  encoding = encoding || 'utf8';

  // The limit option is a fix for issue #636 when the build script would
  // EMFILE error for those systems who had a limit of open files per
  // process.
  //
  // <https://github.com/isagalaev/highlight.js/issues/636>
  return { pattern: pattern, limit: 50, encoding: encoding };
}

function getStyleNames() {
  var stylesDir = 'src/styles/',
      options   = { ignore: stylesDir + 'default.css' };

  return glob(stylesDir + '*.css', options)
    .map(function(style) {
      var basename = path.basename(style, '.css'),
          name     = _.startCase(basename),
          pathName = path.relative('src', style);

      return { path: pathName, name: name };
    });
}

function toQueue(tasks, registry) {
  return _.map(tasks, function(task) {
    return new Queue({ registry: registry })
      .tasks(task);
  });
}

module.exports = {
  buildFilterCallback: buildFilterCallback,
  getStyleNames: getStyleNames,
  glob: globDefaults,
  parseHeader: parseHeader,
  regex: regex,
  replace: replace,
  replaceClassNames: replaceClassNames,
  toQueue: toQueue
};
