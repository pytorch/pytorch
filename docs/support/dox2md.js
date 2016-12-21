#!/usr/bin/env node
var util = require('util');
var cli = require('commander');

cli
  .option('--private', 'Show privates')
  .option('--source', 'Show sources')
  .option('--debug', 'Print debug JSON')
  .option('--module-level [n]', 'Heading level for modules [2]', 2)
  .option('--default-level [n]', 'Heading level for everything [3]', 3)
  .option('--lang <lang>', 'Sets highlight block language [js]', 'js')
  .on('--help', function() {
    console.log('  Basic use:');
    console.log('');
    console.log('    $ dox -r < file.js | dox2md > out');
    process.exit(0);
  });
cli.parse(process.argv);

readStdin(function(json) {
  var blocks = JSON.parse(json);
  var lines = [];

  blocks = removeBlocks(blocks, function(b) { return !b.ignore; });

  if (!cli['private'])
    blocks = removeBlocks(blocks, function(b) { return !b.isPrivate; });

  // Definitions
  blocks.forEach(function(block, i) {
    if (!block.ctx) return;
    lines.push(markdownify(block, i, cli));
  });

  // Jump links
  lines.push("");
  blocks.forEach(function(block, i) {
    if (!block.ctx) return;
    var name = namify(block.ctx);
    lines.push('[' + name + ']: #' + slugify(name));
  });

  if (cli.debug)
    process.stderr.write(util.inspect(blocks, false, Infinity, true));

  console.log(lines.join("\n"));
});

function markdownify(block, i, options) {
  var lines = [];
  var name = namify(block.ctx);
  var level;

  // Heading
  if (i === 0) { level = 1; }
  else if (isModule(name)) { level = options.moduleLevel; }
  else { level = options.defaultLevel; }
  lines.push(heading(name, level));
  lines.push(fixMarkdown(block.description.full, options.lang));
  lines.push("");

  // Sources
  if (options.source) {
    lines.push("> Source:", "", codeBlock(block.code, options.lang));
  }

  return lines.join("\n");
}

function fixMarkdown(buf, lang) {
  var code = buf.match(/^( {4}[^\n]+\n*)+/gm) || [];

  code.forEach(function(block){
    var code = block.replace(/^ {4}/gm, '');
    buf = buf.replace(block, codeBlock(code, lang));
  });

  return buf;
}

function codeBlock(code, lang) {
  return '```'+lang+'\n' + code.trimRight() + '\n```\n\n';
}

// Returns the name for a given context.
function namify(ctx) {
  return ctx.string
    .replace('.prototype.', '#');
}

// Checks if a given name is a module.
function isModule(name) {
  return !! name.match(/^[A-Za-z][A-Za-z0-9_]*$/);
}

function heading(str, level) {
  if (level === 1) return str + "\n" + times("=", str.length) + "\n";
  if (level === 2) return str + "\n" + times("-", str.length) + "\n";
  return times('#', level) + ' ' + str + "\n";
}

function times(str, n) {
  var re = '';
  for (var i=0; i<n; ++i) re += str;
  return re;
}

/**
 * Removes blocks matching a given criteria.
 */
function removeBlocks(blocks, fn) {
  return blocks.reduce(function(arr, b) {
    if (fn(b)) arr.push(b);
    return arr;
  }, []);
}

function readStdin(callback) {
  var buf = '';
  process.stdin.setEncoding('utf8');
  process.stdin.on('data', function(chunk){ buf += chunk; });
  process.stdin.on('end', function(){ callback(buf); });
  process.stdin.resume();
}

function slugify(text) {
  return text.toLowerCase().match(/[a-z0-9]+/g).join('-');
}
