// For the basic introductions on using this build script, see:
//
// <https://highlightjs.readthedocs.org/en/latest/building-testing.html>
//
// Otherwise, lets explain what each build target does in more detail, for
// those that wish to know before running this script.
//
// Build Targets
// -------------
//
// * browser
//
//   The default target. This will package up the core `highlight.js` along
//   with all the language definitions into the file `highlight.pack.js` --
//   which will be compressed without including the option to disable it. It
//   also builds the documentation for our readthedocs page, mentioned
//   above, along with a local instance of the demo at:
//
//   <https://highlightjs.org/static/demo/>.
//
// * cdn
//
//   This will package up the core `highlight.js` along with all the
//   language definitions into the file `highlight.min.js` and compresses
//   all languages and styles into separate files. Since the target is for
//   CDNs -- like cdnjs and jsdelivr -- it doesn't matter if you put the
//   option to disable compression, this target is always be compressed. Do
//   keep in mind that we don't keep the build results in the main
//   repository; however, there is a separate repository for those that want
//   the CDN builds without using a third party site or building it
//   themselves. For those curious, head over to:
//
//   <https://github.com/highlightjs/cdn-release>
//
// * node
//
//   This build will transform the library into a CommonJS module. It
//   includes the generation of an `index.js` file that will be the main
//   file imported for use with Node.js or browserify. Do note that this
//   includes all languages by default and it might be too heavy to use for
//   browserify. Luckily, you can easily do two things to make the build
//   smaller; either specify the specific language/category you need or you
//   can use the browser or cdn builds and it will work like any CommonJS
//   file. Also with the CommonJS module, it includes the documentation for
//   our readthedocs page and the uncompressed styles. Getting this build is
//   pretty easy as it is the one that gets published to npm with the
//   standard `npm install highlight.js`, but you can also visit the package
//   page at:
//
//   <https://www.npmjs.com/package/highlight.js>
//
// * all
//
//   Builds every target and places the results into a sub-directory based
//   off of the target name relative to the `build` directory; for example,
//   "node" with go into `build/node`, "cdn" will go into `build/cdn`,
//   "browser" will go into `build/browser` and so forth.
//
// All build targets will end up in the `build` directory.
'use strict';

var commander = require('commander');
var path      = require('path');
var Queue     = require('gear').Queue;
var registry  = require('./tasks');

var build, dir = {};

commander
  .usage('[options] [<language>...]')
  .option('-n, --no-compress', 'Disable compression')
  .option('-t, --target <name>', 'Build for target ' +
                                 '[all, browser, cdn, node]',
                                 /^(browser|cdn|node|all)$/i, 'browser')
  .parse(process.argv);

commander.target = commander.target.toLowerCase();

build     = require('./' + commander.target);
dir.root  = path.dirname(__dirname);
dir.build = path.join(dir.root, 'build');

new Queue({ registry: registry })
  .clean(dir.build)
  .log('Starting build.')
  .series(build(commander, dir))
  .log('Finished build.')
  .run();
