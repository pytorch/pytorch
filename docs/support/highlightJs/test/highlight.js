'use strict';

// Tests specific to the API exposed inside the hljs object. Right now, that
// only includes tests for auto detection of languages via `highlightAuto`
// and several common regular expressions.
require('./api');

// HTML markup tests for particular languages. Usually when there is an
// incorrect highlighting of one language, once the bug get fixed, the
// expected markup will be added into the `test/markup` folder to keep
// theses highlighting errors from cropping up again.
require('./markup');

// Tests meant for the browser only. Using the `test/index.html` file along
// with `jsdom` these tests check for things like: custom markup already
// existing in the code being highlighted, blocks that disable highlighting,
// and several other cases. Do note that the `test/index.html` file isn't
// actually used to test inside a browser but `jsdom` acts as a virtual
// browser inside of node.js and runs together with all the other tests.
require('./special');
