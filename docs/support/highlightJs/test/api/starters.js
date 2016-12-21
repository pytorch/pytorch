'use strict';

var hljs = require('../../build');

var pattern = new RegExp('^' + hljs.RE_STARTERS_RE + '$');

describe('.RE_STARTERS_RE', function() {
  it('should match boolean operators', function() {
    var operators = [ '!', '!=', '!==', '==', '===',  '<=', '>='
                    , '<', '>', '||', '&&', '?'
                    ];

    operators.should.matchEach(pattern);
  });

  it('should match arithmetic operators', function() {
    var operators = [ '*', '*=', '+', '+=', '-', '-=', '/', '/='
                    , '%', '%='
                    ];

    operators.should.matchEach(pattern);
  });

  it('should match binary operators', function() {
    var operators = [ '&', '&=', '|', '|=', '<<', '<<=', '>>', '>>='
                    , '>>>', '>>>=', '^', '^=', '~'
                    ];

    operators.should.matchEach(pattern);
  });

  it('should match miscellaneous operators', function() {
    var operators = [',', '=', ':', ';', '[', '{', '('];

    operators.should.matchEach(pattern);
  });
});
