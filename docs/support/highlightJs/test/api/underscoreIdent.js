'use strict';

var hljs = require('../../build');

var pattern = new RegExp('^' + hljs.UNDERSCORE_IDENT_RE + '$');

describe('.UNDERSCORE_IDENT_RE', function() {
  it('should match any word starting without numbers', function() {
    var words = [ 'foo'  , 'bar'  , 'baz'
                , 'Foo'  , 'Bar'  , 'Baz'
                , '_foo' , '_bar' , '_baz'
                , '_Foo' , '_Bar' , '_Baz'
                , '_f_oo', '_ba_r', '_baz_'
                , '_F_oo', '_Ba_r', '_Baz_'
                ];

    words.should.matchEach(pattern);
  });

  it('should not match any word starting with numbers', function() {
    var words = [ '1foo' , '6bar' , '0baz'
                , '2Foo' , '7Bar' , '1Baz'
                , '3f_oo', '8ba_r', '2baz_'
                , '4F_oo', '9Ba_r', '3Baz_'
                ];

    words.should.not.matchEach(pattern);
  });
});
