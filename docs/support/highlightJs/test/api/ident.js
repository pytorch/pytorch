'use strict';

var hljs = require('../../build');

var pattern = new RegExp('^' + hljs.IDENT_RE + '$');

describe('.IDENT_RE', function() {
  it('should match non-underscore starting words', function() {
    var words = [ 'foo' , 'bar' , 'baz'
                , 'Foo' , 'Bar' , 'Baz'
                , 'f_oo', 'ba_r', 'baz_'
                , 'F_oo', 'Ba_r', 'Baz_'
                ];

    words.should.matchEach(pattern);
  });

  it('should not match underscore starting words', function() {
    var words = [ '_foo' , '_bar' , '_baz'
                , '_Foo' , '_Bar' , '_Baz'
                , '_f_oo', '_ba_r', '_baz_'
                , '_F_oo', '_Ba_r', '_Baz_'
                ];

    words.should.not.matchEach(pattern);
  });
});
