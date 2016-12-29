/*! Smartquotes (c) 2012, Rico Sta. Cruz. MIT License.
 *  http://github.com/rstacruz/jquery-stuff/tree/master/smartquotes */

// Translates plain ASCII punctuation characters into typographic punctuation
// HTML entities. Inspired by Smartypants.
//
//     $(function() {
//       $("body").smartquotes();
//     });
//
(function($) {
  // http://www.leancrew.com/all-this/2010/11/smart-quotes-in-javascript/
  $.smartquotes = function(a) {
    a = a.replace(/(^|[-\u2014\s(\["])'/g, "$1\u2018");       // opening singles
    a = a.replace(/'/g, "\u2019");                            // closing singles & apostrophes
    a = a.replace(/(^|[-\u2014/\[(\u2018\s])"/g, "$1\u201c"); // opening doubles
    a = a.replace(/"/g, "\u201d");                            // closing doubles
    a = a.replace(/\.\.\./g, "\u2026");                       // ellipses
    a = a.replace(/--/g, "\u2014");                           // em-dashes
    return a;
  };

  // http://stackoverflow.com/questions/298750/how-do-i-select-text-nodes-with-jquery
  function getTextNodesIn(el) {
    return $(el).find(":not(iframe,pre,code)").andSelf().contents().filter(function() {
      return this.nodeType == 3;
    });
  }

  $.fn.smartquotes = function(fn) {
    if (!fn) fn = $.smartquotes;

    var nodes = getTextNodesIn(this);
    for (var i in nodes) {
      if (nodes.hasOwnProperty(i)) {
        var node = nodes[i];
        node.nodeValue = fn(node.nodeValue);
      }
    }
  };
})(jQuery);
