/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
 */

function toggleVisibility(linkObj) {
  return dynsection.toggleVisibility(linkObj);
}

let dynsection = {
  // helper function
  updateStripes : function() {
    $('table.directory tr').
      removeClass('even').filter(':visible:even').addClass('even');
    $('table.directory tr').
      removeClass('odd').filter(':visible:odd').addClass('odd');
  },

  toggleVisibility : function(linkObj) {
    const base = $(linkObj).attr('id');
    const summary = $('#'+base+'-summary');
    const content = $('#'+base+'-content');
    const trigger = $('#'+base+'-trigger');
    const src=$(trigger).attr('src');
    if (content.is(':visible')===true) {
      content.slideUp('fast');
      summary.show();
      $(linkObj).find('.arrowhead').addClass('closed').removeClass('opened');
    } else {
      content.slideDown('fast');
      summary.hide();
      $(linkObj).find('.arrowhead').removeClass('closed').addClass('opened');
    }
    return false;
  },

  toggleLevel : function(level) {
    $('table.directory tr').each(function() {
      const l = this.id.split('_').length-1;
      const i = $('#img'+this.id.substring(3));
      const a = $('#arr'+this.id.substring(3));
      if (l<level+1) {
        i.find('.folder-icon').addClass('open');
        a.find('.arrowhead').removeClass('closed').addClass('opened');
        $(this).show();
      } else if (l==level+1) {
        a.find('.arrowhead').removeClass('opened').addClass('closed');
        i.find('.folder-icon').removeClass('open');
        $(this).show();
      } else {
        $(this).hide();
      }
    });
    this.updateStripes();
  },

  toggleFolder : function(id) {
    // the clicked row
    const currentRow = $('#row_'+id);

    // all rows after the clicked row
    const rows = currentRow.nextAll("tr");

    const re = new RegExp('^row_'+id+'\\d+_$', "i"); //only one sub

    // only match elements AFTER this one (can't hide elements before)
    const childRows = rows.filter(function() { return this.id.match(re); });

    // first row is visible we are HIDING
    if (childRows.filter(':first').is(':visible')===true) {
      // replace down arrow by right arrow for current row
      const currentRowSpans = currentRow.find("span");
      currentRowSpans.filter(".iconfolder").find('.folder-icon').removeClass("open");
      currentRowSpans.filter(".opened").removeClass("opened").addClass("closed");
      rows.filter("[id^=row_"+id+"]").hide(); // hide all children
    } else { // we are SHOWING
      // replace right arrow by down arrow for current row
      const currentRowSpans = currentRow.find("span");
      currentRowSpans.filter(".iconfolder").find('.folder-icon').addClass("open");
      currentRowSpans.filter(".closed").removeClass("closed").addClass("opened");
      // replace down arrows by right arrows for child rows
      const childRowsSpans = childRows.find("span");
      childRowsSpans.filter(".iconfolder").find('.folder-icon').removeClass("open");
      childRowsSpans.filter(".opened").removeClass("opened").addClass("closed");
      childRows.show(); //show all children
    }
    this.updateStripes();
  },

  toggleInherit : function(id) {
    let rows = $('tr.inherit.'+id);
    let header = $('tr.inherit_header.'+id);
    if (rows.filter(':first').is(':visible')===true) {
      rows.hide();
      $(header).find('.arrowhead').addClass('closed').removeClass('opened');
    } else {
      rows.show();
      $(header).find('.arrowhead').removeClass('closed').addClass('opened');
    }
  },

};

let codefold = {
  opened : true,

  // toggle all folding blocks
  toggle_all : function() {
    if (this.opened) {
      $('#fold_all').addClass('plus').removeClass('minus');
      $('div[id^=foldopen]').hide();
      $('div[id^=foldclosed]').show();
      $('div[id^=foldclosed] span.fold').removeClass('minus').addClass('plus');
    } else {
      $('#fold_all').addClass('minus').removeClass('plus');
      $('div[id^=foldopen]').show();
      $('div[id^=foldclosed]').hide();
    }
    this.opened=!this.opened;
  },

  // toggle single folding block
  toggle : function(id) {
    $('#foldopen'+id).toggle();
    $('#foldclosed'+id).toggle();
    $('#foldopen'+id).next().find('span.fold').addClass('plus').removeClass('minus');
  },

  init : function() {
    $('span[class=lineno]').css({
      'padding-right':'4px',
      'margin-right':'2px',
      'display':'inline-block',
      'width':'54px',
      'background':'linear-gradient(var(--fold-line-color),var(--fold-line-color)) no-repeat 46px/2px 100%'
    });
    // add global toggle to first line
    $('span[class=lineno]:first').append('<span class="fold minus" id="fold_all" '+
      'onclick="javascript:codefold.toggle_all();"></span>');
    // add vertical lines to other rows
    $('span[class=lineno]').not(':eq(0)').append('<span class="fold"></span>');
    // add toggle controls to lines with fold divs
    $('div[class=foldopen]').each(function() {
      // extract specific id to use
      const id    = $(this).attr('id').replace('foldopen','');
      // extract start and end foldable fragment attributes
      const start = $(this).attr('data-start');
      const end   = $(this).attr('data-end');
      // replace normal fold span with controls for the first line of a foldable fragment
      $(this).find('span[class=fold]:first').replaceWith('<span class="fold minus" '+
                   'onclick="javascript:codefold.toggle(\''+id+'\');"></span>');
      // append div for folded (closed) representation
      $(this).after('<div id="foldclosed'+id+'" class="foldclosed" style="display:none;"></div>');
      // extract the first line from the "open" section to represent closed content
      const line = $(this).children().first().clone();
      // remove any glow that might still be active on the original line
      $(line).removeClass('glow');
      if (start) {
        // if line already ends with a start marker (e.g. trailing {), remove it
        $(line).html($(line).html().replace(new RegExp('\\s*'+start+'\\s*$','g'),''));
      }
      // replace minus with plus symbol
      $(line).find('span[class=fold]').addClass('plus').removeClass('minus');
      // append ellipsis
      $(line).append(' '+start+'<a href="javascript:codefold.toggle(\''+id+'\')">&#8230;</a>'+end);
      // insert constructed line into closed div
      $('#foldclosed'+id).html(line);
    });
  },
};
/* @license-end */
