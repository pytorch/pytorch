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
const SEARCH_COOKIE_NAME = ''+'search_grp';

const searchResults = new SearchResults();

/* A class handling everything associated with the search panel.

   Parameters:
   name - The name of the global variable that will be
          storing this instance.  Is needed to be able to set timeouts.
   resultPath - path to use for external files
*/
function SearchBox(name, resultsPath, extension) {
  if (!name || !resultsPath) {  alert("Missing parameters to SearchBox."); }
  if (!extension || extension == "") { extension = ".html"; }

  function getXPos(item) {
    let x = 0;
    if (item.offsetWidth) {
      while (item && item!=document.body) {
        x   += item.offsetLeft;
        item = item.offsetParent;
      }
    }
    return x;
  }

  function getYPos(item) {
    let y = 0;
    if (item.offsetWidth) {
      while (item && item!=document.body) {
        y   += item.offsetTop;
        item = item.offsetParent;
      }
    }
    return y;
  }

  // ---------- Instance variables
  this.name                  = name;
  this.resultsPath           = resultsPath;
  this.keyTimeout            = 0;
  this.keyTimeoutLength      = 500;
  this.closeSelectionTimeout = 300;
  this.lastSearchValue       = "";
  this.lastResultsPage       = "";
  this.hideTimeout           = 0;
  this.searchIndex           = 0;
  this.searchActive          = false;
  this.extension             = extension;

  // ----------- DOM Elements

  this.DOMSearchField              = () => document.getElementById("MSearchField");
  this.DOMSearchSelect             = () => document.getElementById("MSearchSelect");
  this.DOMSearchSelectWindow       = () => document.getElementById("MSearchSelectWindow");
  this.DOMPopupSearchResults       = () => document.getElementById("MSearchResults");
  this.DOMPopupSearchResultsWindow = () => document.getElementById("MSearchResultsWindow");
  this.DOMSearchClose              = () => document.getElementById("MSearchClose");
  this.DOMSearchBox                = () => document.getElementById("MSearchBox");

  // ------------ Event Handlers

  // Called when focus is added or removed from the search field.
  this.OnSearchFieldFocus = function(isActive) {
    this.Activate(isActive);
  }

  this.OnSearchSelectShow = function() {
    const searchSelectWindow = this.DOMSearchSelectWindow();
    const searchField        = this.DOMSearchSelect();

    const left = getXPos(searchField);
    const top  = getYPos(searchField) + searchField.offsetHeight;

    // show search selection popup
    searchSelectWindow.style.display='block';
    searchSelectWindow.style.left =  left + 'px';
    searchSelectWindow.style.top  =  top  + 'px';

    // stop selection hide timer
    if (this.hideTimeout) {
      clearTimeout(this.hideTimeout);
      this.hideTimeout=0;
    }
    return false; // to avoid "image drag" default event
  }

  this.OnSearchSelectHide = function() {
    this.hideTimeout = setTimeout(this.CloseSelectionWindow.bind(this),
                                  this.closeSelectionTimeout);
  }

  // Called when the content of the search field is changed.
  this.OnSearchFieldChange = function(evt) {
    if (this.keyTimeout) { // kill running timer
      clearTimeout(this.keyTimeout);
      this.keyTimeout = 0;
    }

    const e = evt ? evt : window.event; // for IE
    if (e.keyCode==40 || e.keyCode==13) {
      if (e.shiftKey==1) {
        this.OnSearchSelectShow();
        const win=this.DOMSearchSelectWindow();
        for (let i=0;i<win.childNodes.length;i++) {
          const child = win.childNodes[i]; // get span within a
          if (child.className=='SelectItem') {
            child.focus();
            return;
          }
        }
        return;
      } else {
        const elem = searchResults.NavNext(0);
        if (elem) elem.focus();
      }
    } else if (e.keyCode==27) { // Escape out of the search field
      e.stopPropagation();
      this.DOMSearchField().blur();
      this.DOMPopupSearchResultsWindow().style.display = 'none';
      this.DOMSearchClose().style.display = 'none';
      this.lastSearchValue = '';
      this.Activate(false);
      return;
    }

    // strip whitespaces
    const searchValue = this.DOMSearchField().value.replace(/ +/g, "");

    if (searchValue != this.lastSearchValue) { // search value has changed
      if (searchValue != "") { // non-empty search
        // set timer for search update
        this.keyTimeout = setTimeout(this.Search.bind(this), this.keyTimeoutLength);
      } else { // empty search field
        this.DOMPopupSearchResultsWindow().style.display = 'none';
        this.DOMSearchClose().style.display = 'none';
        this.lastSearchValue = '';
      }
    }
  }

  this.SelectItemCount = function() {
    let count=0;
    const win=this.DOMSearchSelectWindow();
    for (let i=0;i<win.childNodes.length;i++) {
      const child = win.childNodes[i]; // get span within a
      if (child.className=='SelectItem') {
        count++;
      }
    }
    return count;
  }

  this.GetSelectionIdByName = function(name) {
    let j=0;
    const win=this.DOMSearchSelectWindow();
    for (let i=0;i<win.childNodes.length;i++) {
      const child = win.childNodes[i];
      if (child.className=='SelectItem') {
        if (child.childNodes[1].nodeValue==name) {
          return j;
        }
        j++;
      }
    }
    return 0;
  }

  this.SelectItemSet = function(id) {
    let j=0;
    const win=this.DOMSearchSelectWindow();
    for (let i=0;i<win.childNodes.length;i++) {
      const child = win.childNodes[i]; // get span within a
      if (child.className=='SelectItem') {
        const node = child.firstChild;
        if (j==id) {
          node.innerHTML='&#8226;';
          Cookie.writeSetting(SEARCH_COOKIE_NAME, child.childNodes[1].nodeValue, 0)
        } else {
          node.innerHTML='&#160;';
        }
        j++;
      }
    }
  }

  // Called when an search filter selection is made.
  // set item with index id as the active item
  this.OnSelectItem = function(id) {
    this.searchIndex = id;
    this.SelectItemSet(id);
    const searchValue = this.DOMSearchField().value.replace(/ +/g, "");
    if (searchValue!="" && this.searchActive) { // something was found -> do a search
      this.Search();
    }
  }

  this.OnSearchSelectKey = function(evt) {
    const e = (evt) ? evt : window.event; // for IE
    if (e.keyCode==40 && this.searchIndex<this.SelectItemCount()) { // Down
      this.searchIndex++;
      this.OnSelectItem(this.searchIndex);
    } else if (e.keyCode==38 && this.searchIndex>0) { // Up
      this.searchIndex--;
      this.OnSelectItem(this.searchIndex);
    } else if (e.keyCode==13 || e.keyCode==27) {
      e.stopPropagation();
      this.OnSelectItem(this.searchIndex);
      this.CloseSelectionWindow();
      this.DOMSearchField().focus();
    }
    return false;
  }

  // --------- Actions

  // Closes the results window.
  this.CloseResultsWindow = function() {
    this.DOMPopupSearchResultsWindow().style.display = 'none';
    this.DOMSearchClose().style.display = 'none';
    this.Activate(false);
  }

  this.CloseSelectionWindow = function() {
    this.DOMSearchSelectWindow().style.display = 'none';
  }

  // Performs a search.
  this.Search = function() {
    this.keyTimeout = 0;

    // strip leading whitespace
    const searchValue = this.DOMSearchField().value.replace(/^ +/, "");

    const code = searchValue.toLowerCase().charCodeAt(0);
    let idxChar = searchValue.substr(0, 1).toLowerCase();
    if ( 0xD800 <= code && code <= 0xDBFF && searchValue > 1) { // surrogate pair
      idxChar = searchValue.substr(0, 2);
    }

    let jsFile;
    let idx = indexSectionsWithContent[this.searchIndex].indexOf(idxChar);
    if (idx!=-1) {
      const hexCode=idx.toString(16);
      jsFile = this.resultsPath + indexSectionNames[this.searchIndex] + '_' + hexCode + '.js';
    }

    const loadJS = function(url, impl, loc) {
      const scriptTag = document.createElement('script');
      scriptTag.src = url;
      scriptTag.onload = impl;
      scriptTag.onreadystatechange = impl;
      loc.appendChild(scriptTag);
    }

    const domPopupSearchResultsWindow = this.DOMPopupSearchResultsWindow();
    const domSearchBox = this.DOMSearchBox();
    const domPopupSearchResults = this.DOMPopupSearchResults();
    const domSearchClose = this.DOMSearchClose();
    const resultsPath = this.resultsPath;

    const handleResults = function() {
      document.getElementById("Loading").style.display="none";
      if (typeof searchData !== 'undefined') {
        createResults(resultsPath);
        document.getElementById("NoMatches").style.display="none";
      }

      if (idx!=-1) {
        searchResults.Search(searchValue);
      } else { // no file with search results => force empty search results
        searchResults.Search('====');
      }

      if (domPopupSearchResultsWindow.style.display!='block') {
        domSearchClose.style.display = 'inline-block';
        let left = getXPos(domSearchBox) + 150;
        let top  = getYPos(domSearchBox) + 20;
        domPopupSearchResultsWindow.style.display = 'block';
        left -= domPopupSearchResults.offsetWidth;
        const maxWidth  = document.body.clientWidth;
        const maxHeight = document.body.clientHeight;
        let width = 300;
        if (left<10) left=10;
        if (width+left+8>maxWidth) width=maxWidth-left-8;
        let height = 400;
        if (height+top+8>maxHeight) height=maxHeight-top-8;
        domPopupSearchResultsWindow.style.top     = top  + 'px';
        domPopupSearchResultsWindow.style.left    = left + 'px';
        domPopupSearchResultsWindow.style.width   = width + 'px';
        domPopupSearchResultsWindow.style.height  = height + 'px';
      }
    }

    if (jsFile) {
      loadJS(jsFile, handleResults, this.DOMPopupSearchResultsWindow());
    } else {
      handleResults();
    }

    this.lastSearchValue = searchValue;
  }

  // -------- Activation Functions

  // Activates or deactivates the search panel, resetting things to
  // their default values if necessary.
  this.Activate = function(isActive) {
    if (isActive || // open it
      this.DOMPopupSearchResultsWindow().style.display == 'block'
    ) {
      this.DOMSearchBox().className = 'MSearchBoxActive';
      this.searchActive = true;
    } else if (!isActive) { // directly remove the panel
      this.DOMSearchBox().className = 'MSearchBoxInactive';
      this.searchActive             = false;
      this.lastSearchValue          = ''
      this.lastResultsPage          = '';
      this.DOMSearchField().value   = '';
    }
  }
}

// -----------------------------------------------------------------------

// The class that handles everything on the search results page.
function SearchResults() {

  function convertToId(search) {
    let result = '';
    for (let i=0;i<search.length;i++) {
      const c = search.charAt(i);
      const cn = c.charCodeAt(0);
      if (c.match(/[a-z0-9\u0080-\uFFFF]/)) {
        result+=c;
      } else if (cn<16) {
        result+="_0"+cn.toString(16);
      } else {
        result+="_"+cn.toString(16);
      }
    }
    return result;
  }

  // The number of matches from the last run of <Search()>.
  this.lastMatchCount = 0;
  this.lastKey = 0;
  this.repeatOn = false;

  // Toggles the visibility of the passed element ID.
  this.FindChildElement = function(id) {
    const parentElement = document.getElementById(id);
    let element = parentElement.firstChild;

    while (element && element!=parentElement) {
      if (element.nodeName.toLowerCase() == 'div' && element.className == 'SRChildren') {
        return element;
      }

      if (element.nodeName.toLowerCase() == 'div' && element.hasChildNodes()) {
        element = element.firstChild;
      } else if (element.nextSibling) {
        element = element.nextSibling;
      } else {
        do {
          element = element.parentNode;
        }
        while (element && element!=parentElement && !element.nextSibling);

        if (element && element!=parentElement) {
          element = element.nextSibling;
        }
      }
    }
  }

  this.Toggle = function(id) {
    const element = this.FindChildElement(id);
    if (element) {
      if (element.style.display == 'block') {
        element.style.display = 'none';
      } else {
        element.style.display = 'block';
      }
    }
  }

  // Searches for the passed string.  If there is no parameter,
  // it takes it from the URL query.
  //
  // Always returns true, since other documents may try to call it
  // and that may or may not be possible.
  this.Search = function(search) {
    if (!search) { // get search word from URL
      search = window.location.search;
      search = search.substring(1);  // Remove the leading '?'
      search = unescape(search);
    }

    search = search.replace(/^ +/, ""); // strip leading spaces
    search = search.replace(/ +$/, ""); // strip trailing spaces
    search = search.toLowerCase();
    search = convertToId(search);

    const resultRows = document.getElementsByTagName("div");
    let matches = 0;

    let i = 0;
    while (i < resultRows.length) {
      const row = resultRows.item(i);
      if (row.className == "SRResult") {
        let rowMatchName = row.id.toLowerCase();
        rowMatchName = rowMatchName.replace(/^sr\d*_/, ''); // strip 'sr123_'

        if (search.length<=rowMatchName.length &&
          rowMatchName.substr(0, search.length)==search) {
          row.style.display = 'block';
          matches++;
        } else {
          row.style.display = 'none';
        }
      }
      i++;
    }
    document.getElementById("Searching").style.display='none';
    if (matches == 0) { // no results
      document.getElementById("NoMatches").style.display='block';
    } else { // at least one result
      document.getElementById("NoMatches").style.display='none';
    }
    this.lastMatchCount = matches;
    return true;
  }

  // return the first item with index index or higher that is visible
  this.NavNext = function(index) {
    let focusItem;
    for (;;) {
      const focusName = 'Item'+index;
      focusItem = document.getElementById(focusName);
      if (focusItem && focusItem.parentNode.parentNode.style.display=='block') {
        break;
      } else if (!focusItem) { // last element
        break;
      }
      focusItem=null;
      index++;
    }
    return focusItem;
  }

  this.NavPrev = function(index) {
    let focusItem;
    for (;;) {
      const focusName = 'Item'+index;
      focusItem = document.getElementById(focusName);
      if (focusItem && focusItem.parentNode.parentNode.style.display=='block') {
        break;
      } else if (!focusItem) { // last element
        break;
      }
      focusItem=null;
      index--;
    }
    return focusItem;
  }

  this.ProcessKeys = function(e) {
    if (e.type == "keydown") {
      this.repeatOn = false;
      this.lastKey = e.keyCode;
    } else if (e.type == "keypress") {
      if (!this.repeatOn) {
        if (this.lastKey) this.repeatOn = true;
        return false; // ignore first keypress after keydown
      }
    } else if (e.type == "keyup") {
      this.lastKey = 0;
      this.repeatOn = false;
    }
    return this.lastKey!=0;
  }

  this.Nav = function(evt,itemIndex) {
    const e  = (evt) ? evt : window.event; // for IE
    if (e.keyCode==13) return true;
    if (!this.ProcessKeys(e)) return false;

    if (this.lastKey==38) { // Up
      const newIndex = itemIndex-1;
      let focusItem = this.NavPrev(newIndex);
      if (focusItem) {
        let child = this.FindChildElement(focusItem.parentNode.parentNode.id);
        if (child && child.style.display == 'block') { // children visible
          let n=0;
          let tmpElem;
          for (;;) { // search for last child
            tmpElem = document.getElementById('Item'+newIndex+'_c'+n);
            if (tmpElem) {
              focusItem = tmpElem;
            } else { // found it!
              break;
            }
            n++;
          }
        }
      }
      if (focusItem) {
        focusItem.focus();
      } else { // return focus to search field
        document.getElementById("MSearchField").focus();
      }
    } else if (this.lastKey==40) { // Down
      const newIndex = itemIndex+1;
      let focusItem;
      const item = document.getElementById('Item'+itemIndex);
      const elem = this.FindChildElement(item.parentNode.parentNode.id);
      if (elem && elem.style.display == 'block') { // children visible
        focusItem = document.getElementById('Item'+itemIndex+'_c0');
      }
      if (!focusItem) focusItem = this.NavNext(newIndex);
      if (focusItem)  focusItem.focus();
    } else if (this.lastKey==39) { // Right
      const item = document.getElementById('Item'+itemIndex);
      const elem = this.FindChildElement(item.parentNode.parentNode.id);
      if (elem) elem.style.display = 'block';
    } else if (this.lastKey==37) { // Left
      const item = document.getElementById('Item'+itemIndex);
      const elem = this.FindChildElement(item.parentNode.parentNode.id);
      if (elem) elem.style.display = 'none';
    } else if (this.lastKey==27) { // Escape
      e.stopPropagation();
      searchBox.CloseResultsWindow();
      document.getElementById("MSearchField").focus();
    } else if (this.lastKey==13) { // Enter
      return true;
    }
    return false;
  }

  this.NavChild = function(evt,itemIndex,childIndex) {
    const e  = (evt) ? evt : window.event; // for IE
    if (e.keyCode==13) return true;
    if (!this.ProcessKeys(e)) return false;

    if (this.lastKey==38) { // Up
      if (childIndex>0) {
        const newIndex = childIndex-1;
        document.getElementById('Item'+itemIndex+'_c'+newIndex).focus();
      } else { // already at first child, jump to parent
        document.getElementById('Item'+itemIndex).focus();
      }
    } else if (this.lastKey==40) { // Down
      const newIndex = childIndex+1;
      let elem = document.getElementById('Item'+itemIndex+'_c'+newIndex);
      if (!elem) { // last child, jump to parent next parent
        elem = this.NavNext(itemIndex+1);
      }
      if (elem) {
        elem.focus();
      }
    } else if (this.lastKey==27) { // Escape
      e.stopPropagation();
      searchBox.CloseResultsWindow();
      document.getElementById("MSearchField").focus();
    } else if (this.lastKey==13) { // Enter
      return true;
    }
    return false;
  }
}

function createResults(resultsPath) {

  function setKeyActions(elem,action) {
    elem.setAttribute('onkeydown',action);
    elem.setAttribute('onkeypress',action);
    elem.setAttribute('onkeyup',action);
  }

  function setClassAttr(elem,attr) {
    elem.setAttribute('class',attr);
    elem.setAttribute('className',attr);
  }

  const decodeHtml = (html) => {
    const txt = document.createElement("textarea");
    txt.innerHTML = html;
    return txt.value;
  };

  const results = document.getElementById("SRResults");
  results.innerHTML = '';
  searchData.forEach((elem,index) => {
    const id = elem[0];
    const srResult = document.createElement('div');
    srResult.setAttribute('id','SR_'+id);
    setClassAttr(srResult,'SRResult');
    const srEntry = document.createElement('div');
    setClassAttr(srEntry,'SREntry');
    const srLink = document.createElement('a');
    srLink.setAttribute('id','Item'+index);
    setKeyActions(srLink,'return searchResults.Nav(event,'+index+')');
    setClassAttr(srLink,'SRSymbol');
    srLink.innerHTML = decodeHtml(elem[1][0]);
    srEntry.appendChild(srLink);
    if (elem[1].length==2) { // single result
      if (elem[1][1][0].startsWith('http://') || elem[1][1][0].startsWith('https://')) { // absolute path
        srLink.setAttribute('href',elem[1][1][0]);
      } else { // relative path
        srLink.setAttribute('href',resultsPath+elem[1][1][0]);
      }
      srLink.setAttribute('onclick','searchBox.CloseResultsWindow()');
      if (elem[1][1][1]) {
       srLink.setAttribute('target','_parent');
      } else {
       srLink.setAttribute('target','_blank');
      }
      const srScope = document.createElement('span');
      setClassAttr(srScope,'SRScope');
      srScope.innerHTML = decodeHtml(elem[1][1][2]);
      srEntry.appendChild(srScope);
    } else { // multiple results
      srLink.setAttribute('href','javascript:searchResults.Toggle("SR_'+id+'")');
      const srChildren = document.createElement('div');
      setClassAttr(srChildren,'SRChildren');
      for (let c=0; c<elem[1].length-1; c++) {
        const srChild = document.createElement('a');
        srChild.setAttribute('id','Item'+index+'_c'+c);
        setKeyActions(srChild,'return searchResults.NavChild(event,'+index+','+c+')');
        setClassAttr(srChild,'SRScope');
        if (elem[1][c+1][0].startsWith('http://') || elem[1][c+1][0].startsWith('https://')) { // absolute path
          srChild.setAttribute('href',elem[1][c+1][0]);
        } else { // relative path
          srChild.setAttribute('href',resultsPath+elem[1][c+1][0]);
        }
        srChild.setAttribute('onclick','searchBox.CloseResultsWindow()');
        if (elem[1][c+1][1]) {
         srChild.setAttribute('target','_parent');
        } else {
         srChild.setAttribute('target','_blank');
        }
        srChild.innerHTML = decodeHtml(elem[1][c+1][2]);
        srChildren.appendChild(srChild);
      }
      srEntry.appendChild(srChildren);
    }
    srResult.appendChild(srEntry);
    results.appendChild(srResult);
  });
}

function init_search() {
  const results = document.getElementById("MSearchSelectWindow");

  results.tabIndex=0;
  for (let key in indexSectionLabels) {
    const link = document.createElement('a');
    link.setAttribute('class','SelectItem');
    link.setAttribute('onclick','searchBox.OnSelectItem('+key+')');
    link.href='javascript:void(0)';
    link.innerHTML='<span class="SelectionMark">&#160;</span>'+indexSectionLabels[key];
    results.appendChild(link);
  }

  const input = document.getElementById("MSearchSelect");
  const searchSelectWindow = document.getElementById("MSearchSelectWindow");
  input.tabIndex=0;
  input.addEventListener("keydown", function(event) {
    if (event.keyCode==13 || event.keyCode==40) {
      event.preventDefault();
      if (searchSelectWindow.style.display == 'block') {
        searchBox.CloseSelectionWindow();
      } else {
        searchBox.OnSearchSelectShow();
        searchBox.DOMSearchSelectWindow().focus();
      }
    }
  });
  const name = Cookie.readSetting(SEARCH_COOKIE_NAME,0);
  const id = searchBox.GetSelectionIdByName(name);
  searchBox.OnSelectItem(id);
}
/* @license-end */
