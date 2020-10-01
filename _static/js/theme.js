require=(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
window.utilities = {
  scrollTop: function() {
    var supportPageOffset = window.pageXOffset !== undefined;
    var isCSS1Compat = ((document.compatMode || "") === "CSS1Compat");
    var scrollLeft = supportPageOffset ? window.pageXOffset : isCSS1Compat ? document.documentElement.scrollLeft : document.body.scrollLeft;
    return supportPageOffset ? window.pageYOffset : isCSS1Compat ? document.documentElement.scrollTop : document.body.scrollTop;
  },

  // Modified from https://stackoverflow.com/a/27078401
  throttle: function(func, wait, options) {
    var context, args, result;
    var timeout = null;
    var previous = 0;
    if (!options) options = {};
    var later = function() {
      previous = options.leading === false ? 0 : Date.now();
      timeout = null;
      result = func.apply(context, args);
      if (!timeout) context = args = null;
    };
    return function() {
      var now = Date.now();
      if (!previous && options.leading === false) previous = now;
      var remaining = wait - (now - previous);
      context = this;
      args = arguments;
      if (remaining <= 0 || remaining > wait) {
        if (timeout) {
          clearTimeout(timeout);
          timeout = null;
        }
        previous = now;
        result = func.apply(context, args);
        if (!timeout) context = args = null;
      } else if (!timeout && options.trailing !== false) {
        timeout = setTimeout(later, remaining);
      }
      return result;
    };
  },

  closest: function (el, selector) {
    var matchesFn;

    // find vendor prefix
    ['matches','webkitMatchesSelector','mozMatchesSelector','msMatchesSelector','oMatchesSelector'].some(function(fn) {
      if (typeof document.body[fn] == 'function') {
        matchesFn = fn;
        return true;
      }
      return false;
    });

    var parent;

    // traverse parents
    while (el) {
      parent = el.parentElement;
      if (parent && parent[matchesFn](selector)) {
        return parent;
      }
      el = parent;
    }

    return null;
  },

  // Modified from https://stackoverflow.com/a/18953277
  offset: function(elem) {
    if (!elem) {
      return;
    }

    rect = elem.getBoundingClientRect();

    // Make sure element is not hidden (display: none) or disconnected
    if (rect.width || rect.height || elem.getClientRects().length) {
      var doc = elem.ownerDocument;
      var docElem = doc.documentElement;

      return {
        top: rect.top + window.pageYOffset - docElem.clientTop,
        left: rect.left + window.pageXOffset - docElem.clientLeft
      };
    }
  },

  headersHeight: function() {
    if (document.getElementById("pytorch-left-menu").classList.contains("make-fixed")) {
      return document.getElementById("pytorch-page-level-bar").offsetHeight;
    } else {
      return document.getElementById("header-holder").offsetHeight +
             document.getElementById("pytorch-page-level-bar").offsetHeight;
    }
  },

  windowHeight: function() {
    return window.innerHeight ||
           document.documentElement.clientHeight ||
           document.body.clientHeight;
  }
}

},{}],2:[function(require,module,exports){
var cookieBanner = {
  init: function() {
    cookieBanner.bind();

    var cookieExists = cookieBanner.cookieExists();

    if (!cookieExists) {
      cookieBanner.setCookie();
      cookieBanner.showCookieNotice();
    }
  },

  bind: function() {
    $(".close-button").on("click", cookieBanner.hideCookieNotice);
  },

  cookieExists: function() {
    var cookie = localStorage.getItem("returningPytorchUser");

    if (cookie) {
      return true;
    } else {
      return false;
    }
  },

  setCookie: function() {
    localStorage.setItem("returningPytorchUser", true);
  },

  showCookieNotice: function() {
    $(".cookie-banner-wrapper").addClass("is-visible");
  },

  hideCookieNotice: function() {
    $(".cookie-banner-wrapper").removeClass("is-visible");
  }
};

$(function() {
  cookieBanner.init();
});

},{}],3:[function(require,module,exports){
window.filterTags = {
  bind: function() {
    var options = {
      valueNames: [{ data: ["tags"] }],
      page: "6",
      pagination: true
    };

    var tutorialList = new List("tutorial-cards", options);

    function filterSelectedTags(cardTags, selectedTags) {
      return cardTags.some(function(tag) {
        return selectedTags.some(function(selectedTag) {
          return selectedTag == tag;
        });
      });
    }

    function updateList() {
      var selectedTags = [];

      $(".selected").each(function() {
        selectedTags.push($(this).data("tag"));
      });

      tutorialList.filter(function(item) {
        var cardTags;

        if (item.values().tags == null) {
          cardTags = [""];
        } else {
          cardTags = item.values().tags.split(",");
        }

        if (selectedTags.length == 0) {
          return true;
        } else {
          return filterSelectedTags(cardTags, selectedTags);
        }
      });
    }

    $(".filter-btn").on("click", function() {
      if ($(this).data("tag") == "all") {
        $(this).addClass("all-tag-selected");
        $(".filter").removeClass("selected");
      } else {
        $(this).toggleClass("selected");
        $("[data-tag='all']").removeClass("all-tag-selected");
      }

      // If no tags are selected then highlight the 'All' tag

      if (!$(".selected")[0]) {
        $("[data-tag='all']").addClass("all-tag-selected");
      }

      updateList();
    });
  }
};

},{}],4:[function(require,module,exports){
// Modified from https://stackoverflow.com/a/32396543
window.highlightNavigation = {
  navigationListItems: document.querySelectorAll("#pytorch-right-menu li"),
  sections: document.querySelectorAll(".pytorch-article .section"),
  sectionIdTonavigationLink: {},

  bind: function() {
    if (!sideMenus.displayRightMenu) {
      return;
    };

    for (var i = 0; i < highlightNavigation.sections.length; i++) {
      var id = highlightNavigation.sections[i].id;
      highlightNavigation.sectionIdTonavigationLink[id] =
        document.querySelectorAll('#pytorch-right-menu li a[href="#' + id + '"]')[0];
    }

    $(window).scroll(utilities.throttle(highlightNavigation.highlight, 100));
  },

  highlight: function() {
    var rightMenu = document.getElementById("pytorch-right-menu");

    // If right menu is not on the screen don't bother
    if (rightMenu.offsetWidth === 0 && rightMenu.offsetHeight === 0) {
      return;
    }

    var scrollPosition = utilities.scrollTop();
    var OFFSET_TOP_PADDING = 25;
    var offset = document.getElementById("header-holder").offsetHeight +
                 document.getElementById("pytorch-page-level-bar").offsetHeight +
                 OFFSET_TOP_PADDING;

    var sections = highlightNavigation.sections;

    for (var i = (sections.length - 1); i >= 0; i--) {
      var currentSection = sections[i];
      var sectionTop = utilities.offset(currentSection).top;

      if (scrollPosition >= sectionTop - offset) {
        var navigationLink = highlightNavigation.sectionIdTonavigationLink[currentSection.id];
        var navigationListItem = utilities.closest(navigationLink, "li");

        if (navigationListItem && !navigationListItem.classList.contains("active")) {
          for (var i = 0; i < highlightNavigation.navigationListItems.length; i++) {
            var el = highlightNavigation.navigationListItems[i];
            if (el.classList.contains("active")) {
              el.classList.remove("active");
            }
          }

          navigationListItem.classList.add("active");

          // Scroll to active item. Not a requested feature but we could revive it. Needs work.

          // var menuTop = $("#pytorch-right-menu").position().top;
          // var itemTop = navigationListItem.getBoundingClientRect().top;
          // var TOP_PADDING = 20
          // var newActiveTop = $("#pytorch-side-scroll-right").scrollTop() + itemTop - menuTop - TOP_PADDING;

          // $("#pytorch-side-scroll-right").animate({
          //   scrollTop: newActiveTop
          // }, 100);
        }

        break;
      }
    }
  }
};

},{}],5:[function(require,module,exports){
window.mainMenuDropdown = {
  bind: function() {
    $("[data-toggle='ecosystem-dropdown']").on("click", function() {
      toggleDropdown($(this).attr("data-toggle"));
    });

    $("[data-toggle='resources-dropdown']").on("click", function() {
      toggleDropdown($(this).attr("data-toggle"));
    });

    function toggleDropdown(menuToggle) {
      var showMenuClass = "show-menu";
      var menuClass = "." + menuToggle + "-menu";

      if ($(menuClass).hasClass(showMenuClass)) {
        $(menuClass).removeClass(showMenuClass);
      } else {
        $("[data-toggle=" + menuToggle + "].show-menu").removeClass(
          showMenuClass
        );
        $(menuClass).addClass(showMenuClass);
      }
    }
  }
};

},{}],6:[function(require,module,exports){
window.mobileMenu = {
  bind: function() {
    $("[data-behavior='open-mobile-menu']").on('click', function(e) {
      e.preventDefault();
      $(".mobile-main-menu").addClass("open");
      $("body").addClass('no-scroll');

      mobileMenu.listenForResize();
    });

    $("[data-behavior='close-mobile-menu']").on('click', function(e) {
      e.preventDefault();
      mobileMenu.close();
    });
  },

  listenForResize: function() {
    $(window).on('resize.ForMobileMenu', function() {
      if ($(this).width() > 768) {
        mobileMenu.close();
      }
    });
  },

  close: function() {
    $(".mobile-main-menu").removeClass("open");
    $("body").removeClass('no-scroll');
    $(window).off('resize.ForMobileMenu');
  }
};

},{}],7:[function(require,module,exports){
window.mobileTOC = {
  bind: function() {
    $("[data-behavior='toggle-table-of-contents']").on("click", function(e) {
      e.preventDefault();

      var $parent = $(this).parent();

      if ($parent.hasClass("is-open")) {
        $parent.removeClass("is-open");
        $(".pytorch-left-menu").slideUp(200, function() {
          $(this).css({display: ""});
        });
      } else {
        $parent.addClass("is-open");
        $(".pytorch-left-menu").slideDown(200);
      }
    });
  }
}

},{}],8:[function(require,module,exports){
window.pytorchAnchors = {
  bind: function() {
    // Replace Sphinx-generated anchors with anchorjs ones
    $(".headerlink").text("");

    window.anchors.add(".pytorch-article .headerlink");

    $(".anchorjs-link").each(function() {
      var $headerLink = $(this).closest(".headerlink");
      var href = $headerLink.attr("href");
      var clone = this.outerHTML;

      $clone = $(clone).attr("href", href);
      $headerLink.before($clone);
      $headerLink.remove();
    });
  }
};

},{}],9:[function(require,module,exports){
// Modified from https://stackoverflow.com/a/13067009
// Going for a JS solution to scrolling to an anchor so we can benefit from
// less hacky css and smooth scrolling.

window.scrollToAnchor = {
  bind: function() {
    var document = window.document;
    var history = window.history;
    var location = window.location
    var HISTORY_SUPPORT = !!(history && history.pushState);

    var anchorScrolls = {
      ANCHOR_REGEX: /^#[^ ]+$/,
      offsetHeightPx: function() {
        var OFFSET_HEIGHT_PADDING = 20;
        // TODO: this is a little janky. We should try to not rely on JS for this
        return utilities.headersHeight() + OFFSET_HEIGHT_PADDING;
      },

      /**
       * Establish events, and fix initial scroll position if a hash is provided.
       */
      init: function() {
        this.scrollToCurrent();
        // This interferes with clicks below it, causing a double fire
        // $(window).on('hashchange', $.proxy(this, 'scrollToCurrent'));
        $('body').on('click', 'a', $.proxy(this, 'delegateAnchors'));
        $('body').on('click', '#pytorch-right-menu li span', $.proxy(this, 'delegateSpans'));
      },

      /**
       * Return the offset amount to deduct from the normal scroll position.
       * Modify as appropriate to allow for dynamic calculations
       */
      getFixedOffset: function() {
        return this.offsetHeightPx();
      },

      /**
       * If the provided href is an anchor which resolves to an element on the
       * page, scroll to it.
       * @param  {String} href
       * @return {Boolean} - Was the href an anchor.
       */
      scrollIfAnchor: function(href, pushToHistory) {
        var match, anchorOffset;

        if(!this.ANCHOR_REGEX.test(href)) {
          return false;
        }

        match = document.getElementById(href.slice(1));

        if(match) {
          var anchorOffset = $(match).offset().top - this.getFixedOffset();

          $('html, body').scrollTop(anchorOffset);

          // Add the state to history as-per normal anchor links
          if(HISTORY_SUPPORT && pushToHistory) {
            history.pushState({}, document.title, location.pathname + href);
          }
        }

        return !!match;
      },

      /**
       * Attempt to scroll to the current location's hash.
       */
      scrollToCurrent: function(e) {
        if(this.scrollIfAnchor(window.location.hash) && e) {
          e.preventDefault();
        }
      },

      delegateSpans: function(e) {
        var elem = utilities.closest(e.target, "a");

        if(this.scrollIfAnchor(elem.getAttribute('href'), true)) {
          e.preventDefault();
        }
      },

      /**
       * If the click event's target was an anchor, fix the scroll position.
       */
      delegateAnchors: function(e) {
        var elem = e.target;

        if(this.scrollIfAnchor(elem.getAttribute('href'), true)) {
          e.preventDefault();
        }
      }
    };

    $(document).ready($.proxy(anchorScrolls, 'init'));
  }
};

},{}],10:[function(require,module,exports){
window.sideMenus = {
  rightMenuIsOnScreen: function() {
    return document.getElementById("pytorch-content-right").offsetParent !== null;
  },

  isFixedToBottom: false,

  bind: function() {
    sideMenus.handleLeftMenu();

    var rightMenuLinks = document.querySelectorAll("#pytorch-right-menu li");
    var rightMenuHasLinks = rightMenuLinks.length > 1;

    if (!rightMenuHasLinks) {
      for (var i = 0; i < rightMenuLinks.length; i++) {
        rightMenuLinks[i].style.display = "none";
      }
    }

    if (rightMenuHasLinks) {
      // Don't show the Shortcuts menu title text unless there are menu items
      document.getElementById("pytorch-shortcuts-wrapper").style.display = "block";

      // We are hiding the titles of the pages in the right side menu but there are a few
      // pages that include other pages in the right side menu (see 'torch.nn' in the docs)
      // so if we exclude those it looks confusing. Here we add a 'title-link' class to these
      // links so we can exclude them from normal right side menu link operations
      var titleLinks = document.querySelectorAll(
        "#pytorch-right-menu #pytorch-side-scroll-right \
         > ul > li > a.reference.internal"
      );

      for (var i = 0; i < titleLinks.length; i++) {
        var link = titleLinks[i];

        link.classList.add("title-link");

        if (
          link.nextElementSibling &&
          link.nextElementSibling.tagName === "UL" &&
          link.nextElementSibling.children.length > 0
        ) {
          link.classList.add("has-children");
        }
      }

      // Add + expansion signifiers to normal right menu links that have sub menus
      var menuLinks = document.querySelectorAll(
        "#pytorch-right-menu ul li ul li a.reference.internal"
      );

      for (var i = 0; i < menuLinks.length; i++) {
        if (
          menuLinks[i].nextElementSibling &&
          menuLinks[i].nextElementSibling.tagName === "UL"
        ) {
          menuLinks[i].classList.add("not-expanded");
        }
      }

      // If a hash is present on page load recursively expand menu items leading to selected item
      var linkWithHash =
        document.querySelector(
          "#pytorch-right-menu a[href=\"" + window.location.hash + "\"]"
        );

      if (linkWithHash) {
        // Expand immediate sibling list if present
        if (
          linkWithHash.nextElementSibling &&
          linkWithHash.nextElementSibling.tagName === "UL" &&
          linkWithHash.nextElementSibling.children.length > 0
        ) {
          linkWithHash.nextElementSibling.style.display = "block";
          linkWithHash.classList.add("expanded");
        }

        // Expand ancestor lists if any
        sideMenus.expandClosestUnexpandedParentList(linkWithHash);
      }

      // Bind click events on right menu links
      $("#pytorch-right-menu a.reference.internal").on("click", function() {
        if (this.classList.contains("expanded")) {
          this.nextElementSibling.style.display = "none";
          this.classList.remove("expanded");
          this.classList.add("not-expanded");
        } else if (this.classList.contains("not-expanded")) {
          this.nextElementSibling.style.display = "block";
          this.classList.remove("not-expanded");
          this.classList.add("expanded");
        }
      });

      sideMenus.handleRightMenu();
    }

    $(window).on('resize scroll', function(e) {
      sideMenus.handleNavBar();

      sideMenus.handleLeftMenu();

      if (sideMenus.rightMenuIsOnScreen()) {
        sideMenus.handleRightMenu();
      }
    });
  },

  leftMenuIsFixed: function() {
    return document.getElementById("pytorch-left-menu").classList.contains("make-fixed");
  },

  handleNavBar: function() {
    var mainHeaderHeight = document.getElementById('header-holder').offsetHeight;

    // If we are scrolled past the main navigation header fix the sub menu bar to top of page
    if (utilities.scrollTop() >= mainHeaderHeight) {
      document.getElementById("pytorch-left-menu").classList.add("make-fixed");
      document.getElementById("pytorch-page-level-bar").classList.add("left-menu-is-fixed");
    } else {
      document.getElementById("pytorch-left-menu").classList.remove("make-fixed");
      document.getElementById("pytorch-page-level-bar").classList.remove("left-menu-is-fixed");
    }
  },

  expandClosestUnexpandedParentList: function (el) {
    var closestParentList = utilities.closest(el, "ul");

    if (closestParentList) {
      var closestParentLink = closestParentList.previousElementSibling;
      var closestParentLinkExists = closestParentLink &&
                                    closestParentLink.tagName === "A" &&
                                    closestParentLink.classList.contains("reference");

      if (closestParentLinkExists) {
        // Don't add expansion class to any title links
         if (closestParentLink.classList.contains("title-link")) {
           return;
         }

        closestParentList.style.display = "block";
        closestParentLink.classList.remove("not-expanded");
        closestParentLink.classList.add("expanded");
        sideMenus.expandClosestUnexpandedParentList(closestParentLink);
      }
    }
  },

  handleLeftMenu: function () {
    var windowHeight = utilities.windowHeight();
    var topOfFooterRelativeToWindow = document.getElementById("docs-tutorials-resources").getBoundingClientRect().top;

    if (topOfFooterRelativeToWindow >= windowHeight) {
      document.getElementById("pytorch-left-menu").style.height = "100%";
    } else {
      var howManyPixelsOfTheFooterAreInTheWindow = windowHeight - topOfFooterRelativeToWindow;
      var leftMenuDifference = howManyPixelsOfTheFooterAreInTheWindow;
      document.getElementById("pytorch-left-menu").style.height = (windowHeight - leftMenuDifference) + "px";
    }
  },

  handleRightMenu: function() {
    var rightMenuWrapper = document.getElementById("pytorch-content-right");
    var rightMenu = document.getElementById("pytorch-right-menu");
    var rightMenuList = rightMenu.getElementsByTagName("ul")[0];
    var article = document.getElementById("pytorch-article");
    var articleHeight = article.offsetHeight;
    var articleBottom = utilities.offset(article).top + articleHeight;
    var mainHeaderHeight = document.getElementById('header-holder').offsetHeight;

    if (utilities.scrollTop() < mainHeaderHeight) {
      rightMenuWrapper.style.height = "100%";
      rightMenu.style.top = 0;
      rightMenu.classList.remove("scrolling-fixed");
      rightMenu.classList.remove("scrolling-absolute");
    } else {
      if (rightMenu.classList.contains("scrolling-fixed")) {
        var rightMenuBottom =
          utilities.offset(rightMenuList).top + rightMenuList.offsetHeight;

        if (rightMenuBottom >= articleBottom) {
          rightMenuWrapper.style.height = articleHeight + mainHeaderHeight + "px";
          rightMenu.style.top = utilities.scrollTop() - mainHeaderHeight + "px";
          rightMenu.classList.add("scrolling-absolute");
          rightMenu.classList.remove("scrolling-fixed");
        }
      } else {
        rightMenuWrapper.style.height = articleHeight + mainHeaderHeight + "px";
        rightMenu.style.top =
          articleBottom - mainHeaderHeight - rightMenuList.offsetHeight + "px";
        rightMenu.classList.add("scrolling-absolute");
      }

      if (utilities.scrollTop() < articleBottom - rightMenuList.offsetHeight) {
        rightMenuWrapper.style.height = "100%";
        rightMenu.style.top = "";
        rightMenu.classList.remove("scrolling-absolute");
        rightMenu.classList.add("scrolling-fixed");
      }
    }

    var rightMenuSideScroll = document.getElementById("pytorch-side-scroll-right");
    var sideScrollFromWindowTop = rightMenuSideScroll.getBoundingClientRect().top;

    rightMenuSideScroll.style.height = utilities.windowHeight() - sideScrollFromWindowTop + "px";
  }
};

},{}],"pytorch-sphinx-theme":[function(require,module,exports){
var jQuery = (typeof(window) != 'undefined') ? window.jQuery : require('jquery');

// Sphinx theme nav state
function ThemeNav () {

    var nav = {
        navBar: null,
        win: null,
        winScroll: false,
        winResize: false,
        linkScroll: false,
        winPosition: 0,
        winHeight: null,
        docHeight: null,
        isRunning: false
    };

    nav.enable = function (withStickyNav) {
        var self = this;

        // TODO this can likely be removed once the theme javascript is broken
        // out from the RTD assets. This just ensures old projects that are
        // calling `enable()` get the sticky menu on by default. All other cals
        // to `enable` should include an argument for enabling the sticky menu.
        if (typeof(withStickyNav) == 'undefined') {
            withStickyNav = true;
        }

        if (self.isRunning) {
            // Only allow enabling nav logic once
            return;
        }

        self.isRunning = true;
        jQuery(function ($) {
            self.init($);

            self.reset();
            self.win.on('hashchange', self.reset);

            if (withStickyNav) {
                // Set scroll monitor
                self.win.on('scroll', function () {
                    if (!self.linkScroll) {
                        if (!self.winScroll) {
                            self.winScroll = true;
                            requestAnimationFrame(function() { self.onScroll(); });
                        }
                    }
                });
            }

            // Set resize monitor
            self.win.on('resize', function () {
                if (!self.winResize) {
                    self.winResize = true;
                    requestAnimationFrame(function() { self.onResize(); });
                }
            });

            self.onResize();
        });

    };

    // TODO remove this with a split in theme and Read the Docs JS logic as
    // well, it's only here to support 0.3.0 installs of our theme.
    nav.enableSticky = function() {
        this.enable(true);
    };

    nav.init = function ($) {
        var doc = $(document),
            self = this;

        this.navBar = $('div.pytorch-side-scroll:first');
        this.win = $(window);

        // Set up javascript UX bits
        $(document)
            // Shift nav in mobile when clicking the menu.
            .on('click', "[data-toggle='pytorch-left-menu-nav-top']", function() {
                $("[data-toggle='wy-nav-shift']").toggleClass("shift");
                $("[data-toggle='rst-versions']").toggleClass("shift");
            })

            // Nav menu link click operations
            .on('click', ".pytorch-menu-vertical .current ul li a", function() {
                var target = $(this);
                // Close menu when you click a link.
                $("[data-toggle='wy-nav-shift']").removeClass("shift");
                $("[data-toggle='rst-versions']").toggleClass("shift");
                // Handle dynamic display of l3 and l4 nav lists
                self.toggleCurrent(target);
                self.hashChange();
            })
            .on('click', "[data-toggle='rst-current-version']", function() {
                $("[data-toggle='rst-versions']").toggleClass("shift-up");
            })

        // Make tables responsive
        $("table.docutils:not(.field-list,.footnote,.citation)")
            .wrap("<div class='wy-table-responsive'></div>");

        // Add extra class to responsive tables that contain
        // footnotes or citations so that we can target them for styling
        $("table.docutils.footnote")
            .wrap("<div class='wy-table-responsive footnote'></div>");
        $("table.docutils.citation")
            .wrap("<div class='wy-table-responsive citation'></div>");

        // Add expand links to all parents of nested ul
        $('.pytorch-menu-vertical ul').not('.simple').siblings('a').each(function () {
            var link = $(this);
                expand = $('<span class="toctree-expand"></span>');
            expand.on('click', function (ev) {
                self.toggleCurrent(link);
                ev.stopPropagation();
                return false;
            });
            link.prepend(expand);
        });
    };

    nav.reset = function () {
        // Get anchor from URL and open up nested nav
        var anchor = encodeURI(window.location.hash) || '#';

        try {
            var vmenu = $('.pytorch-menu-vertical');
            var link = vmenu.find('[href="' + anchor + '"]');
            if (link.length === 0) {
                // this link was not found in the sidebar.
                // Find associated id element, then its closest section
                // in the document and try with that one.
                var id_elt = $('.document [id="' + anchor.substring(1) + '"]');
                var closest_section = id_elt.closest('div.section');
                link = vmenu.find('[href="#' + closest_section.attr("id") + '"]');
                if (link.length === 0) {
                    // still not found in the sidebar. fall back to main section
                    link = vmenu.find('[href="#"]');
                }
            }
            // If we found a matching link then reset current and re-apply
            // otherwise retain the existing match
            if (link.length > 0) {
                $('.pytorch-menu-vertical .current').removeClass('current');
                link.addClass('current');
                link.closest('li.toctree-l1').addClass('current');
                link.closest('li.toctree-l1').parent().addClass('current');
                link.closest('li.toctree-l1').addClass('current');
                link.closest('li.toctree-l2').addClass('current');
                link.closest('li.toctree-l3').addClass('current');
                link.closest('li.toctree-l4').addClass('current');
            }
        }
        catch (err) {
            console.log("Error expanding nav for anchor", err);
        }

    };

    nav.onScroll = function () {
        this.winScroll = false;
        var newWinPosition = this.win.scrollTop(),
            winBottom = newWinPosition + this.winHeight,
            navPosition = this.navBar.scrollTop(),
            newNavPosition = navPosition + (newWinPosition - this.winPosition);
        if (newWinPosition < 0 || winBottom > this.docHeight) {
            return;
        }
        this.navBar.scrollTop(newNavPosition);
        this.winPosition = newWinPosition;
    };

    nav.onResize = function () {
        this.winResize = false;
        this.winHeight = this.win.height();
        this.docHeight = $(document).height();
    };

    nav.hashChange = function () {
        this.linkScroll = true;
        this.win.one('hashchange', function () {
            this.linkScroll = false;
        });
    };

    nav.toggleCurrent = function (elem) {
        var parent_li = elem.closest('li');
        parent_li.siblings('li.current').removeClass('current');
        parent_li.siblings().find('li.current').removeClass('current');
        parent_li.find('> ul li.current').removeClass('current');
        parent_li.toggleClass('current');
    }

    return nav;
};

module.exports.ThemeNav = ThemeNav();

if (typeof(window) != 'undefined') {
    window.SphinxRtdTheme = {
        Navigation: module.exports.ThemeNav,
        // TODO remove this once static assets are split up between the theme
        // and Read the Docs. For now, this patches 0.3.0 to be backwards
        // compatible with a pre-0.3.0 layout.html
        StickyNav: module.exports.ThemeNav,
    };
}


// requestAnimationFrame polyfill by Erik Möller. fixes from Paul Irish and Tino Zijdel
// https://gist.github.com/paulirish/1579671
// MIT license

(function() {
    var lastTime = 0;
    var vendors = ['ms', 'moz', 'webkit', 'o'];
    for(var x = 0; x < vendors.length && !window.requestAnimationFrame; ++x) {
        window.requestAnimationFrame = window[vendors[x]+'RequestAnimationFrame'];
        window.cancelAnimationFrame = window[vendors[x]+'CancelAnimationFrame']
                                   || window[vendors[x]+'CancelRequestAnimationFrame'];
    }

    if (!window.requestAnimationFrame)
        window.requestAnimationFrame = function(callback, element) {
            var currTime = new Date().getTime();
            var timeToCall = Math.max(0, 16 - (currTime - lastTime));
            var id = window.setTimeout(function() { callback(currTime + timeToCall); },
              timeToCall);
            lastTime = currTime + timeToCall;
            return id;
        };

    if (!window.cancelAnimationFrame)
        window.cancelAnimationFrame = function(id) {
            clearTimeout(id);
        };
}());

$(".sphx-glr-thumbcontainer").removeAttr("tooltip");
$("table").removeAttr("border");

// This code replaces the default sphinx gallery download buttons
// with the 3 download buttons at the top of the page

var downloadNote = $(".sphx-glr-download-link-note.admonition.note");
if (downloadNote.length >= 1) {
    var tutorialUrlArray = $("#tutorial-type").text().split('/');
        tutorialUrlArray[0] = tutorialUrlArray[0] + "_source"

    var githubLink = "https://github.com/pytorch/tutorials/blob/master/" + tutorialUrlArray.join("/") + ".py",
        notebookLink = $(".reference.download")[1].href,
        notebookDownloadPath = notebookLink.split('_downloads')[1],
        colabLink = "https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads" + notebookDownloadPath;

    $("#google-colab-link").wrap("<a href=" + colabLink + " data-behavior='call-to-action-event' data-response='Run in Google Colab' target='_blank'/>");
    $("#download-notebook-link").wrap("<a href=" + notebookLink + " data-behavior='call-to-action-event' data-response='Download Notebook'/>");
    $("#github-view-link").wrap("<a href=" + githubLink + " data-behavior='call-to-action-event' data-response='View on Github' target='_blank'/>");
} else {
    $(".pytorch-call-to-action-links").hide();
}

//This code makes the Notes section of the Docs Left Nav collapsible

if ($("p.caption:first").text() == "Notes") {

    $("p.caption:first").addClass("left-nav-top-caption");
    $("span.caption-text:first").after("<span class='expand-menu'>[Expand]</span>");
    $(".expand-menu").after("<span class='hide-menu'>[Hide]</span>");
    $("p.caption:first").next("ul").hide();

    $(".expand-menu").on("click", function() {
        $(".hide-menu").toggle();
        toggleList(this);
    });

    $(".hide-menu").on("click", function() {
        $(".expand-menu").toggle();
        toggleList(this);
    });

    function toggleList(menuCommand) {
        $(menuCommand).toggle();
        $("p.caption:first").next("ul").toggle();
    }
}

// Get the card link from the card's link attribute

$(".tutorials-card").on("click", function() {
    window.location = $(this).attr("link");
});

// Build an array from each tag that's present

var tagList = $(".tutorials-card-container").map(function() {
    return $(this).data("tags").split(",").map(function(item) {
        return item.trim();
      });
}).get();

function unique(value, index, self) {
      return self.indexOf(value) == index && value != ""
    }

// Only return unique tags

var tags = tagList.sort().filter(unique);

// Add filter buttons to the top of the page for each tag

function createTagMenu() {
    tags.forEach(function(item){
    $(".tutorial-filter-menu").append(" <div class='tutorial-filter filter-btn filter' data-tag='" + item + "'>" + item + "</div>")
  })
};

createTagMenu();

// Remove hyphens if they are present in the filter buttons

$(".tags").each(function(){
    var tags = $(this).text().split(",");
    tags.forEach(function(tag, i ) {
       tags[i] = tags[i].replace(/-/, ' ')
    })
    $(this).html(tags.join(", "));
});

// Remove hyphens if they are present in the card body

$(".tutorial-filter").each(function(){
    var tag = $(this).text();
    $(this).html(tag.replace(/-/, ' '))
})

// Remove any empty p tags that Sphinx adds

$("#tutorial-cards p").each(function(index, item) {
    if(!$(item).text().trim()) {
        $(item).remove();
    }
});

},{"jquery":"jquery"}]},{},[1,2,3,4,5,6,7,8,9,10,"pytorch-sphinx-theme"]);
