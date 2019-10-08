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
