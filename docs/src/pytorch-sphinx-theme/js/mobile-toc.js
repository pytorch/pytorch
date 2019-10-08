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
