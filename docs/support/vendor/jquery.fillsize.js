/*! fillsize (c) 2012, Rico Sta. Cruz. MIT License.
 *  http://github.com/rstacruz/jquery-stuff/tree/master/fillsize */

// Makes an element fill up its container.
//
//     $(".container").fillsize("> img");
//
// This binds a listener on window resizing to automatically scale down the
// child (`> img` in this example) just so that enough of it will be visible in
// the viewport of the container.
// 
// This assumes that the container has `position: relative` (or any 'position',
// really), and `overflow: hidden`.

(function($) {
  $.fn.fillsize = function(selector) {
    var $parent = this;
    var $img;

    function resize() {
      if (!$img) $img = $parent.find(selector);

      $img.each(function() {
        if (!this.complete) return;
        var $img = $(this);

        var parent = { height: $parent.innerHeight(), width: $parent.innerWidth() };
        var imageRatio     = $img.width() / $img.height();
        var containerRatio = parent.width / parent.height;

        var css = {
          position: 'absolute',
          left: 0, top: 0, right: 'auto', bottom: 'auto'
        };

        // If image is wider than the container
        if (imageRatio > containerRatio) {
          css.left = Math.round((parent.width - imageRatio * parent.height) / 2) + 'px';
          css.width = 'auto';
          css.height = '100%';
        }

        // If the container is wider than the image
        else {
          css.top = Math.round((parent.height - (parent.width / $img.width() * $img.height())) / 2) + 'px';
          css.height = 'auto';
          css.width = '100%';
        }

        $img.css(css);
      });
    }

    // Make it happen on window resize.
    $(window).resize(resize);

    // Allow manual invocation by doing `.trigger('fillsize')` on the container.
    $(document).on('fillsize', $parent.selector, resize);

    // Resize on first load (or immediately if called after).
    $(function() {
      // If the child is an image, fill it up when image's real dimensions are
      // first determined. Needs to be .bind() because the load event will
      // bubble up.
      $(selector, $parent).bind('load', function() {
        setTimeout(resize, 25);
      });

      resize();
    });

    return this;
  };
})(jQuery);
