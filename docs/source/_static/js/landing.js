/**
 * Landing page search – wires the inline search box to Sphinx's
 * built-in search index so results appear on /search.html?q=…
 *
 * Keyboard shortcut: pressing "/" anywhere on the landing page
 * focuses the search input (unless the user is already typing).
 */
document.addEventListener("DOMContentLoaded", function () {
  var input = document.getElementById("landing-search-input");
  if (!input) return;

  /* Focus shortcut: "/" key */
  document.addEventListener("keydown", function (e) {
    if (
      e.key === "/" &&
      !e.ctrlKey &&
      !e.metaKey &&
      !e.altKey &&
      document.activeElement.tagName !== "INPUT" &&
      document.activeElement.tagName !== "TEXTAREA" &&
      !document.activeElement.isContentEditable
    ) {
      e.preventDefault();
      input.focus();
    }
  });

  /* Pressing Escape while focused clears & blurs the input */
  input.addEventListener("keydown", function (e) {
    if (e.key === "Escape") {
      input.value = "";
      input.blur();
    }
  });
});
