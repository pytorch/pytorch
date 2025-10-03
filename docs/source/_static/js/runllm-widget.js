document.addEventListener("DOMContentLoaded", function () {
  var script = document.createElement("script");
  script.type = "module";
  script.id = "runllm-widget-script"

  script.src = "https://widget.runllm.com";

  script.setAttribute("version", "stable");
  script.setAttribute("crossorigin", "true");
  script.setAttribute("runllm-keyboard-shortcut", "Mod+j");
  script.setAttribute("runllm-name", "PyTorch");
  script.setAttribute("runllm-position", "BOTTOM_RIGHT");
  script.setAttribute("runllm-assistant-id", "834");

  script.async = true;
  document.head.appendChild(script);
});
