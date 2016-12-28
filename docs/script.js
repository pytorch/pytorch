function registerSearchbar(searchbar) {
  searchbar.addEventListener('keyup', function(e) {
    filterResults(e.target.value);
  }, false);
}

function filterResults(value) {
  let content = document.getElementsByClassName('content')[0].childNodes;
  console.log(content);
  let lastH1Index = 0;
  let keep = false;
  for (let i = 0; i < content.length; ++i) {
    content[i].classList.toggle('Caffe2_hidden', false);
  }
  for (let i = 0; i <= content.length; ++i) {

    if (i === content.length || (content[i].nodeName === "H1" && i !== 0)) {
      if (!keep) {
        for (let j = lastH1Index; j < i; ++j) {
          content[j].classList.toggle('Caffe2_hidden', true);
        }
      }

      if (i === content.length) {
        break;
      }

      lastH1Index = i;
      keep = false;
    }
    if (content[i].textContent.indexOf(value) > -1) {
      keep = true;
    }
  }
}
