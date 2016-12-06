/** Example */

var output = '', line = '';
all.forEach(function (item) {
  if (12 + 1 + line.length + 1 + item.length + 4 > 120) {
    output += "\n" + "            '" + line + " ' +";
    line = '';
    return;
  }
  if (line) {
    line = line + ' ' + item;
  } else {
    line = item;
  }
});
console.log(output);
