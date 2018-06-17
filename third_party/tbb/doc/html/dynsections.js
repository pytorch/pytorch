function toggleVisibility(linkObj)
{
 var base = $(linkObj).attr('id');
 var summary = $('#'+base+'-summary');
 var content = $('#'+base+'-content');
 var trigger = $('#'+base+'-trigger');
 var src=$(trigger).attr('src');
 if (content.is(':visible')===true) {
   content.hide();
   summary.show();
   $(linkObj).addClass('closed').removeClass('opened');
   $(trigger).attr('src',src.substring(0,src.length-8)+'closed.png');
 } else {
   content.show();
   summary.hide();
   $(linkObj).removeClass('closed').addClass('opened');
   $(trigger).attr('src',src.substring(0,src.length-10)+'open.png');
 } 
 return false;
}

function updateStripes()
{
  $('table.directory tr').
       removeClass('even').filter(':visible:even').addClass('even');
}
function toggleLevel(level)
{
  $('table.directory tr').each(function(){ 
    var l = this.id.split('_').length-1;
    var i = $('#img'+this.id.substring(3));
    var a = $('#arr'+this.id.substring(3));
    if (l<level+1) {
      i.attr('src','ftv2folderopen.png');
      a.attr('src','ftv2mnode.png');
      $(this).show();
    } else if (l==level+1) {
      i.attr('src','ftv2folderclosed.png');
      a.attr('src','ftv2pnode.png');
      $(this).show();
    } else {
      $(this).hide();
    }
  });
  updateStripes();
}

function toggleFolder(id)
{
  //The clicked row
  var currentRow = $('#row_'+id);
  var currentRowImages = currentRow.find("img");

  //All rows after the clicked row
  var rows = currentRow.nextAll("tr");

  //Only match elements AFTER this one (can't hide elements before)
  var childRows = rows.filter(function() {
    var re = new RegExp('^row_'+id+'\\d+_$', "i"); //only one sub
    return this.id.match(re);
  });

  //First row is visible we are HIDING
  if (childRows.filter(':first').is(':visible')===true) {
    currentRowImages.filter("[id^=arr]").attr('src', 'ftv2pnode.png');
    currentRowImages.filter("[id^=img]").attr('src', 'ftv2folderclosed.png');
    rows.filter("[id^=row_"+id+"]").hide();
  } else { //We are SHOWING
    //All sub images
    var childImages = childRows.find("img");
    var childImg = childImages.filter("[id^=img]");
    var childArr = childImages.filter("[id^=arr]");

    currentRow.find("[id^=arr]").attr('src', 'ftv2mnode.png'); //open row
    currentRow.find("[id^=img]").attr('src', 'ftv2folderopen.png'); //open row
    childImg.attr('src','ftv2folderclosed.png'); //children closed
    childArr.attr('src','ftv2pnode.png'); //children closed
    childRows.show(); //show all children
  }
  updateStripes();
}


function toggleInherit(id)
{
  var rows = $('tr.inherit.'+id);
  var img = $('tr.inherit_header.'+id+' img');
  var src = $(img).attr('src');
  if (rows.filter(':first').is(':visible')===true) {
    rows.css('display','none');
    $(img).attr('src',src.substring(0,src.length-8)+'closed.png');
  } else {
    rows.css('display','table-row'); // using show() causes jump in firefox
    $(img).attr('src',src.substring(0,src.length-10)+'open.png');
  }
}

