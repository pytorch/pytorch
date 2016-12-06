use <write.scad>
include <../common/base.scad>

//draw a foobar
module foobar(){
    translate([0,-10,0])
    difference(){
        cube([5,10.04,15]);
        sphere(r=10,$fn=100);
    }
}

foobar();
#cube ([5,5,5]);
echo("done");