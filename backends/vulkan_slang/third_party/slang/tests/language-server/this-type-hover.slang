//TEST:LANG_SERVER(filecheck=CHECK):

struct G
{}

extension G
{
//HOVER:9,7
    This my();
}

// CHECK: struct G
