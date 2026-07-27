#pragma once

#define TH_CONCAT_STRING_2(x, y) TH_CONCAT_STRING_2_EXPAND(x, y)
#define TH_CONCAT_STRING_2_EXPAND(x, y) #x #y

#define TH_CONCAT_STRING_3(x, y, z) TH_CONCAT_STRING_3_EXPAND(x, y, z)
#define TH_CONCAT_STRING_3_EXPAND(x, y, z) #x #y #z

#define TH_CONCAT_STRING_4(x, y, z, w) TH_CONCAT_STRING_4_EXPAND(x, y, z, w)
#define TH_CONCAT_STRING_4_EXPAND(x, y, z, w) #x #y #z #w

#define TH_CONCAT_2(x, y) TH_CONCAT_2_EXPAND(x, y)
#define TH_CONCAT_2_EXPAND(x, y) x##y

#define TH_CONCAT_3(x, y, z) TH_CONCAT_3_EXPAND(x, y, z)
#define TH_CONCAT_3_EXPAND(x, y, z) x##y##z

#define TH_CONCAT_4_EXPAND(x, y, z, w) x##y##z##w
#define TH_CONCAT_4(x, y, z, w) TH_CONCAT_4_EXPAND(x, y, z, w)
