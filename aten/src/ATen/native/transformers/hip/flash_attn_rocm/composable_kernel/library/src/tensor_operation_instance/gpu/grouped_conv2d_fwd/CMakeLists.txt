add_instance_library(device_grouped_conv2d_fwd_instance
   # GNHWC, GKYXC, GNHWK
   device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_bf16_instance.cpp
   device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_f16_instance.cpp
   device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_f32_instance.cpp
   device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_int8_instance.cpp
   # NHWGC, GKYXC, NHWGK
   device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_instance.cpp
   #dl 
   device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_f16_instance.cpp
   device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_f32_instance.cpp
   device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_int8_instance.cpp
)
