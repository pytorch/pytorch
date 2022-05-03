def remote_glog_library(name, srcs, http_archive, exported_preprocessor_flags, **kwargs):

    config_headers = [
                'vlog_is_on',
                'stl_logging',
                'raw_logging',
                'logging',
    ]

    sed = " ".join([
        "sed",
        "-e 's/@ac_cv_cxx_using_operator@/1/g'",
        "-e 's/@ac_cv_have_unistd_h@/1/g'",
        "-e 's/@ac_cv_have_stdint_h@/1/g'",
        "-e 's/@ac_cv_have_systypes_h@/1/g'",
        "-e 's/@ac_cv_have_libgflags@/0/g'",
        "-e 's/@ac_cv_have_uint16_t@/1/g'",
        "-e 's/@ac_cv_have___builtin_expect@/1/g'",
        "-e 's/@ac_cv_have_.*@/0/g'",
        "-e 's/@ac_google_start_namespace@/namespace google {/g'",
        "-e 's/@ac_google_end_namespace@/}/g'",
        "-e 's/@ac_google_namespace@/google/g'",
        "-e 's/@ac_cv___attribute___noinline@/__attribute__((noinline))/g'",
        "-e 's/@ac_cv___attribute___noreturn@/__attribute__((noreturn))/g'",
        "-e 's/@ac_cv___attribute___printf_4_5@/__attribute__((__format__ (__printf__, 4, 5)))/g'",
    ])

    new_srcs = {src: [src] for src in srcs}
    cmd = ["rsync -a $(location :{})/ $OUT".format(http_archive)]

    cmd.append("awk '{ gsub(/^#cmakedefine/, \"//cmakedefine\"); print; }'" + " $(location :{})/src/config.h.cmake.in > $OUT/src/config.h".format(http_archive))
    cmd.append("awk '{ gsub(/^#cmakedefine/, \"//cmakedefine\"); print; }'" + " $(location :{})/src/config.h.cmake.in > $OUT/src/base/config.h".format(http_archive))
    new_srcs["config.h"] = ["src/config.h"]

    cmd.append("rsync -a $(location :{})/src/glog/log_severity.h $OUT/glog/".format(http_archive))
    new_srcs["log_severity.h"] = ["glog/log_severity.h"]

    for f in [
                'vlog_is_on',
                'stl_logging',
                'raw_logging',
                'logging',
            ]:
        cmd.append("{} $(location :{})/src/glog/{}.h.in > $OUT/glog/{}.h".format(sed, http_archive, f, f))
        new_srcs["{}.h".format(f)] = ["glog/{}.h".format(f)]
    new_srcs["."] = ["glog"]

    temp_name = name + "_temp"
    native.genrule(
        name = temp_name,
        outs = new_srcs,
        cmd = " && ".join(cmd),
        # default_outs = ["."],
    )

    native.cxx_library(
        name = name,
        srcs = [":{}[{}]".format(temp_name, src) for src in srcs],
        headers = [":{}[{}]".format(temp_name, "config.h")],
        # this is a hack for OSS genrule, since it cannot get the location of a multi-outpus genrule
        exported_preprocessor_flags = exported_preprocessor_flags + ["-I$(location :glog_temp[.])/../"],
        **kwargs
    )
