use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // torch/_rust -> project root
    let project_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("torch/_rust must be two levels below the project root");

    let tensor_ffi_h = project_root.join("torch/_rust/csrc/tensor_ffi.h");
    let tensor_ffi_cpp = project_root.join("torch/_rust/csrc/tensor_ffi.cpp");

    // Where torch headers live. The OSS CMake build installs them under the
    // install prefix and passes that path via TORCH_RUST_INCLUDE_DIR; fall back
    // to the in-source / in-place layout.
    let torch_include = std::env::var("TORCH_RUST_INCLUDE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| project_root.join("torch/include"));
    let torch_csrc_api_include = project_root.join("torch/csrc/api/include");
    let torch_include_csrc_api_include = torch_include.join("torch/csrc/api/include");

    // Make Python.h available to torch/csrc/autograd/python_variable.h.
    let python_includes = pyo3_build_config::get().run_python_script(
        "import sysconfig; print(sysconfig.get_path('include'))",
    );

    let mut build = cxx_build::bridge("src/bindings/ffi.rs");
    build
        .file(&tensor_ffi_cpp)
        .include(project_root)
        .include(&torch_include)
        .include(&torch_csrc_api_include)
        .include(&torch_include_csrc_api_include)
        // torch's own headers emit many -Wunused-parameter warnings (virtual
        // method stubs); silence them so they don't drown out real ones.
        .flag_if_supported("-Wno-unused-parameter")
        .std("c++20");
    if let Ok(out) = python_includes {
        for line in out.lines() {
            let path = line.trim();
            if !path.is_empty() {
                build.include(path);
            }
        }
    }
    build.compile("torch_rust_tensor_ffi");

    println!("cargo:rerun-if-changed={}", tensor_ffi_h.display());
    println!("cargo:rerun-if-changed={}", tensor_ffi_cpp.display());
    // The OSS CMake build passes header/lib locations via these; re-run the
    // build script (which bakes them into the include/link search paths) when
    // they change, otherwise cargo reuses a stale cached invocation.
    println!("cargo:rerun-if-env-changed=TORCH_RUST_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=TORCH_RUST_LIB_DIR");

    // Link against libtorch_python.so so THPVariable_Check / THPVariableClass /
    // at::Tensor::sizes() resolve at load time. The libs live under torch/lib/
    // in the install tree; the OSS CMake build passes it via TORCH_RUST_LIB_DIR.
    let torch_lib = std::env::var("TORCH_RUST_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| project_root.join("torch/lib"));
    println!("cargo:rustc-link-search=native={}", torch_lib.display());
    println!("cargo:rustc-link-lib=dylib=torch_python");
    println!("cargo:rustc-link-lib=dylib=torch");

    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("linux") => println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/lib"),
        Ok("macos") => println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/lib"),
        _ => {}
    }
}
