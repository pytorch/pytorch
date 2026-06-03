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

    // Where torch headers live in the source tree and post-CMake install tree.
    let torch_include = project_root.join("torch/include");
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
        .std("c++17");
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

    // Link against libtorch_python.so so THPVariable_Check / THPVariableClass /
    // at::Tensor::sizes() resolve at load time. The library is installed under
    // torch/lib/ once the main C++ build has run.
    let torch_lib = project_root.join("torch/lib");
    println!("cargo:rustc-link-search=native={}", torch_lib.display());
    println!("cargo:rustc-link-lib=dylib=torch_python");
    println!("cargo:rustc-link-lib=dylib=torch");

    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("linux") => println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/lib"),
        Ok("macos") => println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/lib"),
        _ => {}
    }
}
