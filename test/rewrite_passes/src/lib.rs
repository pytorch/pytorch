//! What a user's library of passes looks like: see test_rewrite.py.

use torch_rewrite_pass::{export_passes, Bound, Error, Pass, Region};

/// Replaces x * rsqrt(mean(x ** 2) + eps) by the kernel it is bound to.
#[derive(Default)]
struct FuseRmsNorm;

impl Pass for FuseRmsNorm {
    fn name(&self) -> &'static str {
        "fuse_rmsnorm"
    }

    fn run(&self, region: &mut Region<'_>, kernel: Option<Bound>) -> Result<bool, Error> {
        let Some(kernel) = kernel else {
            return region.fail("fuse_rmsnorm needs a kernel: bind(kernel)");
        };
        match region.find(&["pow", "mean", "add", "rsqrt", "mul"]) {
            Some((start, end)) => region.swap(start, end, kernel, 4),
            None => Ok(false),
        }
    }
}

/// Changes nothing, and fails where a block multiplies.
#[derive(Default)]
struct Strict;

impl Pass for Strict {
    fn name(&self) -> &'static str {
        "strict"
    }

    fn run(&self, region: &mut Region<'_>, _bound: Option<Bound>) -> Result<bool, Error> {
        let kernels: Vec<String> = region.insts().into_iter().filter_map(|i| i.kernel).collect();
        if kernels.iter().any(|k| k.starts_with("aten::mul")) {
            return region.fail(&format!("a block that multiplies: {}", kernels.join(", ")));
        }
        Ok(false)
    }
}

export_passes!(FuseRmsNorm, Strict);
