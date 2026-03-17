#![recursion_limit = "256"]

use burn::{backend::Autodiff, tensor::backend::Backend};
use simple_regression::{inference, training};

static ARTIFACT_DIR: &str = "/tmp/burn-example-regression";

#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
    feature = "ndarray-blas-accelerate",
))]
mod ndarray {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    pub fn run() {
        let device = NdArrayDevice::Cpu;
        super::run::<NdArray>(device.clone());
    }
}

#[cfg(feature = "rocm")]
mod rocm {
    use burn::backend::Rocm;

    pub fn run() {
        let device = Default::default();
        super::run::<Rocm>(device);
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use burn::backend::Cuda;

    pub fn run() {
        let device = Default::default();
        super::run::<Cuda>(device);
    }
}

#[cfg(any(feature = "wgpu", feature = "metal"))]
mod wgpu {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    pub fn run() {
        let device = WgpuDevice::default();
        super::run::<Wgpu>(device);
    }
}

/// Train a regression model and predict results on a number of samples.
pub fn run<B: Backend>(device: B::Device) {
    training::run::<Autodiff<B>>(ARTIFACT_DIR, device.clone());
    inference::infer::<B>(ARTIFACT_DIR, device)
}

fn main() {
    #[cfg(any(
        feature = "ndarray",
        feature = "ndarray-blas-netlib",
        feature = "ndarray-blas-openblas",
        feature = "ndarray-blas-accelerate",
    ))]
    ndarray::run();

    #[cfg(feature = "cuda")]
    cuda::run();

    #[cfg(feature = "rocm")]
    rocm::run();

    #[cfg(any(feature = "wgpu", feature = "metal"))]
    wgpu::run();
}
