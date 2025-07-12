#[cfg(feature = "cuda")]
use ort::execution_providers::cuda::CUDAExecutionProvider;
use ort::execution_providers::cpu::CPUExecutionProvider;
use ort::session::builder::{SessionBuilder, GraphOptimizationLevel};
use ort::session::Session;
use ort::logging::LogLevel;
use std::env;

fn is_cpu_execution_provider() -> bool {
    #[cfg(feature = "cuda")]
    {
        // Runtime detection: check if CUDA is actually available/being used
        // For now, we'll use the feature flag but this could be enhanced
        // to check actual GPU availability at runtime
        false
    }
    #[cfg(not(feature = "cuda"))]
    {
        true
    }
}

pub trait OrtBase {
    fn load_model(&mut self, model_path: String) -> Result<(), String> {
        #[cfg(feature = "cuda")]
        let providers = [CUDAExecutionProvider::default().build()];

        #[cfg(not(feature = "cuda"))]
        let providers = [CPUExecutionProvider::default().build()];

        match SessionBuilder::new() {
            Ok(mut builder) => {
                // Runtime platform detection
                let is_cpu = is_cpu_execution_provider();

                if is_cpu {
                    // CPU: Apply validated memory bandwidth optimizations
                    tracing::info!("Applying CPU-specific ONNX Runtime threading optimizations");
                    
                    // Set OMP_NUM_THREADS environment variable for single-threading
                    // SAFETY: Setting OMP_NUM_THREADS is safe as it's a well-defined OpenMP environment variable
                    unsafe {
                        env::set_var("OMP_NUM_THREADS", "1");
                    }
                    
                    builder = builder
                        .with_optimization_level(GraphOptimizationLevel::Level3)
                        .map_err(|e| format!("Failed to set optimization level: {}", e))?
                        .with_inter_threads(1)  // Only constraint inter-op threading
                        .map_err(|e| format!("Failed to set inter-op threads: {}", e))?
                        .with_parallel_execution(false)
                        .map_err(|e| format!("Failed to disable parallel execution: {}", e))?;
                } else {
                    // GPU: Preserve default behavior with explicit configuration
                    tracing::info!("Using GPU execution provider - applying explicit GPU threading configuration");
                    builder = builder
                        .with_inter_threads(1)  // Sequential graph execution
                        .map_err(|e| format!("Failed to set inter-op threads: {}", e))?;
                    // intra_op_num_threads defaults to 0 (use default GPU threading)
                }

                let session = builder
                    .with_execution_providers(providers)
                    .map_err(|e| format!("Failed to build session: {}", e))?
                    .with_log_level(LogLevel::Warning)
                    .map_err(|e| format!("Failed to set log level: {}", e))?
                    .commit_from_file(model_path)
                    .map_err(|e| format!("Failed to commit from file: {}", e))?;
                self.set_sess(session);
                Ok(())
            }
            Err(e) => Err(format!("Failed to create session builder: {}", e)),
        }
    }

    fn print_info(&self) {
        if let Some(session) = self.sess() {
            eprintln!("Input names:");
            for input in &session.inputs {
                eprintln!("  - {}", input.name);
            }
            eprintln!("Output names:");
            for output in &session.outputs {
                eprintln!("  - {}", output.name);
            }

            #[cfg(feature = "cuda")]
            eprintln!("Configured with: CUDA execution provider");

            #[cfg(not(feature = "cuda"))]
            eprintln!("Configured with: CPU execution provider");
        } else {
            eprintln!("Session is not initialized.");
        }
    }

    fn set_sess(&mut self, sess: Session);
    fn sess(&self) -> Option<&Session>;
}
