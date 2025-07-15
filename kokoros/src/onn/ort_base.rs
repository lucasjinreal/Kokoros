
#[cfg(feature = "cuda")]
use ort::execution_providers::cuda::CUDAExecutionProvider;
use ort::execution_providers::cpu::CPUExecutionProvider;
use ort::session::builder::{SessionBuilder};
use ort::session::Session;
use ort::logging::LogLevel;

fn is_cpu_execution_provider() -> bool {
    #[cfg(feature = "cuda")]
    {
        false
    }
    #[cfg(all(not(feature = "cuda")))]
    {
        true
    }
}

fn get_cpu_cores() -> usize {
    num_cpus::get()
}

fn calculate_optimal_threads(total_instances: usize) -> usize {
    let total_cores = get_cpu_cores();
    let cores_per_instance = total_cores / total_instances;
    // Minimum 2 threads to maintain computational parallelism
    std::cmp::max(cores_per_instance, 2)
}

pub trait OrtBase {
    fn load_model(&mut self, model_path: String) -> Result<(), String> {
        self.load_model_with_instances(model_path, 1)
    }

    fn load_model_with_instances(&mut self, model_path: String, total_instances: usize) -> Result<(), String> {
        #[cfg(feature = "cuda")]
        let providers = [CUDAExecutionProvider::default().build()];


        #[cfg(all(not(feature = "cuda")))]
        let providers = [CPUExecutionProvider::default().build()];
        match SessionBuilder::new() {
            Ok(mut builder) => {
                let is_cpu = is_cpu_execution_provider();

                if is_cpu {
                    let optimal_threads = calculate_optimal_threads(total_instances);
                    let total_cores = get_cpu_cores();

                    tracing::info!(
                        "Applying CPU-specific ONNX Runtime threading optimizations: {} threads per instance ({} total instances, {} total cores)",
                        optimal_threads, total_instances, total_cores
                    );

                    builder = builder
                        .with_config_entry("session.use_mimalloc", "1")
                        .map_err(|e| format!("Failed to set mimalloc: {}", e))?
                        .with_intra_threads(optimal_threads)  // Distribute cores across instances
                        .map_err(|e| format!("Failed to set intra-op threads: {}", e))?;
                } else {
                    // GPU: Preserve default behavior with explicit configuration
                    tracing::info!("Using GPU execution provider - applying explicit GPU threading configuration");
                    builder = builder
                        .with_inter_threads(1)
                        .map_err(|e| format!("Failed to set GPU inter-op threads: {}", e))?;
                }

                let session = builder
                    .with_execution_providers(providers)
                    .map_err(|e| format!("Failed to build session: {}", e))?
                    .with_log_level(LogLevel::Warning)  // Suppress verbose INFO messages
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

            #[cfg(all(not(feature = "cuda")))]
            eprintln!("Configured with: CPU execution provider");
        } else {
            eprintln!("Session is not initialized.");
        }
    }

    fn set_sess(&mut self, sess: Session);
    fn sess(&self) -> Option<&Session>;
}
