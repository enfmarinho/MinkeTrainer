use crate::orchestrator::scheduler::Phase;
use clap::Args;
use serde::Deserialize;

#[derive(Args, Debug, Deserialize)]
pub struct CheckpointConfig {
    /// The current training phase
    #[arg(short, long, default_value_t = CheckpointConfig::default().phase)]
    pub phase: Phase,

    /// The superbatch index to resume from
    #[arg(short, long, default_value_t = CheckpointConfig::default().superbatch)]
    pub superbatch: usize,

    /// Path to a checkpoint file to load weights from
    #[arg(short, long, default_value_t = CheckpointConfig::default().load)]
    pub load: String,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        CheckpointConfig {
            phase: Phase::Pretrain,
            superbatch: 0,
            load: String::from(""),
        }
    }
}

#[derive(Args, Debug, Deserialize)]
pub struct DatasetConfig {
    /// Paths to pre-train datasets
    #[arg(long, value_delimiter = ',', required = true)]
    pub pre_train: Vec<String>,

    /// Paths to train datasets
    #[arg(long, value_delimiter = ',', required = true)]
    pub train: Vec<String>,

    /// Paths to tune datasets
    #[arg(long, value_delimiter = ',', required = true)]
    pub tune: Vec<String>,
}
