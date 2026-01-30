use clap::Args;
use serde::Deserialize;

use crate::orchestrator::scheduler::Phase;

#[derive(Args, Debug, Deserialize)]
pub struct CheckpointConfig {
    #[arg(short, long, default_value_t = CheckpointConfig::default().phase)]
    pub phase: Phase,

    #[arg(short, long, default_value_t = CheckpointConfig::default().superbatch)]
    pub superbatch: usize,

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
    #[arg(long, value_delimiter = ',', required = true)]
    pub pre_train: Vec<String>,

    #[arg(long, value_delimiter = ',', required = true)]
    pub train: Vec<String>,

    #[arg(long, value_delimiter = ',', required = true)]
    pub tune: Vec<String>,
}
