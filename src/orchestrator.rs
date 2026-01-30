mod default;
mod datafilter;
mod scheduler;

pub use default::get_material_count_target;

use bullet_lib::{
    game::{formats::bulletformat::ChessBoard, inputs::Chess768, outputs::MaterialCount},
    nn::optimiser::AdamW,
    trainer::save::SavedFormat,
    value::{loader, ValueTrainerBuilder},
    LocalSettings,
};
use clap::{Args, Parser};
use datafilter::DataFilter;
use scheduler::{Phase, Scheduler};
use serde::Deserialize;
use std::io::{self, Write};
use std::{
    fs::{self, create_dir_all, OpenOptions},
    path::PathBuf,
};
use strum::IntoEnumIterator;

#[derive(Parser, Debug)]
#[command(name = "nnue-trainer", version, about = "NNUE Trainer Orchestrator")]
pub enum Commands {
    /// Start the trainer manually using command line arguments
    Run(Orchestrator),

    /// Load the trainer configuration from a TOML file
    File { path: PathBuf },
}

impl Commands {
    pub fn extract_orchestrator(self) -> Orchestrator {
        match self {
            Commands::Run(orch) => orch,
            Commands::File { path } => {
                let content = fs::read_to_string(&path).expect("Could not read config file");
                toml::from_str(&content).expect("Could not parse TOML config")
            }
        }
    }
}

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
    pre_train: Vec<String>,

    #[arg(long, value_delimiter = ',', required = true)]
    train: Vec<String>,

    #[arg(long, value_delimiter = ',', required = true)]
    tune: Vec<String>,
}

#[derive(Args, Debug, Deserialize)]
pub struct Orchestrator {
    #[arg(short, long)]
    output_dir: String,

    #[arg(short, long, default_value_t = Orchestrator::default_threads())]
    #[serde(default = "Orchestrator::default_threads")]
    threads: usize,

    #[arg(long, default_value_t = false)]
    #[serde(default)]
    log_config: bool,

    #[clap(flatten)]
    #[serde(default = "CheckpointConfig::default")]
    load_checkpoint: CheckpointConfig,

    #[clap(flatten)]
    datasets: DatasetConfig,

    #[clap(flatten)]
    #[serde(default = "Scheduler::default")]
    scheduler: Scheduler,

    #[clap(flatten)]
    #[serde(default = "DataFilter::default")]
    filter: DataFilter,

    #[arg(long, default_value_t = default::BUFFER_SIZE_MB)]
    buffer_size_mb: usize,
}

impl Orchestrator {
    pub fn run(&mut self) {
        match create_dir_all(&self.output_dir) {
            Ok(_) => (),
            Err(e) => {
                println!(
                    "Could not create checkpoint directory \"{}\" because {}",
                    self.output_dir, e
                );
                return;
            }
        }

        if self.log_config {
            match self.log_config() {
                Ok(()) => (),
                Err(e) => println!("Writing config log failed because {}", e),
            }
        }

        let mut trainer = ValueTrainerBuilder::default()
            .dual_perspective()
            .optimiser(AdamW)
            .inputs(Chess768)
            .output_buckets(MaterialCount::<{ default::N_OUTPUT_BUCKETS }>)
            .save_format(&[
                SavedFormat::id("l0w").quantise::<i16>(default::QA),
                SavedFormat::id("l0b").quantise::<i16>(default::QA),
                SavedFormat::id("l1w")
                    .quantise::<i16>(default::QB)
                    .transpose(),
                SavedFormat::id("l1b").quantise::<i16>(default::QAB),
            ])
            .loss_fn(|output, target| output.sigmoid().squared_error(target))
            .build(|builder, stm_inputs, ntm_inputs, out_buckets| {
                let l0 = builder.new_affine("l0", 768, default::HIDDEN_LAYER_SIZE);
                let l1 = builder.new_affine(
                    "l1",
                    2 * default::HIDDEN_LAYER_SIZE,
                    default::N_OUTPUT_BUCKETS,
                );

                let stm_hidden = l0.forward(stm_inputs).screlu();
                let ntm_hidden = l0.forward(ntm_inputs).screlu();
                let hidden_layer = stm_hidden.concat(ntm_hidden);
                l1.forward(hidden_layer).select(out_buckets)
            });

        let settings = LocalSettings {
            threads: self.threads,
            test_set: None,
            output_directory: self.output_dir.as_str(),
            batch_queue_size: 64,
        };

        let start_index = Phase::iter()
            .position(|p| p == self.load_checkpoint.phase)
            .expect("Invalid phase argument"); // This is not ideal, but it's safe

        for phase in Phase::iter().skip(start_index) {
            let start_superbatch = if self.load_checkpoint.phase == phase {
                self.load_checkpoint.superbatch
            } else {
                0
            };

            let datasets = match phase {
                Phase::Pretrain => &self.datasets.pre_train,
                Phase::Train => &self.datasets.train,
                Phase::Tune => &self.datasets.tune,
            };
            let scheduler = self.scheduler.get(phase, start_superbatch).expect("");

            trainer.run(&scheduler, &settings, &self.load(datasets));
        }
    }

    pub fn load(&self, datasets: &[String]) -> impl loader::DataLoader<ChessBoard> {
        let datasets = Vec::from_iter(datasets.iter().map(|s| s.as_str()));
        let filter = self.filter.clone();

        loader::SfBinpackLoader::new_concat_multiple(
            &datasets,
            self.buffer_size_mb,
            self.threads,
            move |entry| filter.filter(entry),
        )
    }

    fn default_threads() -> usize {
        1
    }

    fn log_config(&self) -> io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(format!("{}/config.log", self.output_dir.clone()))?;

        writeln!(file, "{:#?}", self)?;

        Ok(())
    }
}
