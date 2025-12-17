mod config;
mod dataloader;
mod scheduler;

use bullet_lib::{
    game::inputs::Chess768, nn::optimiser::AdamW, trainer::save::SavedFormat,
    value::ValueTrainerBuilder, LocalSettings,
};
use clap::Parser;
use dataloader::{DataFilter, DataLoader};
use scheduler::{Phase, Scheduler};
use std::fs::{create_dir_all, OpenOptions};
use std::io::{self, Write};
use strum::IntoEnumIterator;

#[derive(Parser, Debug)]
pub struct Orchestrator {
    #[arg(short, long, value_delimiter = ',', required = true)]
    datasets: Vec<String>,

    #[arg(short, long, default_value_t = String::from(""))]
    load_from: String,

    #[arg(short, long, default_value_t = Phase::Pretrain)]
    phase: Phase,

    #[arg(short, long, default_value_t = 0)]
    start_superbatch: usize,

    #[arg(short, long, default_value_t = String::from("checkpoint"))]
    checkpoint: String,

    #[arg(short, long, default_value_t = 1)]
    threads: usize,

    #[arg(long, default_value_t = true)]
    log_config: bool,

    #[clap(flatten)]
    scheduler: Scheduler,

    #[clap(flatten)]
    filter: DataFilter,

    #[clap(flatten)]
    data_loader: DataLoader,
}

impl Orchestrator {
    pub fn run(&mut self) {
        match create_dir_all(&self.checkpoint) {
            Ok(_) => (),
            Err(e) => {
                println!(
                    "Could not create checkpoint directory \"{}\" because {}",
                    self.checkpoint, e
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
            .save_format(&[
                SavedFormat::id("l0w").quantise::<i16>(255),
                SavedFormat::id("l0b").quantise::<i16>(255),
                SavedFormat::id("l1w").quantise::<i16>(64),
                SavedFormat::id("l1b").quantise::<i16>(255 * 64),
            ])
            .loss_fn(|output, target| output.sigmoid().squared_error(target))
            .build(|builder, stm_inputs, ntm_inputs| {
                let l0 = builder.new_affine("l0", 768, config::HIDDEN_LAYER_SIZE);
                let l1 = builder.new_affine("l1", 2 * config::HIDDEN_LAYER_SIZE, 1);

                let stm_hidden = l0.forward(stm_inputs).screlu();
                let ntm_hidden = l0.forward(ntm_inputs).screlu();
                let hidden_layer = stm_hidden.concat(ntm_hidden);
                l1.forward(hidden_layer)
            });

        let settings = LocalSettings {
            threads: self.threads,
            test_set: None,
            output_directory: self.checkpoint.as_str(),
            batch_queue_size: 64,
        };

        let start_index = Phase::iter()
            .position(|p| p == self.phase)
            .expect("Invalid phase argument"); // This is not ideal, but it's safe
        for phase in Phase::iter().skip(start_index) {
            let start_superbatch = if self.phase == phase {
                self.start_superbatch
            } else {
                0
            };

            let scheduler = self.scheduler.get(phase, start_superbatch).expect("");

            trainer.run(
                &scheduler,
                &settings,
                &self
                    .data_loader
                    .load(&self.datasets, &self.filter, self.threads),
            );
        }
    }

    fn log_config(&self) -> io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(format!("{}/config.log", self.checkpoint.clone()))?;

        writeln!(file, "{:#?}", self)?;

        Ok(())
    }
}
