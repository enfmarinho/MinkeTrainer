use super::config;
use bullet_lib::{
    lr::{LinearDecayLR, Warmup},
    wdl::LinearWDL,
    TrainingSchedule, TrainingSteps,
};
use clap::{Parser, ValueEnum};
use strum_macros::{Display, EnumIter};

#[derive(Parser)]
pub struct Scheduler {
    #[arg(long, default_value_t = 0.10)]
    wdl: f32,

    #[arg(long, default_value_t = 16384)]
    batch_size: usize,

    #[arg(long, default_value_t = 6104)]
    batches_per_superbatch: usize,

    #[arg(long, default_value_t = 10)]
    save_rate: usize,
}

type TrainingSchedulerType = TrainingSchedule<Warmup<LinearDecayLR>, LinearWDL>;

#[derive(Clone, PartialEq, Display, ValueEnum, EnumIter, Debug)]
#[strum(serialize_all = "lowercase")]
pub enum Phase {
    Pretrain,
    Train,
    Tune,
}

impl Scheduler {
    pub fn get(
        &self,
        phase: Phase,
        start_superbatch: usize,
    ) -> Result<TrainingSchedulerType, String> {
        match phase {
            Phase::Pretrain => self.phase_pretrain(start_superbatch),
            Phase::Train => self.phase_train(start_superbatch),
            Phase::Tune => self.phase_tune(start_superbatch),
        }
    }

    fn phase_pretrain(&self, start_superbatch: usize) -> Result<TrainingSchedulerType, String> {
        if start_superbatch >= config::PRETRAIN_END_SUPERBATCH {
            return Err(format!(
                "Starting pretrain with a superbatch higher than the PRETRAIN_END_SUPERBATCH:{}",
                config::PRETRAIN_END_SUPERBATCH
            ));
        }
        Ok(TrainingSchedule {
            net_id: "minke-pretrain".to_string(),
            eval_scale: 400.,
            steps: TrainingSteps {
                batch_size: self.batch_size,
                batches_per_superbatch: self.batches_per_superbatch,
                start_superbatch,
                end_superbatch: config::PRETRAIN_END_SUPERBATCH,
            },
            wdl_scheduler: LinearWDL { start: 0., end: 0. },
            lr_scheduler: Warmup {
                warmup_batches: self.batches_per_superbatch,
                inner: LinearDecayLR {
                    initial_lr: 1e-3,
                    final_lr: 1e-4,
                    final_superbatch: config::PRETRAIN_END_SUPERBATCH,
                },
            },
            save_rate: self.save_rate,
        })
    }

    fn phase_train(&self, start_superbatch: usize) -> Result<TrainingSchedulerType, String> {
        if start_superbatch >= config::TUNE_END_SUPERBATCH {
            return Err(format!(
                "Starting train with a superbatch higher than the TRAIN_END_SUPERBATCH:{}",
                config::TRAIN_END_SUPERBATCH
            ));
        }
        Ok(TrainingSchedule {
            net_id: "minke-train".to_string(),
            eval_scale: 400.,
            steps: TrainingSteps {
                batch_size: self.batch_size,
                batches_per_superbatch: self.batches_per_superbatch,
                start_superbatch,
                end_superbatch: config::TRAIN_END_SUPERBATCH,
            },
            wdl_scheduler: LinearWDL {
                start: 0.,
                end: self.wdl,
            },
            lr_scheduler: Warmup {
                warmup_batches: self.batches_per_superbatch,
                inner: LinearDecayLR {
                    initial_lr: 5e-4,
                    final_lr: 1e-5,
                    final_superbatch: config::TRAIN_END_SUPERBATCH,
                },
            },
            save_rate: self.save_rate,
        })
    }

    fn phase_tune(&self, start_superbatch: usize) -> Result<TrainingSchedulerType, String> {
        if start_superbatch >= config::TUNE_END_SUPERBATCH {
            return Err(format!(
                "Starting tune with a superbatch higher than the TUNE_END_SUPERBATCH:{}",
                config::TUNE_END_SUPERBATCH
            ));
        }
        Ok(TrainingSchedule {
            net_id: "minke-tune".to_string(),
            eval_scale: 400.,
            steps: TrainingSteps {
                batch_size: self.batch_size,
                batches_per_superbatch: self.batches_per_superbatch,
                start_superbatch,
                end_superbatch: config::TUNE_END_SUPERBATCH,
            },
            wdl_scheduler: LinearWDL {
                start: self.wdl,
                end: self.wdl,
            },
            lr_scheduler: Warmup {
                warmup_batches: self.batches_per_superbatch,
                inner: LinearDecayLR {
                    initial_lr: 1e-4,
                    final_lr: 0.,
                    final_superbatch: config::TUNE_END_SUPERBATCH,
                },
            },
            save_rate: self.save_rate,
        })
    }
}
