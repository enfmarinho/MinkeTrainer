use super::default;
use bullet_lib::{
    lr::{LinearDecayLR, Warmup},
    wdl::LinearWDL,
    TrainingSchedule, TrainingSteps,
};
use clap::{Parser, ValueEnum};
use serde::Deserialize;
use strum_macros::{Display, EnumIter};

#[derive(Parser, Debug, Deserialize)]
pub struct Scheduler {
    #[arg(long, default_value_t = Scheduler::default().wdl)]
    wdl: f32,

    #[arg(long, default_value_t = Scheduler::default().batch_size)]
    batch_size: usize,

    #[arg(long, default_value_t = Scheduler::default().batches_per_superbatch)]
    batches_per_superbatch: usize,

    #[arg(long, default_value_t = Scheduler::default().save_rate)]
    save_rate: usize,
}

impl Default for Scheduler {
    fn default() -> Self {
        Scheduler {
            wdl: default::WDL,
            batch_size: default::BATCH_SIZE,
            batches_per_superbatch: default::BATCHES_PER_SUPERBATCH,
            save_rate: default::SAVE_RATE,
        }
    }
}

type TrainingSchedulerType = TrainingSchedule<Warmup<LinearDecayLR>, LinearWDL>;

#[derive(Clone, PartialEq, Display, ValueEnum, EnumIter, Debug, Deserialize)]
#[strum(serialize_all = "lowercase")]
#[serde(rename_all = "lowercase")]
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
        if start_superbatch >= default::PRETRAIN_END_SUPERBATCH {
            return Err(format!(
                "Starting pretrain with a superbatch higher than the PRETRAIN_END_SUPERBATCH:{}",
                default::PRETRAIN_END_SUPERBATCH
            ));
        }
        Ok(TrainingSchedule {
            net_id: "pretrain".to_string(),
            eval_scale: default::EVAL_SCALE,
            steps: TrainingSteps {
                batch_size: self.batch_size,
                batches_per_superbatch: self.batches_per_superbatch,
                start_superbatch,
                end_superbatch: default::PRETRAIN_END_SUPERBATCH,
            },
            wdl_scheduler: LinearWDL { start: 0., end: 0. },
            lr_scheduler: Warmup {
                warmup_batches: self.batches_per_superbatch,
                inner: LinearDecayLR {
                    initial_lr: default::PRETRAIN_INITIAL_LR,
                    final_lr: default::PRETRAIN_FINAL_LR,
                    final_superbatch: default::PRETRAIN_END_SUPERBATCH,
                },
            },
            save_rate: self.save_rate,
        })
    }

    fn phase_train(&self, start_superbatch: usize) -> Result<TrainingSchedulerType, String> {
        if start_superbatch >= default::TUNE_END_SUPERBATCH {
            return Err(format!(
                "Starting train with a superbatch higher than the TRAIN_END_SUPERBATCH:{}",
                default::TRAIN_END_SUPERBATCH
            ));
        }
        Ok(TrainingSchedule {
            net_id: "train".to_string(),
            eval_scale: default::EVAL_SCALE,
            steps: TrainingSteps {
                batch_size: self.batch_size,
                batches_per_superbatch: self.batches_per_superbatch,
                start_superbatch,
                end_superbatch: default::TRAIN_END_SUPERBATCH,
            },
            wdl_scheduler: LinearWDL {
                start: 0.,
                end: self.wdl,
            },
            lr_scheduler: Warmup {
                warmup_batches: self.batches_per_superbatch,
                inner: LinearDecayLR {
                    initial_lr: default::TRAIN_INITIAL_LR,
                    final_lr: default::TRAIN_FINAL_LR,
                    final_superbatch: default::TRAIN_END_SUPERBATCH,
                },
            },
            save_rate: self.save_rate,
        })
    }

    fn phase_tune(&self, start_superbatch: usize) -> Result<TrainingSchedulerType, String> {
        if start_superbatch >= default::TUNE_END_SUPERBATCH {
            return Err(format!(
                "Starting tune with a superbatch higher than the TUNE_END_SUPERBATCH:{}",
                default::TUNE_END_SUPERBATCH
            ));
        }
        Ok(TrainingSchedule {
            net_id: "tune".to_string(),
            eval_scale: default::EVAL_SCALE,
            steps: TrainingSteps {
                batch_size: self.batch_size,
                batches_per_superbatch: self.batches_per_superbatch,
                start_superbatch,
                end_superbatch: default::TUNE_END_SUPERBATCH,
            },
            wdl_scheduler: LinearWDL {
                start: self.wdl,
                end: self.wdl,
            },
            lr_scheduler: Warmup {
                warmup_batches: self.batches_per_superbatch,
                inner: LinearDecayLR {
                    initial_lr: default::TUNE_INITIAL_LR,
                    final_lr: default::TUNE_FINAL_LR,
                    final_superbatch: default::TUNE_END_SUPERBATCH,
                },
            },
            save_rate: self.save_rate,
        })
    }
}
