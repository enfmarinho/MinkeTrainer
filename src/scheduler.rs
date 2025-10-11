use bullet_lib::{
    lr::{self, CosineDecayLR},
    wdl::LinearWDL,
    TrainingSchedule, TrainingSteps,
};
use clap::Parser;

#[derive(Parser)]
pub struct Scheduler {
    #[arg(short, long, default_value_t = 1)]
    start_superbatch: usize,

    #[arg(short, long, default_value_t = 800)]
    end_superbatch: usize,

    #[arg(long, default_value_t = 0.10)]
    wdl: f32,

    #[arg(long, default_value_t = 16384)]
    batch_size: usize,

    #[arg(long, default_value_t = 6104)]
    batches_per_superbatch: usize,

    #[arg(long, default_value_t = 10)]
    save_rate: usize,
}

impl Scheduler {
    pub fn get(&self) -> TrainingSchedule<CosineDecayLR, LinearWDL> {
        TrainingSchedule {
            net_id: "minke".to_string(),
            eval_scale: 400.,
            steps: TrainingSteps {
                batch_size: self.batch_size,
                batches_per_superbatch: self.batches_per_superbatch,
                start_superbatch: self.start_superbatch,
                end_superbatch: self.end_superbatch,
            },
            wdl_scheduler: LinearWDL {
                start: 0.,
                end: self.wdl,
            },
            lr_scheduler: lr::CosineDecayLR {
                initial_lr: 0.001,
                final_lr: 0.001 * 0.3f32.powi(5),
                final_superbatch: self.end_superbatch,
            },
            save_rate: self.save_rate,
        }
    }
}
