// Default values
use std::sync::OnceLock;

/// Network
pub const HIDDEN_LAYER_SIZE: usize = 1024;
pub const N_OUTPUT_BUCKETS: usize = 8;
pub const QA: i16 = 255;
pub const QB: i16 = 64;
pub const QAB: i16 = QA * QB;

/// DataFilter
pub const MIN_PLY: u16 = 16;
pub const MAX_SCORE: u16 = 5000;
pub const EXCLUDE_IN_CHECK: bool = true;
pub const EXCLUDE_SPECIAL_MOVES: bool = true;
pub const EXCLUDE_CAPTURE: bool = true;
pub const OUT_BUCKET_COUNT_FILTER: bool = true;

/// DataLoader
pub const BUFFER_SIZE_MB: usize = 2048;

/// Scheduler
pub const WDL: f32 = 0.1;
pub const BATCH_SIZE: usize = 16384;
pub const BATCHES_PER_SUPERBATCH: usize = 6104;
pub const SAVE_RATE: usize = 10;
pub const EVAL_SCALE: f32 = 400.;

/// Pre-Train
pub const PRETRAIN_INITIAL_LR: f32 = 1e-3;
pub const PRETRAIN_FINAL_LR: f32 = 1e-4;
pub const PRETRAIN_END_SUPERBATCH: usize = 200;

/// Train
pub const TRAIN_END_SUPERBATCH: usize = 600;
pub const TRAIN_INITIAL_LR: f32 = 5e-4;
pub const TRAIN_FINAL_LR: f32 = 1e-5;

/// Tune
pub const TUNE_END_SUPERBATCH: usize = 200;
pub const TUNE_INITIAL_LR: f32 = 1e-4;
pub const TUNE_FINAL_LR: f32 = 0.;

/// Outbut buckets material count target distribution
pub const CHESS_PIECE_COUNT: usize = 32;

fn material_count_target_distribution() -> [f64; CHESS_PIECE_COUNT] {
    const SIGMA: f64 = 16.0;
    const MU: f64 = (CHESS_PIECE_COUNT - 1) as f64 / 2.0;

    let mut weights = [0.; CHESS_PIECE_COUNT];
    let mut sum = 0.0;

    // unnormalized Gaussian distribution
    for i in 0..CHESS_PIECE_COUNT {
        let x = i as f64 - MU;
        let v = (-x * x / (2.0 * SIGMA * SIGMA)).exp();
        weights[i] = v;
        sum += v;
    }

    // normalize Gaussian distribution
    for v in &mut weights {
        *v /= sum;
    }

    weights
}

pub fn get_material_count_target(index: usize) -> f64 {
    static MATERIAL_COUNT_TARGET_DISTRIBUTION: OnceLock<[f64; 32]> = OnceLock::new();
    MATERIAL_COUNT_TARGET_DISTRIBUTION.get_or_init(material_count_target_distribution)[index]
}
