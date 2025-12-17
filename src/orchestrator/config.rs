// Default values

/// Network
pub const HIDDEN_LAYER_SIZE: usize = 1024;

/// DataFilter
pub const MIN_PLY: u16 = 16;
pub const MAX_SCORE: u16 = 5000;
pub const EXCLUDE_IN_CHECK: bool = true;
pub const EXCLUDE_SPECIAL_MOVES: bool = true;
pub const EXCLUDE_CAPTURE: bool = true;

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
