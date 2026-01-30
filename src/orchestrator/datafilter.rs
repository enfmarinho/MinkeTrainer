use std::sync::atomic::{AtomicU64, Ordering};

use crate::orchestrator::get_material_count_target;

use super::config;
use bullet_lib::game::formats::sfbinpack::{
    chess::{piecetype::PieceType, r#move::MoveType},
    TrainingDataEntry,
};
use clap::Parser;
use rand::{rng, Rng};
use serde::Deserialize;

#[derive(Parser, Clone, Debug, Deserialize)]
pub struct DataFilter {
    #[arg(long, default_value_t = DataFilter::default().min_ply)]
    min_ply: u16,

    #[arg(long, default_value_t = DataFilter::default().max_score)]
    max_score: u16,

    #[arg(long, default_value_t = DataFilter::default().exclude_in_check)]
    exclude_in_check: bool,

    #[arg(long, default_value_t = DataFilter::default().exclude_special_moves)]
    exclude_special_moves: bool,

    #[arg(long, default_value_t = DataFilter::default().exclude_capture)]
    exclude_capture: bool,

    #[arg(long, default_value_t = DataFilter::default().material_count_filter)]
    material_count_filter: bool,
}

impl Default for DataFilter {
    fn default() -> Self {
        DataFilter {
            min_ply: config::MIN_PLY,
            max_score: config::MAX_SCORE,
            exclude_in_check: config::EXCLUDE_IN_CHECK,
            exclude_special_moves: config::EXCLUDE_SPECIAL_MOVES,
            exclude_capture: config::EXCLUDE_CAPTURE,
            material_count_filter: config::OUT_BUCKET_COUNT_FILTER,
        }
    }
}

impl DataFilter {
    pub fn filter(&self, entry: &TrainingDataEntry) -> bool {
        entry.ply >= self.min_ply
            && entry.score.unsigned_abs() <= self.max_score
            && (!self.material_count_filter || self.material_count_filter(entry))
            && (!self.exclude_capture
                || entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None)
            && (!self.exclude_special_moves || entry.mv.mtype() == MoveType::Normal)
            && (!self.exclude_in_check || !entry.pos.is_checked(entry.pos.side_to_move()))
    }

    fn material_count_filter(&self, entry: &TrainingDataEntry) -> bool {
        const C: f64 = 0.6;
        static MATERIAL_COUNT_APPEARENCES: [AtomicU64; config::CHESS_PIECE_COUNT] =
            [const { AtomicU64::new(0) }; config::CHESS_PIECE_COUNT];
        static TOTAL_POSITIONS: AtomicU64 = AtomicU64::new(0);

        let mc = entry.pos.occupied().count() as usize - 1; // 0 indexed
        let mc_appearences = MATERIAL_COUNT_APPEARENCES[mc].fetch_add(1, Ordering::Relaxed) + 1;
        let total_appearences = TOTAL_POSITIONS.fetch_add(1, Ordering::Relaxed) + 1;
        let observed_frequency = mc_appearences as f64 / total_appearences as f64;
        let mc_desired_distribution = get_material_count_target(mc);

        let rejection_probability =
            1. - (C * mc_desired_distribution / observed_frequency).clamp(0., 1.);

        rng().random_bool(rejection_probability)
    }
}
