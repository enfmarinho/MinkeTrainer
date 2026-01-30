use super::default;
use crate::orchestrator::get_material_count_target;
use bullet_lib::game::formats::sfbinpack::{
    TrainingDataEntry,
    chess::{r#move::MoveType, piecetype::PieceType},
};
use clap::Parser;
use rand::{Rng, rng};
use serde::Deserialize;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Parser, Clone, Debug, Deserialize)]
pub struct DataFilter {
    /// Minimum number of plies (half-moves) a game must have to be included
    #[arg(long, default_value_t = DataFilter::default().min_ply)]
    min_ply: u16,

    /// Maximum absolute eval score allowed
    #[arg(long, default_value_t = DataFilter::default().max_score)]
    max_score: u16,

    /// Whether to filter position where side to move is in check
    #[arg(long, default_value_t = DataFilter::default().exclude_in_check)]
    exclude_in_check: bool,

    /// Whether to filter positions where best move is a special move (castling or en passant)
    #[arg(long, default_value_t = DataFilter::default().exclude_special_moves)]
    exclude_special_moves: bool,

    /// Whether to filter positions where the best move is a capture
    #[arg(long, default_value_t = DataFilter::default().exclude_capture)]
    exclude_capture: bool,

    /// Enable filtering based on the total number of pieces remaining on the board
    #[arg(long, default_value_t = DataFilter::default().material_count_filter)]
    material_count_filter: bool,
}

impl Default for DataFilter {
    fn default() -> Self {
        DataFilter {
            min_ply: default::MIN_PLY,
            max_score: default::MAX_SCORE,
            exclude_in_check: default::EXCLUDE_IN_CHECK,
            exclude_special_moves: default::EXCLUDE_SPECIAL_MOVES,
            exclude_capture: default::EXCLUDE_CAPTURE,
            material_count_filter: default::OUT_BUCKET_COUNT_FILTER,
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
        static MATERIAL_COUNT_APPEARENCES: [AtomicU64; default::CHESS_PIECE_COUNT] =
            [const { AtomicU64::new(0) }; default::CHESS_PIECE_COUNT];
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
