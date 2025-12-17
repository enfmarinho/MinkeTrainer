use super::config;
use bullet_lib::{
    game::formats::{
        bulletformat::ChessBoard,
        sfbinpack::{
            chess::{piecetype::PieceType, r#move::MoveType},
            TrainingDataEntry,
        },
    },
    value::loader,
};
use clap::Parser;

#[derive(Parser, Clone, Debug)]
pub struct DataFilter {
    #[arg(long, default_value_t = config::MIN_PLY)]
    min_ply: u16,

    #[arg(long, default_value_t = config::MAX_SCORE)]
    max_score: u16,

    #[arg(long, default_value_t = config::EXCLUDE_IN_CHECK)]
    exclude_in_check: bool,

    #[arg(long, default_value_t = config::EXCLUDE_SPECIAL_MOVES)]
    exclude_special_moves: bool,

    #[arg(long, default_value_t = config::EXCLUDE_CAPTURE)]
    exclude_capture: bool,
}

impl DataFilter {
    pub fn filter(&self, entry: &TrainingDataEntry) -> bool {
        let mut valid_entry = true;
        if self.exclude_capture {
            valid_entry &= entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None;
        }
        if self.exclude_special_moves {
            valid_entry &= entry.mv.mtype() == MoveType::Normal;
        }
        if self.exclude_in_check {
            valid_entry &= !entry.pos.is_checked(entry.pos.side_to_move());
        }

        valid_entry && entry.ply >= self.min_ply && entry.score.unsigned_abs() <= self.max_score
    }
}

#[derive(Parser, Debug)]
pub struct DataLoader {
    #[arg(long, default_value_t = config::BUFFER_SIZE_MB)]
    buffer_size_mb: usize,
}

impl DataLoader {
    pub fn load(
        &self,
        datasets: &[String],
        filter: &DataFilter,
        threads: usize,
    ) -> impl loader::DataLoader<ChessBoard> {
        let datasets = Vec::from_iter(datasets.iter().map(|s| s.as_str()));
        let filter = filter.clone();

        loader::SfBinpackLoader::new_concat_multiple(
            &datasets,
            self.buffer_size_mb,
            threads,
            move |entry| filter.filter(entry),
        )
    }
}
