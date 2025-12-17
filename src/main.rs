mod trainer;

use clap::Parser;
use trainer::Trainer;

fn main() {
    Trainer::parse().run();
}
