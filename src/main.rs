mod orchestrator;

use clap::Parser;
use orchestrator::Commands;

fn main() {
    Commands::parse().extract_orchestrator().run();
}
