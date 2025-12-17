mod orchestrator;

use clap::Parser;
use orchestrator::Orchestrator;

fn main() {
    Orchestrator::parse().run();
}
