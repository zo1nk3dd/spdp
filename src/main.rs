use spdp::model::*;
use spdp::utils::*;
use std::time::Instant;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: cargo run <instance> <--verbose>");
        return;
    }
    let instance = &args[1];
    let data = SPDPData::from_file(&format!("./SkipData/Benchmark/RecDep_day_{}.dat", instance));

    let mut verbose = false;
    if args.len() == 3 && args[2] == "--verbose" {
        println!("Verbose mode enabled");
        verbose = true;
    }

    let time = Instant::now();
    
    let mut model = ColGenModel::new(data);
    
    model.solve(verbose);

    println!("Time elapsed: {:?}", time.elapsed());
}

