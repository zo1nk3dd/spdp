use spdp::model::*;
use spdp::utils::*;
use std::time::Instant;

fn main() {
    let start = Instant::now();

    let data = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_D20.dat");
    let model = ModelRestricted::new(data); 

    let duration = start.elapsed();

    println!("Model: {:?}", model.fragments.len());
    println!("Time taken: {:?}", duration);
}