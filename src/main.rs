use spdp::model::*;
use spdp::utils::*;
use std::time::Instant;

fn main() {
    let data = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_A1.dat");

    let start = Instant::now();

    let model = ModelRestricted::new(data); 

    let duration = start.elapsed();

    println!("Fragments: {:?}", model.fragments.len());
    println!("Time taken: {:?}", duration);
}