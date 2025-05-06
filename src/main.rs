use spdp::model::*;
use spdp::utils::*;
use std::time::Instant;

fn main() {
    let data = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_B1.dat");

    let time = Instant::now();
    
    let mut model = ColGenModel::new(data);
    
    model.solve();

    println!("Time elapsed: {:?}", time.elapsed());
}

