use spdp::model::*;
use spdp::utils::*;
use std::time::Instant;

fn main() {

    let data = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_B11.dat");

    let time = Instant::now();
    
    let mut model = ColGenModel::new(data, None);

    // for request in model.data.requests.iter() {
    //     print!("{:?} \n", request);
    // }
    
    model.solve();

    println!("Time elapsed: {:?}", time.elapsed());
}

