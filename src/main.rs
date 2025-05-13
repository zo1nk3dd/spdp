use spdp::model::*;
use spdp::utils::*;
use std::time::Instant;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: cargo run <instance>");
        return;
    }
    let instance = &args[1];

    let data = SPDPData::from_file(&format!("./SkipData/Benchmark/RecDep_day_{}.dat", instance));

    let time = Instant::now();
    
    let mut model = ColGenModel::new(data, None);

    // for request in model.data.requests.iter() {
    //     print!("{:?} \n", request);
    // }
    
    model.solve();

    println!("Time elapsed: {:?}", time.elapsed());
}

