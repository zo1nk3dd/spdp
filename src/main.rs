use spdp::fragment::Generator;
use spdp::model::*;
use spdp::utils::*;
use std::time::Instant;

fn main() {
    let data = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_C10.dat");

    let time = Instant::now();
    let gen = Generator::new(data); 
    let arcs = gen.generate_restricted_fragments();

    println!("Generated {} fragments in {} ms", arcs.num_arcs(), time.elapsed().as_millis());
}

