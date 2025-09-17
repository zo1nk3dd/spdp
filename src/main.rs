use spdp::model::*;
use spdp::utils::*;
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

    println!("Solving for the lower bound");

    let mut model = ColGenModel::new(data.clone());

    let zlb = model.solve(verbose);
    let mut zub = 1.01 * zlb; // Assuming a 10% upper bound for demonstration


    println!("Upper bound guess: {}", zub);

    println!("Lower bound: {}", zlb);
    let v_count = model.max_vehicles.unwrap();
    println!("Number of vehicles: {}", v_count);

    let lbs = model.get_lower_bounds(verbose);

    let mut master = MasterProblemModel::new(data.clone(), model.arcs, model.nodes, v_count);

    loop {
        master.filter_arcs(&lbs, zlb, zub);

        let result: Result<f64, grb::Error> = master.solve(verbose);

        match result {
            Ok(obj) => {
                // master.print_solution();
                println!("Objective value: {}", obj);
                if obj > zub {
                    println!("Objective value exceeds upper bound guess. Updating upper bound.");
                    zub = if obj > 1.01 * zub { 1.01 * zub } else { obj };
                    continue;
                } else {
                    break;
                }
            }
            Err(e) => {
                println!("Master problem is infeasible. Updating upper bound");
                zub = 1.02 * zub;
                continue;
            }
        }
    }

    println!("Optimal objective value: {}", master.model.get_attr(grb::attr::ObjVal).unwrap());
}

