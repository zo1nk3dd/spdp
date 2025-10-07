use spdp::constants::GAP_MULTIPLIER;
use spdp::model::*;
use spdp::utils::*;
use std::env;

fn main() {
    let start = std::time::Instant::now();

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

    let (zlb, vlb, v_guess, vehicle_lbs, cost_lbs) = model.solve(verbose);
    let mut zub = GAP_MULTIPLIER * zlb; // Assuming a 10% upper bound for demonstration


    println!("Upper bound guess: {}", zub);

    println!("Lower bound: {}", zlb);
    let v_count = model.max_vehicles.unwrap();
    println!("Number of vehicles: {}", v_count);

    println!("Time elapsed: {:?}", start.elapsed());

    let v_gap = v_guess - vlb;

    let vehicle_filter = vehicle_lbs.iter().map(|&lb| lb > v_gap).collect::<Vec<bool>>();

    let mut master = MasterProblemModel::new(data.clone(), model.arcs, model.nodes, v_count);

    loop {
        let filter = vehicle_filter.iter().enumerate().map(|(idx, &b)| b || (cost_lbs[idx] > zub - zlb)).collect::<Vec<bool>>();
        master.filter_arcs(filter);
        let result: Result<f64, grb::Error> = master.solve(verbose);

        match result {
            Ok(obj) => {
                // master.print_solution();
                println!("Objective value: {}", obj);
                if obj > zub {
                    println!("Objective value exceeds upper bound guess. Updating upper bound.");
                    zub = if obj > GAP_MULTIPLIER * zub { GAP_MULTIPLIER * zub } else { obj };
                    continue;
                } else {
                    break;
                }
            }
            Err(_) => {
                println!("Master problem is infeasible. Updating upper bound");
                zub = GAP_MULTIPLIER * zub;
                continue;
            }
        }
    }
    println!("CG vehicle soln: {}", vlb);
    println!("CG cost soln: {}", zlb);

    let final_filter = vehicle_filter.iter().enumerate().map(|(idx, &b)| b || (cost_lbs[idx] > zub - zlb)).collect::<Vec<bool>>();

    println!("Routes filtered: {} / {}", final_filter.iter().filter(|&&b| b).count(), final_filter.len());
    println!("Time elapsed: {:?}", start.elapsed());
}