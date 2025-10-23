use spdp::constants::EPS;
use spdp::model::*;
use spdp::utils::*;
use std::env;
use std::vec;

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

    let mut route_ip: RouteIPModel = RouteIPModel::new(&data, &model.route_costs, &model.routes_covering_request, vlb.ceil() as usize);

    let zub = route_ip.solve(verbose).unwrap();

    println!("Upper bound guess: {}", zub);

    println!("Lower bound: {}", zlb);
    let v_count = model.max_vehicles.unwrap();
    println!("Number of vehicles: {}", v_count);

    println!("Time elapsed: {:?}", start.elapsed());

    let v_gap = v_guess - vlb;

    println!(" Vehicle gap: {}", v_gap);

    let mut filter = vec![false; vehicle_lbs.len()];
    
    for (_idx, v_lb) in vehicle_lbs.iter().enumerate() {
        if *v_lb > v_gap + EPS {
            // filter[idx] = true;
        }
    }

    let z_gap = zub - zlb;

    println!(" Cost gap: {}", z_gap);

    for (idx, cost_lb) in cost_lbs.iter().enumerate() {
        if *cost_lb > z_gap + EPS {
            filter[idx] = true;
        }
    }

    println!(
        "Filtered {:} / {} arcs before solving MP",
        filter.iter().filter(|&&b| b).count(), filter.len()
    );

    let mut master = MasterProblemModel::new(data.clone(), model.arcs, model.nodes, v_count, filter.clone());

    loop {
        let result: Result<f64, grb::Error> = master.solve(verbose, zlb, cost_lbs.clone());

        match result {  
            Ok(obj) => {
                // master.print_solution();
                println!("Objective value: {}", obj);
                break;
            }
            Err(_) => {
                println!("Master problem is infeasible. ABORTING!?!?!?!?!");
                panic!();
            }
        }
    }
    println!("CG vehicle soln: {}", vlb);
    println!("CG cost soln: {}", zlb);
    println!("Time elapsed: {:?}", start.elapsed());
}