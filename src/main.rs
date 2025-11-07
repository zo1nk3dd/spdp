use spdp::constants::EPS;
use spdp::constants::NUM_ROUTES_FOR_RIP;
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
    };

    let mut model = ColGenModel::new(data.clone());

    let (mut zlb, vlb, mut v_guess, vehicle_lbs, mut cost_lbs, mut route_arcs) = model.solve(verbose);

    let mut best_sol: Vec<Vec<usize>> = Vec::new();

    route_arcs.sort_by(|a, b| {
        a.0.reduced_cost.total_cmp(&b.0.reduced_cost)
    });

    let mut route_costs = Vec::new();
    let mut routes_covering_request: Vec<Vec<usize>> = vec![Vec::new(); data.requests.len()];

    for arc in route_arcs.iter().take(NUM_ROUTES_FOR_RIP as usize) {
        route_costs.push(arc.0.cost as f64);
        for (idx, cnt) in arc.0.coverset.to_vec().iter().enumerate() {
            routes_covering_request[idx].push(*cnt);
        }
    }

    let zub: f64;

    loop {
        let mut route_ip: RouteIPModel = RouteIPModel::new(&data, &route_costs, &routes_covering_request, v_guess as usize);

        let (result, route_sol) = route_ip.solve(verbose);

        if result.is_ok() {
            zub = result.unwrap();
            println!("Upper bound found: {}", zub);
            for (idx, obj) in route_sol.iter().enumerate() {
                if *obj > EPS {
                    best_sol.push(route_arcs[idx].1.clone());
                }
            }
            break;
        }

        println!("Route IP infeasible, continuing column generation...");

        (zlb, v_guess, cost_lbs) = model.solve_with_increased_vehicles(verbose, vlb, &vehicle_lbs);
    }

    // Extract the solution information
    for (v_id, route) in best_sol.iter().enumerate() {
        println!("Vehicle {}: Route {:?}", v_id + 1, route);
        for &arc_id in route {
            let arc = model.arcs.get_arc(arc_id);
            println!("  Arc {}: from node {} to node {}", arc.id, arc.start.id, arc.end.id);
        }
    }

    println!("Upper bound guess: {}", zub);

    println!("Lower bound: {}", zlb);
    let v_count = model.max_vehicles.unwrap();
    println!("Number of vehicles: {}", v_count);

    println!("Time elapsed: {:?}", start.elapsed());

    let v_gap = v_guess - vlb;

    println!(" Vehicle gap: {}", v_gap);

    let mut filter = vec![false; cost_lbs.len()];
    
    for (_idx, v_lb) in cost_lbs.iter().enumerate() {
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

    for arc_id in best_sol.iter().flatten() {
        filter[*arc_id] = false;
    }

    println!(
        "Filtered {:} / {} arcs before solving MP",
        filter.iter().filter(|&&b| b).count(), filter.len()
    );

    let mut master = MasterProblemModel::new(data.clone(), model.arcs, model.nodes, v_count, filter.clone());

    println!("Best solution so far: {:?}", best_sol);

    loop {
        let result: Result<f64, grb::Error> = master.solve(verbose, zlb, cost_lbs.clone(), best_sol);

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