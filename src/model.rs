extern crate grb;

use std::collections::HashMap;
use std::collections::HashSet;
use itertools::Itertools;

use grb::attribute::ConstrDoubleAttr::RHS;
use grb::attribute::ConstrIntAttr::IISConstr;
use grb::attribute::ModelDoubleAttr::ObjVal;
use grb::attribute::ModelModelSenseAttr::ModelSense;
use grb::attribute::VarDoubleAttr::Obj;
use grb::attribute::VarDoubleAttr::UB;
use grb::attribute::VarDoubleAttr::X;
use grb::attribute::VarIntAttr::IISLB;
use grb::attribute::VarIntAttr::IISUB;
use grb::callback;
use grb::callback::CbResult;
use grb::callback::MIPSolCtx;
use grb::parameter::IntParam::LazyConstraints;
use grb::parameter::IntParam::OutputFlag;
use grb::parameter::StrParam::ResultFile;
use grb::prelude::*;
use grb::Error;

use crate::model;

use super::utils::*;
use super::fragment::*;
use super::pricing::*;
use super::constants::*;


struct CallbackContext<'a> {
    data: &'a SPDPData,
    nodes: &'a NodeContainer,
    arcs: &'a ArcContainer,
    x: &'a Vec<Vec<Var>>,
    verbose: bool,
}

impl<'a> CallbackContext<'a> {
    fn new(data: &'a SPDPData, nodes: &'a NodeContainer, arcs: &'a ArcContainer, x: &'a Vec<Vec<Var>>, verbose: bool) -> Self {
        Self {
            data,
            nodes,
            arcs,
            x,
            verbose,
        }
    }

    fn callback_mipsol(&mut self, ctx: MIPSolCtx) {
        for (_, arcs) in self.x.iter().enumerate() {
            let soln = ctx.get_solution(arcs).unwrap();
            let mut sets: Vec<HashSet<usize>> = Vec::new();
            for arc_id in 0..arcs.len() {
                let val = soln[arc_id];
                if val > EPS {
                    let nodes: Vec<usize> = vec![
                        self.arcs.arcs[arc_id].start.id,
                        self.arcs.arcs[arc_id].end.id,
                    ];
                    sets.push(HashSet::from_iter(nodes));
                }
            }

            // See if any of these overlap
            // Iterate over every pair of sets
            let mut i = 0;
            let mut j = 1;

            while i < sets.len() - 1 {
                if !sets[i].is_disjoint(&sets[j]) {
                    // Found an overlap
                    let set1 = sets.remove(i);
                    let set2 = sets.remove(j-1);
                    let mut new_set = HashSet::new();
                    for x in set1.union(&set2) {
                        new_set.insert(*x);
                    }
                    sets.push(new_set);

                    i = 0;
                    j = 1;
                    continue;
                }
                j += 1;
                if j >= sets.len() {
                    i += 1;
                    j = i + 1;
                }
            }

            for set in sets {
                if !set.contains(&0) {
                    if self.verbose { println!("Cutting cycle of {:?}", set); }
                    // Create a cut
                    for k in 0..self.x.len() {
                        let arcs_enter = set.iter().map(|node_id| {
                            let node = self.nodes.nodes.get(*node_id).unwrap();
                            self.arcs.arcs_to.get(node).unwrap().iter()
                                .filter_map(|arc| {
                                    if !set.contains(&arc.start.id) {
                                        Some((self.data.t_limit - arc.time) * self.x[k][arc.id])
                                    } else {
                                        None
                                    }
                                }).collect::<Vec<_>>()
                        }).flatten().grb_sum();

                        let arcs_leave = set.iter().map(|node_id| {
                            let node = self.nodes.nodes.get(*node_id).unwrap();
                            self.arcs.arcs_from.get(node).unwrap().iter()
                                .filter_map(|arc| {
                                    if !set.contains(&arc.end.id) {
                                        Some(arc.time * self.x[k][arc.id])
                                    } else {
                                        None
                                    }
                                }).collect::<Vec<_>>()
                        }).flatten().grb_sum();

                        let arcs_inside = set.iter().map(|node_id| {
                            let node = self.nodes.nodes.get(*node_id).unwrap();
                            self.arcs.arcs_from.get(node).unwrap().iter()
                                .filter_map(|arc| {
                                    if set.contains(&arc.end.id) {
                                        Some(arc.time * self.x[k][arc.id])
                                    } else {
                                        None
                                    }
                                }).collect::<Vec<_>>()
                        }).flatten().grb_sum();

                        ctx.add_lazy(c!(arcs_inside + arcs_leave <= arcs_enter)).unwrap();
                    }
                }
            }
        }
    }
}

impl callback::Callback for CallbackContext<'_> {
    fn callback(&mut self, where_: Where) -> CbResult {
        match where_ {
            Where::MIPSol(ctx) => {
                // Handle MIP solution context
                self.callback_mipsol(ctx);
            }
            _ => {}
        }
        CbResult::Ok(())
    }
}


pub struct MasterProblemModel {
    pub data : SPDPData,
    pub arcs : ArcContainer,
    pub nodes: NodeContainer,
    pub model: Model,
    pub x: Vec<Vec<Var>>,
    pub cover: Vec<Constr>,
    pub flow: Vec<Vec<Constr>>,
    pub time_lim: Vec<Constr>,
    pub set_z: Vec<Constr>,
    pub sym_z: Vec<Constr>,
    pub a: Vec<Vec<f64>>,
    pub vehicle_limit: Constr,
    pub single_request_symmetry: Vec<Constr>,
}

impl MasterProblemModel {
    pub fn new(data: SPDPData, arc_container: ArcContainer, node_container: NodeContainer, vehicle_count: usize) -> Self {

        let mut model = Model::new("Master Problem").unwrap();

        let mut a: Vec<Vec<f64>> = vec![vec![0.0; arc_container.num_arcs()]; data.num_requests];

        for arc in arc_container.arcs.iter() {
            if arc.done.len() > 0 {
                a[arc.done.left()][arc.id] += 1.0;
            }
            if arc.done.len() == 2 {
                a[arc.done.right()][arc.id] += 1.0;
            }
        }

        let x: Vec<Vec<Var>> = (0..vehicle_count)
            .map(|vehicle_id: usize| {
                (0..arc_container.num_arcs()).map(|arc_id| {
                    let arc = &arc_container.arcs[arc_id];
                    add_var!(model, Integer, name: &format!("a_{arc_id}_v{vehicle_id}"), obj: arc.cost as f64).unwrap()
                }).collect()
            })
            .collect();

        let z: Vec<Var> = (0..vehicle_count).map(|vehicle_id| {
            add_var!(model, Binary, name: &format!("z_{vehicle_id}"), obj: 0.0).unwrap()
        }).collect();

        let cover: Vec<Constr> = data.requests.iter()
            .enumerate()
            .map(|(r_id, request)| {
                let constr_expr = x.iter().enumerate()
                    .map(|(_, xs)| {
                        xs.iter().zip(a[r_id].iter()).map(|(arc, coeff)| *coeff * *arc).grb_sum()
                    })
                    .grb_sum();

                let quantity = request.quantity as f64;

                model.add_constr(
                    &format!("request_{}", r_id),
                    c!(constr_expr == quantity)
                ).unwrap()
            })
            .collect();

        let flow: Vec<Vec<Constr>> = node_container.nodes
            .iter()
            .enumerate()
            .map(|(node_id, node)| {
                (0..vehicle_count).map(|vehicle_id| {
                    let inflow = arc_container.arcs_to.get(node).unwrap()
                        .iter()
                        .map(|arc| x[vehicle_id][arc.id])
                        .grb_sum();

                    let outflow = arc_container.arcs_from.get(node).unwrap()
                        .iter()
                        .map(|arc| x[vehicle_id][arc.id])
                        .grb_sum();

                    model.add_constr(
                        &format!("flow_{}_{}", node_id, vehicle_id),
                        c!(inflow == outflow)
                    ).unwrap()
                }).collect()
            })
            .collect();

        let time_lim: Vec<Constr> = (0..vehicle_count)
            .map(|vehicle_id| {
                let lhs = arc_container.arcs.iter().map(|arc| {
                    arc.time * x[vehicle_id][arc.id]
                }).grb_sum();
                model.add_constr(
                    &format!("time_lim_{}", vehicle_id),
                    c!(lhs <= data.t_limit as f64 * z[vehicle_id])
                ).unwrap()
            })
            .collect();

        let set_z: Vec<Constr> = (0..vehicle_count)
            .map(|vehicle_id| {
                let rhs = arc_container.arcs_from.get(&node_container.depot).unwrap()
                    .iter()
                    .map(|arc| x[vehicle_id][arc.id])
                    .grb_sum();
                model.add_constr(
                    &format!("set_z_{}", vehicle_id),
                    c!(z[vehicle_id] == rhs)
                ).unwrap()
            })
            .collect();

        let sym_z: Vec<Constr> = (1..vehicle_count)
            .map(|vehicle_id| {
                let lhs = z[vehicle_id];
                let rhs = z[vehicle_id - 1];
                model.add_constr(
                    &format!("sym_z_{}", vehicle_id),
                    c!(lhs <= rhs)
                ).unwrap()
            })
            .collect();

        let vehicle_limit = model.add_constr(
            "vehicle_limit",
            c!(z.iter().grb_sum() == vehicle_count as f64)
        ).unwrap();

        // Inequalities
        let single_request_symmetry: Vec<Constr> = data.requests.iter().enumerate()
            .filter(|(idx, r)| r.quantity == 1 && idx < &vehicle_count)
            .map(|(idx, r)| {
                let lhs = (0..vehicle_count).filter(|k| k > &idx).map(|k| {
                    arc_container.arcs.iter().filter_map(|arc| {
                        if a[idx][arc.id] == 1.0 {
                            Some(x[k][arc.id])
                        } else {
                            None
                        }
                    }).grb_sum()
                }).grb_sum();
                model.add_constr(
                    &format!("single_request_symmetry_{}", idx),
                    c!(lhs == 0.0)
                ).unwrap()
            }).collect();

        // Subset-row

        let basic_odd_requests: Vec<Constr> = data.requests.iter().enumerate()
            .filter(|(r_id, r)| r.quantity % 2 == 1 || r.quantity > 2)
            .map(|(r_id, r)| {
                let lhs = (0..vehicle_count).map(|k| {
                    arc_container.arcs.iter().filter_map(|arc| {
                        if a[r_id][arc.id] == 2.0 {
                            Some(x[k][arc.id])
                        } else {
                            None
                        }
                    }).grb_sum()
                }).grb_sum();
                let rhs = (r.quantity / 2) as f64;

                model.add_constr(
                    &format!("odd_request_{}", r_id),
                    c!(lhs <= rhs)
                ).unwrap()
            }).collect();

        for location in node_container.pickup_nodes.iter() {
            let requests_at_location: Vec<(usize, &Request)> = data.requests.iter().enumerate()
                .filter(|(r_id, r)| r.from_id == location.id)
                .collect();

            let quantity_sum = requests_at_location.iter().map(|(r_id, r)| r.quantity).sum::<usize>();

            if quantity_sum > 2 && quantity_sum % 2 == 1 && requests_at_location.len() > 1 {
                let lhs = (0..vehicle_count).map(|k| {
                    arc_container.arcs.iter().filter_map(|arc| {
                        let q = requests_at_location.iter().map(|(r_id, r)| a[*r_id][arc.id]).sum::<f64>();
                        if q == 2.0 {
                            Some(x[k][arc.id])
                        } else {
                            None
                        }
                    }).grb_sum()
                }).grb_sum();
                let rhs = (quantity_sum / 2) as f64;

                model.add_constr(
                    &format!("odd_location_{}", location.id),
                    c!(lhs <= rhs)
                ).unwrap();   
            }
        }

        // For each subset of the requests of size 2 or 3
        let request_ids: Vec<usize> = (0..data.num_requests).collect();
        let subsets_2 = request_ids.iter().combinations(2);
        let subsets_3 = request_ids.iter().combinations(3);

        for set in subsets_2.chain(subsets_3) {
            if set.iter().map(|r_id| data.requests[**r_id].quantity).sum::<usize>() != 3 {
                continue; // All requests must be from different locations
            }

            let treatment = data.requests[*set[0]].to_id;

            if set.iter().all(|r_id| data.requests[**r_id].to_id == treatment) {
                // add the constraint
                let lhs = (0..vehicle_count).map(|k| {
                    arc_container.arcs.iter().filter_map(|arc| {
                        let q = set.iter().map(|r_id| a[**r_id][arc.id]).sum::<f64>();
                        if q == 2.0 {
                            Some(x[k][arc.id])
                        } else {
                            None
                        }
                    }).grb_sum()
                }).grb_sum();
                model.add_constr(
                    &format!("treatment_{}", set.iter().map(|r_id| r_id.to_string()).collect::<Vec<String>>().join("_")),
                    c!(lhs <= 1.0)
                ).unwrap();
            }

            let deliver = data.requests[*set[0]].from_id;

            if set.iter().all(|r_id| data.requests[**r_id].from_id == deliver) {
                // add the constraint
                let lhs = (0..vehicle_count).map(|k| {
                    arc_container.arcs.iter().filter_map(|arc| {
                        let q = set.iter().map(|r_id| a[**r_id][arc.id]).sum::<f64>();
                        if q == 2.0 {
                            Some(x[k][arc.id])
                        } else {
                            None
                        }
                    }).grb_sum()
                }).grb_sum();
                model.add_constr(
                    &format!("deliver_{}", set.iter().map(|r_id| r_id.to_string()).collect::<Vec<String>>().join("_")),
                    c!(lhs <= 1.0)
                ).unwrap();
            }
        }

        Self {
            data,
            arcs: arc_container,
            nodes: node_container,
            model,
            x,
            cover,
            flow,
            time_lim,
            set_z,
            sym_z,
            a,
            vehicle_limit,
            single_request_symmetry
        }
    }

    pub fn filter_arcs(&mut self, lbs: &Vec<f64>, zlb: f64, zub: f64) {
        let mut filtered = 0;
        for (arc_id, lb) in lbs.iter().enumerate() {
            let arc = &self.arcs.arcs[arc_id];
            assert!(arc_id == arc.id);
            if *lb > (zub - zlb) + EPS {
                for vehicle_id in 0..self.x.len() {
                    self.model.set_obj_attr(UB, &self.x[vehicle_id][arc_id], 0.0).unwrap();
                }
                filtered += 1;
            } else {
                for vehicle_id in 0..self.x.len() {
                    self.model.set_obj_attr(UB, &self.x[vehicle_id][arc_id], 1e20).unwrap();
                }
            }
        }
        println!("Filtered {}/{} arcs from the master problem", filtered, lbs.len());
    }

    pub fn solve(&mut self, verbose: bool) -> Result<f64, Error> {
        self.model.set_attr(ModelSense, Minimize).unwrap();
        self.model.set_param(LazyConstraints, 1).unwrap();
        let mut callback_context = CallbackContext::new(&self.data, &self.nodes, &self.arcs, &self.x, verbose);
        self.model.optimize_with_callback(&mut callback_context).unwrap();

        self.model.get_attr(attr::ObjVal)
    }

    pub fn print_solution(&self) {
        // Print the solution details
        for (k, vars) in self.x.iter().enumerate() {
            for (i, lambda) in vars.iter().enumerate() {
                let obj = self.model.get_obj_attr(X, lambda).unwrap();
                if obj > EPS {
                    println!("Vehicle {}, Arc {}: {}", k, i, obj);
                    println!("    Arc details: start: {:?}, end: {:?}, cost: {}, time: {}", self.arcs.arcs[i].start, self.arcs.arcs[i].end, self.arcs.arcs[i].cost, self.arcs.arcs[i].time);
                }
            }
        }
    }
}

pub struct ColGenModel {
    pub data: SPDPData,
    pub arcs: ArcContainer,
    pub nodes: NodeContainer,
    pub model: Model,
    pub routes_covering_request: Vec<Vec<usize>>,
    pub lambda: Vec<Var>,
    pub route_costs: Vec<f64>,
    pub cover_constraints: Vec<Constr>,
    pub vehicle_constraint: Option<Constr>,
    pub max_vehicles: Option<usize>,
}

impl ColGenModel {
    pub fn new(data: SPDPData) -> Self {
        let generator = Generator::new(data.clone());

        let node_container = generator.generate_nodes();
        let arc_container = generator.generate_arcs(&node_container);

        let model = Model::new("COLGEN Problem").unwrap();

        let mut result = ColGenModel {
            data: data.clone(),
            arcs: arc_container,
            nodes: node_container,
            model,
            routes_covering_request: vec![Vec::new(); data.num_requests],
            lambda: Vec::new(),
            route_costs: Vec::new(),
            cover_constraints: Vec::new(),
            vehicle_constraint: None,
            max_vehicles: None,
        };

        result.first_initialisation();
        result.initialise_model_constraints();

        result
    }

    pub fn add_route_var(&mut self, cost: f64, covered: Vec<usize>) {
        // Add a new route to the model
        let route_id = self.lambda.len();

        // let model = &mut self.model;

        for request_id in 0..self.data.num_requests {
            self.routes_covering_request[request_id].push(0);
        }

        for (request_id, amount) in covered.iter().enumerate() {
            self.routes_covering_request[request_id][route_id] += amount;
        }

        let mut col_coeff: Vec<(Constr, f64)> = self.cover_constraints.iter()
            .map(|c| *c)
            .zip(covered.iter().map(|q| *q as f64).collect::<Vec<f64>>())
            .collect();

        if self.vehicle_constraint.is_some() {
            col_coeff.push((self.vehicle_constraint.as_ref().unwrap().clone(), 1.0));
        };
        let lambda = self.model.add_var(
            &format!("route_{route_id}"),
            Continuous,
            if self.max_vehicles.is_some() {cost} else {1.0},
            0.0,
            grb::INFINITY,
            col_coeff,
        ).unwrap();

        self.lambda.push(lambda);
        self.route_costs.push(cost);
    }

    pub fn initialise_model_constraints(&mut self) {
        self.cover_constraints = (0..self.data.num_requests)
            .map(|request_id| {
                let constr_expr = self.lambda.iter()
                    .zip(self.routes_covering_request[request_id].iter())
                    .map(|(lambda, a)| (*a as f64 * *lambda))
                    .grb_sum();

                self.model.add_constr(
                    &format!("request_{}", request_id),
                    c!(constr_expr >= self.data.requests[request_id].quantity as f64)
                ).unwrap()
            })
            .collect::<Vec<_>>();

        self.model.update().unwrap();
    }

    pub fn first_initialisation(&mut self) {
        let mut initial_routes: Vec<(usize, Vec<usize>)> = Vec::new();
        for arc in self.arcs.arcs.iter() {
            if arc.start.is_pickup() && arc.end.is_depot() {
                let mut covered = vec![0; self.data.num_requests];
                if arc.done.len() > 0 {
                    covered[arc.done.left()] += 1;
                }
                if arc.done.len() == 2 {
                    covered[arc.done.right()] += 1;
                }

                initial_routes.push((arc.cost + self.data.fixed_vehicle_cost, covered));
            }
        }

        for (cost, covered) in initial_routes.iter() {
            self.add_route_var(*cost as f64, covered.clone());
        }
    }

    fn _print_solution(&self) {
        let obj = self.model.get_attr(ObjVal).unwrap();
        println!("Objective value: {:?}", obj);

        for (idx, var) in self.lambda.iter().enumerate() {
            let value = self.model.get_obj_attr(X, var).unwrap();
            if value < 0.0001 {
                continue;
            }
            let covered: Vec<f64> = (0..self.data.num_requests).map(|idx| self.model.get_coeff(var, &self.cover_constraints[idx]).unwrap()).collect();
            print!("Variable: {:?}, value: {:?}, covered: {:?}, cost: {:?}\n", idx, value, covered, self.model.get_obj_attr(Obj, var).unwrap());
        }
    }

    /// Solves the column generation model
    /// # Arguments
    /// * `verbose` - If true, prints detailed information about the solving process
    /// # Returns
    /// The optimal objective value found by the model
    pub fn solve(&mut self, verbose: bool) -> f64 {
        self.model.set_param(OutputFlag, 0).unwrap();
        self.model.set_attr(ModelSense, Minimize).unwrap();

        let mut iter = 1;

        let mut best_vehicle_count = 0;

        let mut mode = LPSolvePhase::VehicleNoCover;

        let mut route_pool: Vec<Label> = Vec::new();

        loop {
            self.model.update().unwrap();
            self.model.optimize().unwrap();

            let result = self.model.get_attr(attr::Status).unwrap();
            
            if result == grb::Status::Infeasible {
                if self.vehicle_constraint.is_some() {
                    println!("No feasible solution found after {} iterations\n", iter);
                    println!("Updating vehicle count");
                    best_vehicle_count += 1;
                    self.model.set_obj_attr(RHS, &self.vehicle_constraint.unwrap(), best_vehicle_count as f64).unwrap();
                    continue;
                }
            }

            let mut new_routes: Vec<Label> = Vec::new();            
            let obj = self.model.get_attr(attr::ObjVal).unwrap();
            println!("OBJVAL: {:?}", obj);
            if self.vehicle_constraint.is_none() && obj - EPS < 1.0 {
                if verbose {
                    println!("Smallest possible vehicles count of 1 is found");
                }
                println!("Optimal number of vehicles found after {} iterations", iter);
                let obj = self.model.get_attr(attr::ObjVal).unwrap();
                best_vehicle_count = obj.ceil() as usize;
                self.max_vehicles = Some(best_vehicle_count);

                self.vehicle_constraint = Some(self.model.add_constr(
                    "vehicle_constraint",
                    c!(self.lambda.iter().grb_sum() >= best_vehicle_count as f64)
                ).unwrap());

                for (idx, var) in self.lambda.iter().enumerate() {
                    self.model.set_obj_attr(attr::Obj, var, self.route_costs[idx]).unwrap();
                }
                mode = LPSolvePhase::CostNoCover; // This will skip straight to cost optimization
                if verbose {
                    println!("\nCOST OPTIMIZATION PHASE");
                }
                continue;
            } else {
                let cover_duals = self.cover_constraints.iter()
                    .map(|c| self.model.get_obj_attr(attr::Pi, c).unwrap())
                    .collect::<Vec<_>>();

                let vehicle_dual = if self.max_vehicles.is_some() {
                    Some(self.model.get_obj_attr(attr::Pi, self.vehicle_constraint.as_ref().unwrap()).unwrap())
                } else {
                    None
                };

                if verbose {
                    println!("Cover duals: {:?}", cover_duals);
                    if let Some(dual) = vehicle_dual {
                        println!("Vehicle dual: {:?}", dual);
                    }
                }

                // Check the route pool for any candidate routes

                for label in route_pool.iter() {
                    let reduced_cost = match mode {
                        LPSolvePhase::VehicleNoCover | LPSolvePhase::VehicleCover | LPSolvePhase::VehicleQuantity=> {
                            1.0 - label.coverset.to_vec().iter().enumerate()
                                .map(|(idx, amount)| *amount as f64 * cover_duals[idx])
                                .sum::<f64>()
                        },
                        LPSolvePhase::CostNoCover | LPSolvePhase::CostCover | LPSolvePhase::CostQuantity => {
                            label.cost as f64
                                - label.coverset.to_vec().iter().enumerate()
                                    .map(|(idx, amount)| *amount as f64 * cover_duals[idx])
                                    .sum::<f64>()
                                - vehicle_dual.unwrap_or(0.0)
                        },
                    };

                    if reduced_cost < -EPS {
                        if verbose {
                            println!("Route from pool with reduced cost: {:.4}", reduced_cost);
                        }
                        new_routes.push(Label {
                            id: label.id,
                            reduced_cost,
                            duration: label.duration,
                            predecessor: label.predecessor,
                            cost: label.cost,
                            coverset: label.coverset,
                            node_id: label.node_id,
                            in_arc: label.in_arc,
                        });
                    }
                }

                if !new_routes.is_empty() {
                    if verbose {
                        println!("Found {} candidate routes in the pool", new_routes.len());
                        new_routes.sort_by(|a, b| a.reduced_cost.partial_cmp(&b.reduced_cost).unwrap());
                        new_routes.truncate(NUM_ROUTES_PER_NODE_ADDED * self.nodes.nodes.len());
                        // panic!("Candidate routes found in the pool, not implemented yet");
                    }
                } else {
                    if verbose {
                        println!("No candidate routes found in the pool");
                    }

                    let mut pricer: Box<dyn Pricer> = Box::new(BucketPricer::new(
                        &self.data,
                        &self.nodes,
                        &self.arcs,
                        cover_duals,
                        vehicle_dual,
                        mode,
                    ));

                    let mut candidates = pricer.solve_pricing_problem(verbose);

                    for node_id in 0..self.nodes.nodes.len() {
                        let node_routes = &mut candidates[node_id];
                        
                        node_routes.sort_by(|a, b| a.reduced_cost.partial_cmp(&b.reduced_cost).unwrap());

                        for i in 0..NUM_ROUTES_PER_NODE_ADDED {
                            let label = if i < node_routes.len() {
                                node_routes[i]
                            } else {
                                break;
                            };

                            if label.reduced_cost <= -EPS {
                                new_routes.push(label);
                            } else {
                                route_pool.push(label);
                            }
                        }

                        for i in NUM_ROUTES_PER_NODE_ADDED..NUM_ROUTES_PER_NODE_CALCULATED {
                            if i < node_routes.len() {
                                let label = node_routes[i];
                                route_pool.push(label);
                            } else {
                                break;
                            }
                        }
                    }


                    // if self.vehicle_constraint.is_none() && !new_routes.is_empty() {
                    //     let obj = self.model.get_attr(attr::ObjVal).unwrap();
                    //     let best_rc = new_routes.iter()
                    //         .map(|r| r.reduced_cost)
                    //         .min_by(|a, b| a.partial_cmp(b).unwrap())
                    //         .unwrap();
                    //     let reduction_till_jump = obj - obj.floor() - EPS;
                    //     if reduction_till_jump >= -best_rc * 2.0 {
                    //         println!("New routes can't improve the objective value enough");
                    //         println!("Current objective value: {}, best reduced cost: {}", obj, best_rc);
                    //         new_routes.clear();
                    //     }
                    // }
                }
            }

            if new_routes.is_empty() {
                println!("No new routes found");


                match mode {
                    LPSolvePhase::VehicleNoCover => {
                        mode = LPSolvePhase::VehicleQuantity;
                        println!("Adding quantity checks for vehicles");
                    },
                    LPSolvePhase::VehicleQuantity => {
                        mode = LPSolvePhase::VehicleCover;
                        println!("Adding coverage checks for vehicles");
                    },
                    LPSolvePhase::VehicleCover => {
                        mode = LPSolvePhase::CostNoCover;
                        println!("Optimal number of vehicles found after {} iterations", iter);
                        let obj = self.model.get_attr(attr::ObjVal).unwrap();
                        best_vehicle_count = obj.ceil() as usize;
                        self.max_vehicles = Some(best_vehicle_count);

                        self.vehicle_constraint = Some(self.model.add_constr(
                            "vehicle_constraint",
                            c!(self.lambda.iter().grb_sum() >= best_vehicle_count as f64)
                        ).unwrap());

                        for (idx, var) in self.lambda.iter().enumerate() {
                            self.model.set_obj_attr(attr::Obj, var, self.route_costs[idx]).unwrap();
                        }

                        println!("\nCOST OPTIMIZATION PHASE");
                    },
                    LPSolvePhase::CostNoCover => {
                        mode = LPSolvePhase::CostQuantity;
                        println!("Adding quantity checks for costs");
                    },
                    LPSolvePhase::CostQuantity => {
                        mode = LPSolvePhase::CostCover;
                        println!("Adding coverage checks for costs");
                    },
                    LPSolvePhase::CostCover => {
                        // Successfully solved the second phase of the problem
                        println!("Optimal obj {} found after {} iterations", self.model.get_attr(ObjVal).unwrap(), iter);
                        return self.model.get_attr(attr::ObjVal).unwrap();
                    },
                }

                // match mode {
                //     LPSolvePhase::VehicleNoCover => {
                //         mode = LPSolvePhase::CostNoCover;
                //         println!("Optimal number of vehicles found after {} iterations", iter);
                //         let obj = self.model.get_attr(attr::ObjVal).unwrap();
                //         best_vehicle_count = obj.ceil() as usize;
                //         self.max_vehicles = Some(best_vehicle_count);

                //         self.vehicle_constraint = Some(self.model.add_constr(
                //             "vehicle_constraint",
                //             c!(self.lambda.iter().grb_sum() >= best_vehicle_count as f64)
                //         ).unwrap());

                //         for (idx, var) in self.lambda.iter().enumerate() {
                //             self.model.set_obj_attr(attr::Obj, var, self.route_costs[idx]).unwrap();
                //         }

                //         println!("\nCOST OPTIMIZATION PHASE");
                //     },
                //     LPSolvePhase::CostNoCover => {
                //         // Successfully solved the second phase of the problem
                //         println!("Optimal obj {} found after {} iterations", self.model.get_attr(ObjVal).unwrap(), iter);
                //         return self.model.get_attr(attr::ObjVal).unwrap();
                //     },
                //     LPSolvePhase::CostCover | LPSolvePhase::VehicleCover | LPSolvePhase::CostQuantity | LPSolvePhase::VehicleQuantity => {
                //         panic!("Unexpected mode: {:?}", mode);
                //     },
                // }
            }

            // if new_routes.is_empty() && !mode.coverage {
            //     // Need to do the coverage dominance case
            //     mode.coverage = true;
            //     println!("\n\nAdding dominance checks for coverage\n");
            // }

            // else if new_routes.is_empty() && self.max_vehicles.is_none() {
            //     // Successfully solved the first phase of the problem
            //     print!("Optimal number of vehicles found after {} iterations\n", iter);
            //     let obj = self.model.get_attr(attr::ObjVal).unwrap();
            //     println!("Objective value: {:?}\n", obj);
            //     best_vehicle_count = obj.ceil() as usize;
            //     self.max_vehicles = Some(best_vehicle_count);

            //     self.vehicle_constraint = Some(self.model.add_constr(
            //         "vehicle_constraint",
            //         c!(self.lambda.iter().grb_sum() == best_vehicle_count as f64)
            //     ).unwrap());

            //     for (idx, var) in self.lambda.iter().enumerate() {
            //         self.model.set_obj_attr(attr::Obj, var, self.route_costs[idx]).unwrap();
            //     }

            //     mode.coverage = false;
            //     continue;
            // }

            // else if new_routes.is_empty() && self.max_vehicles.is_some() {
            //     // Successfully solved the second phase of the problem
            //     print!("Optimal obj {} found after {} iterations\n", self.model.get_attr(ObjVal).unwrap(), iter);
            //     break;
            // }

            else {         
                println!("Adding {} new routes", new_routes.len());
                let raw_ref: *mut ColGenModel = self;
                for route in new_routes.iter() {
                    let cost = route.cost as f64;
                    let covered = route.coverset.to_vec();
                    let reduced_cost = route.reduced_cost;
                    unsafe {
                        raw_ref.as_mut().unwrap().add_route_var(cost, covered);
                    }
                    if verbose { println!("COST: {}, RC: {:.4}, COVER: {:?}", cost, reduced_cost, route.coverset.to_vec()); }
                }
            }
            iter += 1;
            println!();
        }
    }

    pub fn get_lower_bounds(&mut self, verbose: bool, gap: f64) -> Vec<f64> {
        if verbose {
            println!("Calculating lower bounds for arcs");
        }

        self.model.optimize().unwrap();
        let result = self.model.get_attr(attr::Status).unwrap();

        if result == grb::Status::Infeasible {
            panic!("Model is infeasible, cannot calculate lower bounds");
        }

        let cover_duals = self.cover_constraints.iter()
            .map(|c| self.model.get_obj_attr(attr::Pi, c).unwrap())
            .collect::<Vec<_>>();
        let vehicle_dual = if self.max_vehicles.is_some() {
            Some(self.model.get_obj_attr(attr::Pi, self.vehicle_constraint.as_ref().unwrap()).unwrap())
        } else {
            None
        };

        if verbose {
            println!("Cover duals: {:?}", cover_duals);
            if let Some(dual) = vehicle_dual {
                println!("Vehicle dual: {:?}", dual);
            }
        }

        let mut pricer = BucketPricer::new(
            &self.data,
            &self.nodes,
            &self.arcs,
            cover_duals,
            vehicle_dual,
            if vehicle_dual.is_some() { LPSolvePhase::CostCover } else { LPSolvePhase::VehicleCover }, // mode
        );

        pricer.calculate_lower_rc_bounds(verbose, gap)
    }
}


// impl MasterProblemModel {
//     pub fn new(data: SPDPData) -> Self {
//         let generator = Generator::new(data.clone());

//         let arc_container = generator.generate_arcs();
//         let node_container = generator.generate_nodes();
//         let mut model = Model::new("Master Problem").unwrap();

//         let c: Vec<f64> = arc_container.arcs.iter()
//             .map(|arc| arc.cost as f64)
//             .collect();

//         let mut a = vec![vec![0.0; arc_container.num_arcs()]; data.num_requests];

//         for arc in arc_container.arcs.iter() {
//             if arc.done.len() > 0 {
//                 a[arc.done.left()][arc.id] += 1.0;
//             }
//             if arc.done.len() == 2 {
//                 a[arc.done.right()][arc.id] += 1.0;
//             }
//         }

//         let omega: Vec<Var> = (0..arc_container.num_arcs())
//             .map(|arc_id| {
//                 let arc = &arc_container.arcs[arc_id];
//                 add_var!(model, Continuous, name: &format!("a_{arc_id}"), obj: arc.cost as f64, bounds: 0.0..0.0).unwrap()
//             })
//             .collect();

//         let cover: Vec<Constr> = data.requests.iter()
//             .enumerate()
//             .map(|(r_id, request)| {
//                 let constr_expr = omega.iter()
//                     .zip((0..arc_container.num_arcs()).map(|arc_id| a[r_id][arc_id]))
//                     .map(|(omega, a)| (a * *omega))
//                     .grb_sum();

//                 let quantity = request.quantity as f64;
//                 assert!(quantity == 1.0);

//                 model.add_constr(
//                     &format!("request_{}", r_id),
//                     c!(constr_expr == quantity)
//                 ).unwrap()
//             })
//             .collect();

//         let flow: Vec<Constr> = node_container.nodes
//             .iter()
//             .enumerate()
//             .map(|(node_id, node)| {
//                 let inflow = arc_container.arcs_to.get(node).unwrap()
//                     .iter()
//                     .map(|arc| omega[arc.id])
//                     .grb_sum();

//                 let outflow = arc_container.arcs_from.get(node).unwrap()
//                     .iter()
//                     .map(|arc| omega[arc.id])
//                     .grb_sum();

//                 model.add_constr(
//                     &format!("flow_{}", node_id),
//                     c!(inflow == outflow)
//                 ).unwrap()
//             })
//             .collect();

//         // The number of vehicles leaving the depot is accurate
//         let depot_outflow = arc_container.arcs_from.get(&node_container.depot).unwrap()
//             .iter()
//             .map(|arc| omega[arc.id])
//             .grb_sum();

//         model.add_constr(
//             "depot_outflow",
//             c!(depot_outflow == 1 as f64)
//         ).unwrap();

//         // Set the upper bound of a small subset of variables to be 1.0
//         for arc in arc_container.arcs_from.get(&node_container.depot).unwrap() {
//             let result = model.set_obj_attr(UB, &omega[arc.id], 1.0);
//             assert!(result.is_ok());
//         }    

//         for arc in arc_container.arcs_to.get(&node_container.depot).unwrap() {
//             let result = model.set_obj_attr(UB, &omega[arc.id], 1.0);
//             assert!(result.is_ok());
//         }

//         MasterProblemModel {
//             data,
//             arcs: arc_container,
//             nodes: node_container,
//             model,
//             omega,
//             cover,
//             flow,
//             c,
//             a,
//         }
//     }

//     pub fn solve(&mut self) {
//         let result = self.model.optimize();
//         assert!(result.is_ok());
//     }
// }