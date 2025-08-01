extern crate grb;

use grb::attribute::ConstrDoubleAttr::RHS;
use grb::attribute::ModelDoubleAttr::ObjVal;
use grb::attribute::ModelModelSenseAttr::ModelSense;
use grb::attribute::VarDoubleAttr::Obj;
use grb::attribute::VarDoubleAttr::X;
use grb::parameter::IntParam::OutputFlag;
use grb::prelude::*;

use super::utils::*;
use super::fragment::*;
use super::pricing::*;
use super::constants::*;


pub struct MasterProblemModel {
    pub data : SPDPData,
    pub arcs : ArcContainer,
    pub nodes: NodeContainer,
    pub model: Model,
    pub omega: Vec<Var>,
    pub cover: Vec<Constr>,
    pub flow: Vec<Constr>,
    pub c: Vec<f64>,
    pub a: Vec<Vec<f64>>,
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

    pub fn solve(&mut self, verbose: bool) {
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
                mode = LPSolvePhase::VehicleCover; // This will skip straight to cost optimization
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
                        break;
                    },
                }
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