extern crate grb;

use core::panic;
use std::collections::HashMap;
use std::collections::HashSet;
use std::vec;
use grb::attribute::VarDoubleAttr::Start;
use grb::parameter::IntParam::BranchDir;
use grb::parameter::IntParam::MIPFocus;
use grb::parameter::IntParam::Threads;
use itertools::Itertools;

use grb::attribute::ConstrDoubleAttr::RHS;
use grb::attribute::ModelDoubleAttr::ObjVal;
use grb::attribute::ModelModelSenseAttr::ModelSense;
use grb::attribute::VarDoubleAttr::Obj;
use grb::attribute::VarDoubleAttr::UB;
use grb::attribute::VarDoubleAttr::X;
use grb::callback;
use grb::callback::CbResult;
use grb::callback::MIPSolCtx;
use grb::parameter::IntParam::LazyConstraints;
use grb::parameter::IntParam::OutputFlag;
use grb::prelude::*;
use grb::Error;

use super::utils::*;
use super::fragment::*;
use super::pricing::*;
use super::constants::*;


struct CallbackContext<'a> {
    data: &'a SPDPData,
    nodes: &'a NodeContainer,
    arcs: &'a ArcContainer,
    x: &'a Vec<Vec<Var>>,
    y: &'a Vec<Var>,
    verbose: bool,
    cg_lb: f64,
    best_rc_per_arc: Vec<f64>,
    already_filtered: Vec<bool>,
    prev_best_obj: f64,
}

impl<'a> CallbackContext<'a> {
    fn new(data: &'a SPDPData, nodes: &'a NodeContainer, arcs: &'a ArcContainer, x: &'a Vec<Vec<Var>>, y: &'a Vec<Var>, verbose: bool, cg_lb: f64, best_rc_per_arc: Vec<f64>, already_filtered: Vec<bool>) -> Self {
        Self {
            data,
            nodes,
            arcs,
            x,
            y,
            verbose,
            cg_lb,
            best_rc_per_arc,
            already_filtered,
            prev_best_obj: INFINITY,
        }
    }

    fn callback_mipsol(&mut self, ctx: MIPSolCtx) {
        let obj = ctx.obj_best().unwrap();

        // Add filtering cuts for arcs with reduced cost above (obj - cg_lb)
        if obj < self.prev_best_obj {
            let gap = obj - self.cg_lb;
            let mut arcs_above_half = vec![];
            for arc_id in 0..self.arcs.num_arcs() {
                if self.best_rc_per_arc[arc_id] > gap + EPS && !self.already_filtered[arc_id] {
                    for k in 0..self.x.len() {
                        ctx.add_lazy(c!(self.x[k][arc_id] == 0.0)).unwrap();
                    }
                    self.already_filtered[arc_id] = true;
                }

                else if self.best_rc_per_arc[arc_id] > gap / 2.0 + EPS {
                    arcs_above_half.push(arc_id);
                }
            }
            if OVER_HALF_GAP_CUTS_ENABLED {
                // Add cuts for these arcs
                for k in 0..self.x.len() {
                    for arc in arcs_above_half.iter() {
                        ctx.add_lazy(c!(self.x[k][*arc] <= self.y[k])).unwrap();
                    }
                }
            }

            if self.verbose {
                println!("Added filtering cuts for arcs with reduced cost above {:.2}", obj - self.cg_lb);
                println!("Filtered out {} arcs so far", self.already_filtered.iter().filter(|&&b| b).count());
                let arcs_over_half = (0..self.arcs.num_arcs()).filter_map(|arc_id| {
                    if self.best_rc_per_arc[arc_id] > (obj - self.cg_lb) / 2.0 && self.best_rc_per_arc[arc_id] < (obj - self.cg_lb) + EPS {
                        Some(arc_id)
                    } else {
                        None
                    }
                }).count();
                println!("Arcs over half the gap: {}", arcs_over_half);
            }

            self.prev_best_obj = obj;
        }

        for (_, arcs) in self.x.iter().enumerate() {
            let soln = ctx.get_solution(arcs).unwrap();
            let mut sets: Vec<HashSet<usize>> = Vec::new();
            for arc_id in 0..arcs.len() {
                let val = soln[arc_id];
                if val > EPS {
                    let nodes: Vec<usize> = vec![
                        self.arcs.get_arc(arc_id).start.id,
                        self.arcs.get_arc(arc_id).end.id,
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
            },
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
    pub y: Vec<Var>,
    pub z: Vec<Var>,
    pub cover: Vec<Constr>,
    pub flow: Vec<Vec<Constr>>,
    pub time_lim: Vec<Constr>,
    pub set_z: Vec<Constr>,
    pub sym_z: Vec<Constr>,
    pub a: Vec<Vec<f64>>,
    pub vehicle_limit: Constr,
    pub filter_cuts: Vec<Constr>,
    pub half_gap_cut: Constr,
    pub already_filtered: Vec<bool>,
    pub single_request_symmetry: Vec<Constr>,
}

impl MasterProblemModel {
    pub fn new(data: SPDPData, arc_container: ArcContainer, node_container: NodeContainer, vehicle_count: usize, vehicle_filter: Vec<bool>) -> Self {

        let mut model = Model::new("Master Problem").unwrap();

        let mut a: Vec<Vec<f64>> = vec![vec![0.0; arc_container.num_arcs()]; data.num_requests];

        for arc in arc_container.get_arcs() {
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
                    let arc = arc_container.get_arc(arc_id);
                    add_var!(model, Integer, name: &format!("a_{arc_id}_v{vehicle_id}"), obj: arc.cost as f64).unwrap()
                }).collect()
            })
            .collect();

        let y: Vec<Var> = (0..vehicle_count).map(|vehicle_id| {
            add_var!(model, Binary, name: &format!("y_{vehicle_id}"), obj: 0.0).unwrap()
        }).collect();

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
                let lhs = arc_container.get_arcs().iter().map(|arc| {
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
            .map(|(idx, _)| {
                let lhs = (0..vehicle_count).filter(|k| k > &idx).map(|k| {
                    arc_container.get_arcs().iter().filter_map(|arc| {
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
        let _basic_odd_requests: Vec<Constr> = data.requests.iter().enumerate()
            .filter(|(_r_id, r)| r.quantity % 2 == 1 || r.quantity > 2)
            .map(|(r_id, r)| {
                let lhs = (0..vehicle_count).map(|k| {
                    arc_container.get_arcs().iter().filter_map(|arc| {
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
                .filter(|(_r_id, r)| r.from_id == location.id)
                .collect();

            let quantity_sum = requests_at_location.iter().map(|(_r_id, r)| r.quantity).sum::<usize>();

            if quantity_sum > 2 && quantity_sum % 2 == 1 && requests_at_location.len() > 1 {
                let lhs = (0..vehicle_count).map(|k| {
                    arc_container.get_arcs().iter().filter_map(|arc| {
                        let q = requests_at_location.iter().map(|(r_id, _r)| a[*r_id][arc.id]).sum::<f64>();
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
                    arc_container.get_arcs().iter().filter_map(|arc| {
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
                    arc_container.get_arcs().iter().filter_map(|arc| {
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

        let filter_cuts: Vec<Constr> = vec![
            model.add_constr(&format!("Vehicle_filter"),
                c!(vehicle_filter.iter().enumerate().filter_map(|(idx, &filtered)| if filtered { Some((0..vehicle_count).map(|k| x[k][idx]).grb_sum()) } else { None }).grb_sum() == 0.0)
            ).unwrap()
        ];

        let half_gap_cut: Constr = model.add_constr(&format!("Half gap cut"), c!(y.iter().grb_sum() <= 1.0)).unwrap();

        Self {
            data,
            arcs: arc_container,
            nodes: node_container,
            model,
            x,
            y,
            z,
            cover,
            flow,
            time_lim,
            set_z,
            sym_z,
            a,
            vehicle_limit,
            filter_cuts,
            half_gap_cut,
            already_filtered: vehicle_filter,
            single_request_symmetry
        }
    }

    pub fn _filter_arcs(&mut self, filter: Vec<bool>) {
        let mut filtered = 0;

        for (arc_id, &is_filtered) in filter.iter().enumerate() {
            if is_filtered {
                filtered += 1;
                for vehicle_id in 0..self.x.len() {
                    self.model.set_obj_attr(UB, &self.x[vehicle_id][arc_id], 0.0).unwrap();
                }
            } else {
                for vehicle_id in 0..self.x.len() {
                    self.model.set_obj_attr(UB, &self.x[vehicle_id][arc_id], 1e20).unwrap();
                }
            }
        }
        println!("Filtered {}/{} arcs from the master problem", filtered, filter.len());
    }

    fn set_initial_solution(&mut self, best_sol: &Vec<Vec<usize>>, verbose: bool) {
        let mut route_cover: Vec<Vec<usize>> = vec![vec![0; self.data.num_requests]; self.x.len()];

        // Turn the vectors of arc_ids in best_sol to (arc_id, count) pairs
        let mut arc_counts: Vec<Vec<(usize, usize)>> = best_sol.iter().enumerate().map(|(vehicle_id, arc_ids)| {
            let mut counts = HashMap::new();
            for &arc_id in arc_ids {
                *counts.entry(arc_id).or_insert(0) += 1;
                for r_id in 0..self.data.num_requests {
                    route_cover[vehicle_id][r_id] += self.a[r_id][arc_id] as usize;
                }
            }
            counts.into_iter().collect::<Vec<_>>()
        }).collect();

        if verbose { println!("Initial coverage per vehicle: {:?}", route_cover); }

        let mut coverage: Vec<usize> = vec![0; self.data.num_requests];

        // Are any requests over covered?
        arc_counts = arc_counts.iter().map(|route_arcs| {
            route_arcs.iter().filter_map(|(arc_id, count)| {
                let done = self.arcs.get_arc(*arc_id).done;
                if done.len() == 0 {
                    return Some((*arc_id, *count));
                }
                coverage[done.left()] += count;
                if coverage[done.left()] > self.data.requests[done.left()].quantity {
                    // Overcovered
                    coverage[done.left()] -= count;
                    if verbose { println!("Request {} is overcovered", done.left()); }
                    return None
                }
                if done.len() == 2 {
                    coverage[done.right()] += count;
                    if coverage[done.right()] > self.data.requests[done.right()].quantity {
                        // Overcovered
                        coverage[done.left()] -= count;
                        coverage[done.right()] -= count;
                        if verbose { println!("Request {} is overcovered", done.right()); }
                        return None
                    }
                }
                Some((*arc_id, *count))
            }).collect()
        }).collect::<Vec<Vec<(usize, usize)>>>();

        if verbose {
            // Clauclate the coverage per vehicle again
            let mut new_route_cover: Vec<Vec<usize>> = vec![vec![0; self.data.num_requests]; self.x.len()];
            for (vehicle_id, route_arcs) in arc_counts.iter().enumerate() {
                for (arc_id, count) in route_arcs.iter() {
                    for r_id in 0..self.data.num_requests {
                        new_route_cover[vehicle_id][r_id] += self.a[r_id][*arc_id] as usize * *count;
                    }
                }
            }
            println!("Adjusted coverage per vehicle: {:?}", new_route_cover);
        }

        let mut add_back = vec![0; self.data.num_requests];
        for r_id in 0..self.data.num_requests {
            if coverage[r_id] < self.data.requests[r_id].quantity {
                add_back[r_id] = self.data.requests[r_id].quantity - coverage[r_id];
                if verbose { println!("Request {} is undercovered by {}", r_id, add_back[r_id]); }
            }
        }

        // Turn the invalid routes into valid ones. These are the routes that have changed
        for route in arc_counts.iter_mut() {
            let mut start_locs = Vec::new();
            let mut end_locs = Vec::new();
            for (arc_id, _) in route.iter() {
                let arc = self.arcs.get_arc(*arc_id);
                start_locs.push(arc.start.id);
                end_locs.push(arc.end.id);
            }
            if verbose {
                println!("Scanning route: starting locations {:?}, ending locations {:?}", start_locs, end_locs);
            }
            start_locs = start_locs.into_iter().filter(|&loc| {
                // Remove this location from the end locations
                if let Some(pos) = end_locs.iter().position(|&x| x == loc) {
                    end_locs.remove(pos);
                    false
                } else {
                    true
                }
            }).collect();

            // The remaining locations are the problematic ones
            // Do we need to add any requests back???
            loop {
                let end_loc = end_locs.pop();
                let start_loc = start_locs.pop();
                if verbose {
                    println!("Fixing route: going from start {:?} to end {:?}", start_loc, end_loc);
                }
                if end_loc.is_none() && start_loc.is_none() {
                    break;
                }
                else if end_loc.is_none() || start_loc.is_none() {
                    println!("Start: {:?}\nEnd: {:?}", start_locs, end_locs);
                    panic!("Mismatched start and end locations");
                }

                if add_back.iter().all(|&x| x == 0) {
                    // Find the arc ending at end_loc
                    let remove_idx = route.iter().position(|(arc_id, _)| {
                        let arc = self.arcs.get_arc(*arc_id);
                        arc.end.id == end_loc.unwrap()
                    }).unwrap();
                    let to_remove = self.arcs.get_arc(route[remove_idx].0);
                    if route[remove_idx].1 > 1 {
                        route[remove_idx].1 -= 1;
                    } else {
                        route.remove(remove_idx);
                    }
                    // Find an arc, covering the same as the remove arc that starts at the same place and ends at start loc
                    let new_arc = self.arcs.arcs_from.get(&to_remove.start).unwrap().iter().find(|arc| arc.end.id == start_loc.unwrap() && arc.done == to_remove.done);
                    if new_arc.is_none() {
                        println!("Could not find replacement arc for arc {}", to_remove.id);
                    } else {
                        route.push((new_arc.unwrap().id, 1));
                        if verbose {
                            println!("Replaced arc {} with arc {:?}", to_remove.id, new_arc.unwrap());
                        }
                    }
                } else if add_back.iter().sum::<usize>() == 1 {
                    let new_arc = self.arcs.arcs_from.get(&self.nodes.nodes.get(end_loc.unwrap()).unwrap()).unwrap().iter().find(|arc| {
                        arc.end.id == start_loc.unwrap() && arc.done.len() == 1 && add_back[arc.done.left()] > 0
                    });
                    if new_arc.is_none() {
                        println!("Could not find replacement arc from {} to {} to add back request", start_loc.unwrap(), end_loc.unwrap());
                    } else {
                        route.push((new_arc.unwrap().id, 1));
                        if verbose {
                            println!("Added back request {} using arc {}", new_arc.unwrap().done.left(), new_arc.unwrap().id);
                        }
                        add_back[new_arc.unwrap().done.left()] -= 1;
                    }
                } else {
                    let new_arc = self.arcs.arcs_from.get(&self.nodes.nodes.get(end_loc.unwrap()).unwrap()).unwrap().iter().find(|arc| {
                        arc.end.id == start_loc.unwrap() && arc.done.len() > 1 && add_back[arc.done.left()] > 0 && add_back[arc.done.right()] > 0 && 
                        (arc.done.left() != arc.done.right() && add_back[arc.done.left()] == 1) // prevent using the same request twice if only one is needed
                    });
                    if new_arc.is_none() {
                        println!("Could not find replacement arc from {} to {} to add back request: trying a one length arc", start_loc.unwrap(), end_loc.unwrap());
                        let new_arc = self.arcs.arcs_from.get(&self.nodes.nodes.get(end_loc.unwrap()).unwrap()).unwrap().iter().find(|arc| {
                            arc.end.id == start_loc.unwrap() && arc.done.len() == 1 && add_back[arc.done.left()] > 0
                        });
                        if new_arc.is_none() {
                            println!("Could not find replacement arc from {} to {} to add back request", start_loc.unwrap(), end_loc.unwrap());
                        } else {
                            route.push((new_arc.unwrap().id, 1));
                            if verbose {
                                println!("Added back request {} using arc {}", new_arc.unwrap().done.left(), new_arc.unwrap().id);
                            }
                            add_back[new_arc.unwrap().done.left()] -= 1;
                        }
                    } else {
                        route.push((new_arc.unwrap().id, 1));
                        if verbose {
                            println!("Added back requests {} and {} using arc {}", new_arc.unwrap().done.left(), new_arc.unwrap().done.right(), new_arc.unwrap().id);
                        }
                        add_back[new_arc.unwrap().done.left()] -= 1;
                        add_back[new_arc.unwrap().done.right()] -= 1;
                    }
                }
            }
        }

        while add_back.iter().sum::<usize>() > 0 {
            if verbose { println!("Warning: could not add back all missing requests, remaining {:?}", add_back); }
            // Find the shortest route and add them there

            let route_lengths: Vec<usize> = arc_counts.iter().map(|route| {
                route.iter().map(|(arc_id, count)| {
                    let arc = self.arcs.get_arc(*arc_id);
                    arc.time * *count
                }).sum()
            }).collect();

            let shortest_route = &mut arc_counts[route_lengths.iter().position(|&len| len == *route_lengths.iter().min().unwrap()).unwrap()];

            let last_arc_idx = shortest_route.iter().position(|(arc, _)| self.arcs.get_arc(*arc).end == self.nodes.depot).unwrap();
            let last_arc = self.arcs.get_arc(shortest_route[last_arc_idx].0);

            for arc in self.arcs.arcs_to.get(&self.nodes.depot).unwrap() {
                if arc.done.len() == 2 && add_back[arc.done.left()] > 0 && add_back[arc.done.right()] > 0 {
                    // Add this arc
                    shortest_route.push((arc.id, 1));
                    add_back[arc.done.left()] -= 1;
                    add_back[arc.done.right()] -= 1;
                    if verbose {
                        println!("Added back requests {} and {} using arc {:?}", arc.done.left(), arc.done.right(), arc);
                    }
                    // Add the arc that connects this from the previous one
                    let connecting_arc = self.arcs.arcs_from.get(&last_arc.start).unwrap().iter().find(|a| a.end == arc.start && a.done == last_arc.done);
                    if connecting_arc.is_none() {
                        println!("Could not find connecting arc from {} to {}", last_arc.start.id, arc.start.id);
                    } else {
                        shortest_route.push((connecting_arc.unwrap().id, 1));
                        if verbose {
                            println!("Added connecting arc {:?} from {} to {}", connecting_arc.unwrap(), connecting_arc.unwrap().start.id, connecting_arc.unwrap().end.id);
                        }
                    }
                    shortest_route.remove(last_arc_idx);
                    break;
                } else if arc.done.len() == 1 && add_back[arc.done.left()] > 0 {
                    // Add this arc
                    shortest_route.push((arc.id, 1));
                    add_back[arc.done.left()] -= 1;
                    if verbose {
                        println!("Added back request {} using arc {:?}", arc.done.left(), arc);
                    }
                    // Add the arc that connects this from the previous one
                    let connecting_arc = self.arcs.arcs_from.get(&last_arc.start).unwrap().iter().find(|a| a.end == arc.start && a.done == last_arc.done);
                    if connecting_arc.is_none() {
                        println!("Could not find connecting arc from {} to {}", last_arc.start.id, arc.start.id);
                    } else {
                        shortest_route.push((connecting_arc.unwrap().id, 1));
                        if verbose {
                            println!("Added connecting arc {:?} from {} to {}", connecting_arc.unwrap(), connecting_arc.unwrap().start.id, connecting_arc.unwrap().end.id);
                        }
                    }
                    shortest_route.remove(last_arc_idx);
                    break;
                }
            }

        }

        // Double check coverage
        let mut final_coverage: Vec<usize> = vec![0; self.data.num_requests];
        for route in arc_counts.iter() {
            for (arc_id, count) in route.iter() {
                for r_id in 0..self.data.num_requests {
                    final_coverage[r_id] += self.a[r_id][*arc_id] as usize * count;
                }
            }
        }
        for r_id in 0..self.data.num_requests {
            if verbose && final_coverage[r_id] != self.data.requests[r_id].quantity {
                println!("Final coverage for request {} is {}, expected {}", r_id, final_coverage[r_id], self.data.requests[r_id].quantity);
            }
        }

        // Display the routes
        if verbose {
            for (v_id, route) in arc_counts.iter().enumerate() {
                println!("Vehicle {}: Route {:?}", v_id + 1, route);
                for (arc_id, count) in route.iter() {
                    let arc = self.arcs.get_arc(*arc_id);
                    println!("  Arc {}: from node {} to node {} (count {})", arc.id, arc.start.id, arc.end.id, count);
                }
            }
        }

        // We need to give request 0 to the first vehicle, request 1 to the one of the first two, etc. to satisfy the symmetry constraints

        let num_vehicles = best_sol.len();
        let mut vehicle_to_be_assigned = 0;
        let mut assigned_route = num_vehicles;

        for r_id in 0..num_vehicles {
            for (idx, route) in arc_counts.iter().enumerate() {
                for (arc_id, _) in route.iter() {
                    if self.a[r_id][*arc_id] > 0.0 {
                        assigned_route = idx;
                    }
                }
            }
            if assigned_route < num_vehicles {
                // Assign this route to vehicle r_id
                for (arc_id, count) in arc_counts[assigned_route].iter() {
                    if verbose {
                        println!("Assigning arc {} (count {}) to vehicle {}", arc_id, count, vehicle_to_be_assigned);
                    }
                    self.model.set_obj_attr(Start, &self.x[vehicle_to_be_assigned][*arc_id], *count as f64).unwrap();
                }
                // Remove this route from consideration
                arc_counts.remove(assigned_route);
                assigned_route = num_vehicles;
                vehicle_to_be_assigned += 1;
            }
        }

        for route in arc_counts.iter() {
            for (arc_id, count) in route.iter() {
                if verbose {
                    println!("Assigning arc {} (count {}) to vehicle {}", arc_id, count, vehicle_to_be_assigned);
                }
                self.model.set_obj_attr(Start, &self.x[vehicle_to_be_assigned][*arc_id], *count as f64).unwrap();
            }
            vehicle_to_be_assigned += 1;
        }
    }

    pub fn solve(&mut self, verbose: bool, cg_lb: f64, best_rc_per_arc: Vec<f64>, best_sol: Vec<Vec<usize>>) -> Result<f64, Error> {
        self.model.set_attr(ModelSense, Minimize).unwrap();
        self.model.set_param(LazyConstraints, 1).unwrap();
        self.model.set_param(BranchDir, 1).unwrap();
        self.model.set_param(MIPFocus, 2).unwrap();
        self.model.set_param(Threads, 16).unwrap();

        self.set_initial_solution(&best_sol, verbose);

        let mut callback_context = CallbackContext::new(&self.data, &self.nodes, &self.arcs, &self.x, &self.y, verbose, cg_lb, best_rc_per_arc, self.already_filtered.clone());
        self.model.optimize_with_callback(&mut callback_context).unwrap();
        println!("% filtered: {:.4}", callback_context.already_filtered.iter().filter(|&&b| b).count() as f64 / callback_context.already_filtered.len() as f64);
        self.model.get_attr(attr::ObjVal)
    }

    pub fn print_solution(&self) {
        // Print the solution details
        for (k, vars) in self.x.iter().enumerate() {
            for (i, lambda) in vars.iter().enumerate() {
                let obj = self.model.get_obj_attr(X, lambda).unwrap();
                if obj > EPS {
                    println!("Vehicle {}, Arc {}: {}", k, i, obj);
                    println!("    Arc details: start: {:?}, end: {:?}, cost: {}, time: {}", self.arcs.get_arc(i).start, self.arcs.get_arc(i).end, self.arcs.get_arc(i).cost, self.arcs.get_arc(i).time);
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
    pub route_arcs: Vec<Vec<usize>>,
    pub cover_constraints: Vec<Constr>,
    pub vehicle_constraint: Option<Constr>,
    pub max_vehicles: Option<usize>,
    pub subset_row_ineqs: Vec<Constr>,
}

impl ColGenModel {
    pub fn new(data: SPDPData) -> Self {
        let generator = Generator::new(data.clone());

        let node_container = generator.generate_nodes();
        let arc_container = generator.generate_arcs(&node_container);

        let model = Model::new("COLGEN Problem").unwrap();

        let subset_row_ineqs = vec![];

        let mut result = ColGenModel {
            data: data.clone(),
            arcs: arc_container,
            nodes: node_container,
            model,
            routes_covering_request: vec![Vec::new(); data.num_requests],
            lambda: Vec::new(),
            route_costs: Vec::new(),
            route_arcs: Vec::new(),
            cover_constraints: Vec::new(),
            vehicle_constraint: None,
            max_vehicles: None,
            subset_row_ineqs,
        };

        result.first_initialisation();
        result.initialise_model_constraints();

        result
    }

    pub fn add_route_var(&mut self, cost: f64, covered: Vec<usize>, arcs: Vec<usize>) {
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

        for (id, ssi) in self.subset_row_ineqs.iter().enumerate() {
            let q = self.data.requests[id].quantity;
            if q % 2 == 1 && 2 * covered[id] > q && SRI_CONSTRAINTS_ENABLED {
                col_coeff.push((*ssi, 1.0));
            } else {
                col_coeff.push((*ssi, 0.0));
            }
        }

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
        self.route_arcs.push(arcs);
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

        for (id, _r) in self.data.requests.iter().enumerate() {
            self.subset_row_ineqs.push(self.model.add_constr(
                &format!("ssi_{}", id),
                c!(0.0 <= INFINITY)
            ).unwrap());
        }

        self.model.update().unwrap();
    }

    pub fn first_initialisation(&mut self) {
        let mut initial_routes: Vec<(usize, Vec<usize>, Vec<usize>)> = Vec::new();
        for arc in self.arcs.get_arcs() {
            if arc.start.is_pickup() && arc.end.is_depot() {
                let mut covered = vec![0; self.data.num_requests];
                if arc.done.len() > 0 {
                    covered[arc.done.left()] += 1;
                }
                if arc.done.len() == 2 {
                    covered[arc.done.right()] += 1;
                }

                let mut route_arcs = vec![arc.id];

                route_arcs.push(self.arcs.arcs_from.get(&self.nodes.depot).unwrap().iter().find(|depot_arc: &&Arc| depot_arc.end.id == arc.start.id).map(|depot_arc| depot_arc.id).unwrap());

                initial_routes.push((arc.cost + self.data.fixed_vehicle_cost, covered, route_arcs));
            }
        }

        for (cost, covered, route_arcs) in initial_routes.iter() {
            self.add_route_var(*cost as f64, covered.clone(), route_arcs.clone());
        }
    }

    fn activate_sri_constraints(&mut self, verbose: bool) {
        if verbose {
            println!("Activating subset-row inequalities");
        }
        for c in self.subset_row_ineqs.iter() {
            self.model.set_obj_attr(attr::RHS, c, 1.0).unwrap();
        }
    }

    fn deactivate_sri_constraints(&mut self, verbose: bool) {
        if verbose {
            println!("Deactivating subset-row inequalities");
        }
        for c in self.subset_row_ineqs.iter() {
            self.model.set_obj_attr(attr::RHS, c, INFINITY).unwrap();
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

    pub fn solve_with_increased_vehicles(&mut self, verbose: bool, vlb: f64, vehicle_lbs: &Vec<f64>) -> (f64, f64, Vec<f64>) {
        // Increase the vehicle limit by 1 and solve
        if self.max_vehicles.is_none() {
            panic!("Vehicle constraint is not set");
        } else {
            self.max_vehicles = Some(self.max_vehicles.unwrap() + 1);
        }

        self.model.set_obj_attr(RHS, &self.vehicle_constraint.as_ref().unwrap(), (self.max_vehicles.unwrap()) as f64).unwrap();

        let vehicle_filter = Some(vehicle_lbs.iter().map(|&lb| lb >= self.max_vehicles.unwrap() as f64 - vlb + EPS).collect::<Vec<bool>>());
        let mut cost_lbs = vec![];
        let mut route_pool: Vec<(Label, Vec<usize>)> = vec![];
        let mut mode = LPSolvePhase::CostNoCover;
        self.deactivate_sri_constraints(verbose);

        loop {
            self.model.update().unwrap();
            self.model.optimize().unwrap();

            let mut new_routes: Vec<(Label, Vec<usize>)> = Vec::new();
            let obj = self.model.get_attr(attr::ObjVal).unwrap();
            println!("OBJVAL: {:?}", obj);
            let cover_duals = self.cover_constraints.iter()
                .map(|c| self.model.get_obj_attr(attr::Pi, c).unwrap())
                .collect::<Vec<_>>();

            let vehicle_dual = if self.max_vehicles.is_some() {
                Some(self.model.get_obj_attr(attr::Pi, self.vehicle_constraint.as_ref().unwrap()).unwrap())
            } else {
                None
            };

            let ssi_duals = self.subset_row_ineqs.iter()
                .map(|c| self.model.get_obj_attr(attr::Pi, c).unwrap())
                .collect::<Vec<_>>();

            if verbose {
                println!("Cover duals: {:?}", cover_duals);
                if let Some(dual) = vehicle_dual {
                    println!("Vehicle dual: {:?}", dual);
                }
                println!("SSI duals: {:?}", ssi_duals);
            }

            // Check the route pool for any candidate routes

            for (label, visited) in route_pool.iter() {
                let reduced_cost = match mode {
                    LPSolvePhase::VehicleNoCover | LPSolvePhase::VehicleCover | LPSolvePhase::VehicleQuantity=> {
                        panic!("Should not be in vehicle phase");
                    },
                    LPSolvePhase::CostNoCover | LPSolvePhase::CostCover | LPSolvePhase::CostQuantity => {
                        label.cost as f64
                            - label.coverset.to_vec().iter().enumerate()
                                .map(|(idx, amount)| *amount as f64 * cover_duals[idx])
                                .sum::<f64>()
                            - vehicle_dual.unwrap_or(0.0)
                            - label.coverset.to_vec().iter().enumerate()
                                .map(|(idx, amount)| {
                                    if 2 * amount > self.data.requests[idx].quantity {
                                        ssi_duals[idx]
                                    } else {
                                        0.0
                                    }
                                })
                                .sum::<f64>()
                    },
                };

                if reduced_cost < -EPS {
                    if verbose {
                        println!("Route from pool with reduced cost: {:.4}", reduced_cost);
                    }
                    new_routes.push((Label {
                        id: label.id,
                        reduced_cost,
                        duration: label.duration,
                        predecessor: label.predecessor,
                        cost: label.cost,
                        coverset: label.coverset,
                        node_id: label.node_id,
                        in_arc: label.in_arc,
                    }, visited.clone()));
                }
            }

            if !new_routes.is_empty() {
                if verbose {
                    println!("Found {} candidate routes in the pool", new_routes.len());
                    new_routes.sort_by(|a, b| a.0.reduced_cost.partial_cmp(&b.0.reduced_cost).unwrap());
                    new_routes.truncate(NUM_ROUTES_PER_NODE_ADDED * self.nodes.nodes.len());
                    // panic!("Candidate routes found in the pool, not implemented yet");
                }
            } else {
                if verbose {
                    println!("No candidate routes found in the pool");
                }

                let mut pricer = BucketPricer::new(
                    &self.data,
                    &self.nodes,
                    &self.arcs,
                    cover_duals,
                    vehicle_dual,
                    ssi_duals,
                    mode,
                    &vehicle_filter,
                );

                let mut candidates: Vec<Vec<(Label, Vec<usize>)>> = pricer.solve_pricing_problem(verbose, obj);

                for node_id in 0..self.nodes.nodes.len() {
                    let node_routes = &mut candidates[node_id];

                    node_routes.sort_by(|a, b| a.0.reduced_cost.partial_cmp(&b.0.reduced_cost).unwrap());

                    for i in 0..NUM_ROUTES_PER_NODE_ADDED {
                        let label = if i < node_routes.len() {
                            &node_routes[i]
                        } else {
                            break;
                        };

                        if label.0.reduced_cost <= -EPS {
                            new_routes.push(label.clone());
                        } else {
                            route_pool.push(label.clone());
                        }
                    }

                    for i in NUM_ROUTES_PER_NODE_ADDED..NUM_ROUTES_PER_NODE_CALCULATED {
                        if i < node_routes.len() {
                            let (label, visited) = &node_routes[i];
                            route_pool.push((label.clone(), visited.clone()));
                        } else {
                            break;
                        }
                    }
                }
                if new_routes.is_empty() {
                    if mode == LPSolvePhase::CostCover {
                        cost_lbs = pricer.get_lbs();
                    }
                }
            }

            if new_routes.is_empty() {
                println!("No new routes found");

                match mode {
                    LPSolvePhase::VehicleNoCover | LPSolvePhase::VehicleQuantity | LPSolvePhase::VehicleCover => {
                        panic!("Should not be in vehicle phase");
                    },
                    LPSolvePhase::CostNoCover => {
                        mode = LPSolvePhase::CostQuantity;
                        println!("Adding quantity checks for costs");
                    },
                    LPSolvePhase::CostQuantity => {
                        mode = LPSolvePhase::CostCover;
                        self.activate_sri_constraints(verbose);
                        println!("Adding coverage checks for costs");
                    },
                    LPSolvePhase::CostCover => {
                        // Successfully solved the second phase of the problem
                        println!("Optimal obj {} found", self.model.get_attr(ObjVal).unwrap());

                        return (self.model.get_attr(attr::ObjVal).unwrap(), self.max_vehicles.unwrap() as f64, cost_lbs);
                    },
                }
            }

            else {         
                println!("Adding {} new routes", new_routes.len());
                let raw_ref: *mut ColGenModel = self;
                for (route, visited) in new_routes.iter() {
                    let cost = route.cost as f64;
                    let covered = route.coverset.to_vec();
                    let reduced_cost = route.reduced_cost;
                    unsafe {
                        raw_ref.as_mut().unwrap().add_route_var(cost, covered, visited.clone());
                    }
                    if verbose { println!("COST: {}, RC: {:.4}, COVER: {:?}", cost, reduced_cost, route.coverset.to_vec()); }
                }
            }
        }
    }

    /// Solves the column generation model
    /// # Arguments
    /// * `verbose` - If true, prints detailed information about the solving process
    /// # Returns
    /// The optimal objective value found by the model
    pub fn solve(&mut self, verbose: bool) -> (f64, f64, f64, Vec<f64>, Vec<f64>) {
        self.model.set_param(OutputFlag, 0).unwrap();
        self.model.set_attr(ModelSense, Minimize).unwrap();

        let mut iter = 1;

        let mut best_vehicle_count = 0;

        let mut mode = LPSolvePhase::VehicleNoCover;

        let mut route_pool: Vec<(Label, Vec<usize>)> = Vec::new();

        let mut vlb = 0.0;
    
        let mut vehicle_lbs = Vec::new();
        let mut cost_lbs = Vec::new();

        let mut vehicle_filter: Option<Vec<bool>> = None;

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

            let mut new_routes: Vec<(Label, Vec<usize>)> = Vec::new();
            let obj = self.model.get_attr(attr::ObjVal).unwrap();
            println!("OBJVAL: {:?}", obj);
            if self.vehicle_constraint.is_none() && obj - EPS < 1.0 {
                if verbose {
                    println!("Smallest possible vehicles count of 1 is found");
                }
                println!("Optimal number of vehicles found after {} iterations", iter);
                let obj = self.model.get_attr(attr::ObjVal).unwrap();
                best_vehicle_count = obj.ceil() as usize;
                vlb = obj;
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

                let ssi_duals = self.subset_row_ineqs.iter()
                    .map(|c| self.model.get_obj_attr(attr::Pi, c).unwrap())
                    .collect::<Vec<_>>();

                if verbose {
                    println!("Cover duals: {:?}", cover_duals);
                    if let Some(dual) = vehicle_dual {
                        println!("Vehicle dual: {:?}", dual);
                    }
                    println!("SSI duals: {:?}", ssi_duals);
                }

                // Check the route pool for any candidate routes

                for (label, visited) in route_pool.iter() {
                    let reduced_cost = match mode {
                        LPSolvePhase::VehicleNoCover | LPSolvePhase::VehicleCover | LPSolvePhase::VehicleQuantity=> {
                            1.0 - label.coverset.to_vec().iter().enumerate()
                                .map(|(idx, amount)| *amount as f64 * cover_duals[idx])
                                .sum::<f64>()
                                - label.coverset.to_vec().iter().enumerate()
                                    .map(|(idx, amount)| {
                                        if 2 * amount > self.data.requests[idx].quantity {
                                            ssi_duals[idx]
                                        } else {
                                            0.0
                                        }
                                    })
                                    .sum::<f64>()
                        },
                        LPSolvePhase::CostNoCover | LPSolvePhase::CostCover | LPSolvePhase::CostQuantity => {
                            label.cost as f64
                                - label.coverset.to_vec().iter().enumerate()
                                    .map(|(idx, amount)| *amount as f64 * cover_duals[idx])
                                    .sum::<f64>()
                                - vehicle_dual.unwrap_or(0.0)
                                - label.coverset.to_vec().iter().enumerate()
                                    .map(|(idx, amount)| {
                                        if 2 * amount > self.data.requests[idx].quantity {
                                            ssi_duals[idx]
                                        } else {
                                            0.0
                                        }
                                    })
                                    .sum::<f64>()
                        },
                    };

                    if reduced_cost < -EPS {
                        if verbose {
                            println!("Route from pool with reduced cost: {:.4}", reduced_cost);
                        }
                        new_routes.push((Label {
                            id: label.id,
                            reduced_cost,
                            duration: label.duration,
                            predecessor: label.predecessor,
                            cost: label.cost,
                            coverset: label.coverset,
                            node_id: label.node_id,
                            in_arc: label.in_arc,
                        }, visited.clone()));
                    }
                }

                if !new_routes.is_empty() {
                    if verbose {
                        println!("Found {} candidate routes in the pool", new_routes.len());
                        new_routes.sort_by(|a, b| a.0.reduced_cost.partial_cmp(&b.0.reduced_cost).unwrap());
                        new_routes.truncate(NUM_ROUTES_PER_NODE_ADDED * self.nodes.nodes.len());
                        // panic!("Candidate routes found in the pool, not implemented yet");
                    }
                } else {
                    if verbose {
                        println!("No candidate routes found in the pool");
                    }

                    let mut pricer = BucketPricer::new(
                        &self.data,
                        &self.nodes,
                        &self.arcs,
                        cover_duals,
                        vehicle_dual,
                        ssi_duals,
                        mode,
                        &vehicle_filter,
                    );

                    let mut candidates = pricer.solve_pricing_problem(verbose, obj);

                    for node_id in 0..self.nodes.nodes.len() {
                        let node_routes = &mut candidates[node_id];

                        node_routes.sort_by(|a, b| a.0.reduced_cost.partial_cmp(&b.0.reduced_cost).unwrap());

                        for i in 0..NUM_ROUTES_PER_NODE_ADDED {
                            let label = if i < node_routes.len() {
                                &node_routes[i]
                            } else {
                                break;
                            };

                            if label.0.reduced_cost <= -EPS {
                                new_routes.push(label.clone());
                            } else {
                                route_pool.push(label.clone());
                            }
                        }

                        for i in NUM_ROUTES_PER_NODE_ADDED..NUM_ROUTES_PER_NODE_CALCULATED {
                            if i < node_routes.len() {
                                let label = &node_routes[i];
                                route_pool.push(label.clone());
                            } else {
                                break;
                            }
                        }
                    }
                    if new_routes.is_empty() {
                        if mode == LPSolvePhase::VehicleCover {
                            vehicle_lbs = pricer.get_lbs();
                        }

                        if mode == LPSolvePhase::CostCover {
                            cost_lbs = pricer.get_lbs();
                        }
                    }
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
                        self.activate_sri_constraints(verbose);
                        println!("Adding coverage checks for vehicles");
                    },
                    LPSolvePhase::VehicleCover => {
                        mode = LPSolvePhase::CostNoCover;
                        self.deactivate_sri_constraints(verbose);
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

                        println!("Leveraging vehicle objective");
                        
                        vlb = obj;
                        let v_gap = obj.ceil() - obj;
                        vehicle_filter = Some(vehicle_lbs.iter().map(|lb| *lb > v_gap + EPS).collect::<Vec<_>>());
                        let num_filtered = vehicle_filter.as_ref().unwrap().iter().fold(0, |acc, &x| if x { acc + 1 } else { acc });

                        println!("Leveraged out {}/{} arcs", num_filtered, self.arcs.num_arcs());

                        println!("\nCOST OPTIMIZATION PHASE");
                    },
                    LPSolvePhase::CostNoCover => {
                        mode = LPSolvePhase::CostQuantity;
                        println!("Adding quantity checks for costs");
                    },
                    LPSolvePhase::CostQuantity => {
                        mode = LPSolvePhase::CostCover;
                        self.activate_sri_constraints(verbose);
                        println!("Adding coverage checks for costs");
                    },
                    LPSolvePhase::CostCover => {
                        // Successfully solved the second phase of the problem
                        println!("Optimal obj {} found after {} iterations", self.model.get_attr(ObjVal).unwrap(), iter);

                        return (self.model.get_attr(attr::ObjVal).unwrap(), vlb, best_vehicle_count as f64, vehicle_lbs, cost_lbs);
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
                for (route, visited) in new_routes.iter() {
                    let cost = route.cost as f64;
                    let covered = route.coverset.to_vec();
                    let reduced_cost = route.reduced_cost;
                    unsafe {
                        raw_ref.as_mut().unwrap().add_route_var(cost, covered, visited.clone());
                    }
                    if verbose { println!("COST: {}, RC: {:.4}, COVER: {:?}", cost, reduced_cost, route.coverset.to_vec()); }
                }
            }
            iter += 1;
            println!();
        }
    }

    // pub fn filter_by_lower_bounds(&mut self, verbose: bool, gap: f64) -> Vec<f64> {
    //     panic!("Lower bound filtering not implemented for column generation");
    //     if verbose {
    //         println!("Calculating lower bounds for arcs");
    //     }

    //     self.model.optimize().unwrap();
    //     let result = self.model.get_attr(attr::Status).unwrap();

    //     if result == grb::Status::Infeasible {
    //         panic!("Model is infeasible, cannot calculate lower bounds");
    //     }

    //     let cover_duals = self.cover_constraints.iter()
    //         .map(|c| self.model.get_obj_attr(attr::Pi, c).unwrap())
    //         .collect::<Vec<_>>();
    //     let vehicle_dual = if self.max_vehicles.is_some() {
    //         Some(self.model.get_obj_attr(attr::Pi, self.vehicle_constraint.as_ref().unwrap()).unwrap())
    //     } else {
    //         None
    //     };

    //     if verbose {
    //         println!("Cover duals: {:?}", cover_duals);
    //         if let Some(dual) = vehicle_dual {
    //             println!("Vehicle dual: {:?}", dual);
    //         }
    //     }

    //     let mut pricer = BucketPricer::new(
    //         &self.data,
    //         &self.nodes,
    //         &self.arcs,
    //         cover_duals,
    //         vehicle_dual,
    //         if vehicle_dual.is_some() { LPSolvePhase::CostCover } else { LPSolvePhase::VehicleCover }, // mode
    //     );

    //     pricer.calculate_lower_rc_bounds(verbose, gap)
    // }
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

pub struct RouteIPModel {
    pub model: Model,
    pub routes: Vec<Var>,
    pub cover: Vec<Constr>,
    pub vehicle_limit: Constr,
    pub costs: Vec<f64>,
    pub covered: Vec<Vec<usize>>,
}

impl RouteIPModel {
    pub fn new(data: &SPDPData, route_costs: &Vec<f64>, routes_covering_request: &Vec<Vec<usize>>, max_vehicles: usize) -> Self {
        let mut model = Model::new("Route IP Problem").unwrap();

        let routes: Vec<Var> = route_costs.iter()
            .enumerate()
            .map(|(route_id, &cost)| {
                add_var!(model, Integer, name: &format!("route_{}", route_id), obj: cost).unwrap()
            })
            .collect();

        let cover: Vec<Constr> = data.requests.iter()
            .enumerate()
            .map(|(r_id, request)| {
                let constr_expr = routes.iter()
                    .zip(routes_covering_request[r_id].iter())
                    .map(|(route_var, &amount)| (amount as f64 * *route_var))
                    .grb_sum();

                model.add_constr(
                    &format!("request_{}", r_id),
                    c!(constr_expr >= request.quantity as f64)
                ).unwrap()
            })
            .collect();

        let vehicle_limit = model.add_constr(
            "vehicle_limit",
            c!(routes.iter().grb_sum() <= max_vehicles as f64)
        ).unwrap();

        Self {
            model,
            routes,
            cover,
            vehicle_limit,
            costs: route_costs.clone(),
            covered: routes_covering_request.clone(),
        }
    }

    pub fn solve(&mut self, verbose: bool) -> (Result<f64, Error>, Vec<f64>) {
        if verbose {
            println!("Solving Route IP Model with {} routes", self.routes.len());
        }
        self.model.set_attr(ModelSense, Minimize).unwrap();
        self.model.optimize().unwrap();

        for (idx, route) in self.routes.iter().enumerate() {
            let value = self.model.get_obj_attr(X, route).unwrap();
            if value < EPS {
                continue;
            }
            println!("Route variable: {:?}, cost: {:?}, covered: {:?}", route, self.costs[idx], (0..self.covered.len()).map(|r_id| self.covered[r_id][idx]).collect::<Vec<_>>());
        }

        (self.model.get_attr(attr::ObjVal), self.routes.iter().map(|route| self.model.get_obj_attr(X, route).unwrap()).collect())
    }
}