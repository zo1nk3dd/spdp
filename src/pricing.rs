use core::{panic};
use std::collections::{HashMap};
use std::fmt::Display;
use std::time::{Duration, Instant};
use std::{f64, fmt, vec};
use std::thread;

use grb::INFINITY;

use crate::fragment::{Arc, ArcContainer, NodeContainer};
use crate::utils::{SPDPData, Label};
use crate::coverset::{get_manager, init_manager, CoverSet};
use crate::constants::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DominanceMode {
    RC,
    DurRC,
    DurRCCover,
    Dur,
    DurRCQuantity,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LPSolvePhase {
    VehicleNoCover,
    VehicleQuantity,
    VehicleCover,
    CostNoCover,
    CostQuantity,
    CostCover,
}

#[derive(Debug, Clone)]
struct VisitedData {
    pub visited: Vec<HashMap<SIZE, Vec<usize>>>,
    pub depot_labels: Vec<usize>,
    pub labels: Vec<Label>,
    pub cut_in: usize,
    pub cut_out: usize,
    pub better: usize,
}

impl VisitedData {
    fn new(num_nodes: usize) -> Self {
        VisitedData {
            visited: vec![HashMap::new(); num_nodes],
            depot_labels: Vec::new(),
            labels: Vec::new(),
            cut_in: 0,
            cut_out: 0,
            better: 0,
        }
    }

    fn _add_with_dominance_mode(&mut self, label: Label, mode: DominanceMode) -> bool {
        let mut new_label = label;
        new_label.id = self.labels.len(); // Set the label id to the current length of labels
        let bucket = &mut self.visited[label.node_id];
        // assert!(bucket.iter().all(|l| !l.is_empty()), "Bucket should not contain empty labels");

        if bucket.is_empty() {
            assert!(self.labels.len() == new_label.id);
            bucket.insert(new_label.coverset.covered, vec![new_label.id]);
            self.labels.push(new_label);
            return true; // If no labels visited, this is an improvement
        }

        let result = bucket.get_mut(&label.coverset.covered);

        if let Some(label_ids) = result {
            for id in label_ids.iter() {
                if self.labels[*id].dominates(&new_label, mode) {
                    self.cut_in += 1;
                    return false; // If the new label is dominated by any of the previous labels, it's not an improvement
                }
            }
            let len_before = label_ids.len();
            label_ids.retain(|id| {
                // Remove dominated labels
                !new_label.dominates(&self.labels[*id], mode)
            });
            let len_after = label_ids.len();
            self.cut_out += len_before - len_after;
            self.better += 1;
            label_ids.push(new_label.id);
        }
        else {
            // This cover set has not been found before;
            bucket.insert(label.coverset.covered, vec![new_label.id]);
        }

        self.labels.push(new_label); // Add the label to the labels vector
        true
    }
    
    /// Checks if a label is already visited in the current phase
    /// It needs to use the correct key based on the phase
    /// This depends upon the logic in the `add_if_improvement` function
    fn contains_label(&self, label: &Label, phase: LPSolvePhase) -> bool {
        let bucket = &self.visited[label.node_id];
        let key = match phase {
            LPSolvePhase::VehicleNoCover | LPSolvePhase::CostNoCover => 0,
            LPSolvePhase::VehicleQuantity | LPSolvePhase::CostQuantity => label.coverset.len as SIZE,
            LPSolvePhase::VehicleCover | LPSolvePhase::CostCover => label.coverset.covered,
        };
        if let Some(label_ids) = bucket.get(&key) {
            return label_ids.iter().any(|id| *id == label.id);
        }
        false
    }

    /// Attempts to add a label to the visited data structure, and returns the result
    fn add_if_improvement(&mut self, label: Label, phase: LPSolvePhase) -> bool {       
        if label.node_id == 0 {
            // If the label is for the depot, we add it to the depot labels
            self.depot_labels.push(label.id);
            self.labels.push(label);
            return true; // Depot labels are always considered improvements
        } 
        let bucket = &mut self.visited[label.node_id];
        assert!(self.labels.len() == label.id);
        // assert!(bucket.iter().all(|l| !l.is_empty()), "Bucket should not contain empty labels");

        // Vehicle No Cover, we want to store only the best reduced cost label at each node
        match phase {
            LPSolvePhase::VehicleNoCover | LPSolvePhase::CostNoCover => {
            // assert!(bucket.len() == 1, "Bucket should contain only one label for VehicleNoCover or CostNoCover phase");
                let label_ids = bucket.get_mut(&0);

                if let Some(label_ids) = label_ids {
                    // println!("Adding label to bucket: {:?}", label_ids);
                    // This cover set has been previously found
                    for label_id in label_ids.iter() {
                        if self.labels[*label_id].dominates(&label, DominanceMode::DurRC) {
                            // println!("Label with rc {:?} and dur {} dominates rc {} and dur {:?}", 
                            //     self.labels[*label_id].reduced_cost, self.labels[*label_id].duration,
                            //     label.reduced_cost, label.duration);
                            self.cut_in += 1;
                            return false; // If the new label is dominated by any of the previous labels, it's not an improvement
                        }
                    }
                    label_ids.retain(|id| {
                        // Remove dominated labels
                        !label.dominates(&self.labels[*id], DominanceMode::DurRC)
                    });
                    label_ids.push(label.id);
                }

                else {
                    // This cover set has not been found before
                    self.better += 1; // Count this as a better label
                    bucket.insert(0, vec![label.id]);
                }
            },
            LPSolvePhase::VehicleQuantity | LPSolvePhase::CostQuantity => {
                let mode = DominanceMode::DurRC;
                let result = bucket.get_mut(&(label.coverset.len as SIZE));

                if let Some(label_ids) = result {
                    for id in label_ids.iter() {
                        if self.labels[*id].dominates(&label, mode) {
                            self.cut_in += 1;
                            return false; // If the new label is dominated by any of the previous labels, it's not an improvement
                        }
                    }
                    let len_before = label_ids.len();
                    label_ids.retain(|id| {
                        // Remove dominated labels
                        !label.dominates(&self.labels[*id], mode)
                    });
                    let len_after = label_ids.len();
                    self.cut_out += len_before - len_after;
                    self.better += 1;
                    label_ids.push(label.id);
                }
                // if label.coverset.len > 1 {
                //     let result = bucket.get(&(label.coverset.len as SIZE - 1));
                //     if let Some(label_ids) = result {
                //         for id in label_ids.iter() {
                //             if self.labels[*id].dominates(&label, mode) {
                //                 self.cut_in += 1;
                //                 return false; // If the new label is dominated by any of the previous labels, it's not an improvement
                //             }
                //         }
                //     }
                // }
                else {
                    // This cover set has not been found before;
                    bucket.insert(label.coverset.len as SIZE, vec![label.id]);
                }
            },
            LPSolvePhase::VehicleCover => {
                let label_ids = bucket.get_mut(&label.coverset.covered);

                if let Some(label_ids) = label_ids {
                    // This cover set has been previously found
                    let prev_label = &self.labels[label_ids[0]];
                    let mode = DominanceMode::Dur;
                    if label.dominates(&prev_label, mode) {
                        label_ids[0] = label.id; // Update the label id to the new label
                        self.cut_out += 1;
                        self.better += 1; // Count this as a better label
                    } else {
                        self.cut_in += 1;
                        return false; // If the new label does not dominate the best label, it's not an improvement
                    }
                }

                else {
                    // This cover set has not been found before
                    bucket.insert(label.coverset.covered, vec![label.id]);
                }
            },
            LPSolvePhase::CostCover => {
                // If we reach here, the label is an improvement
                let result = bucket.get(&label.coverset.covered);

                if let Some(label_ids) = result {
                    for id in label_ids.iter() {
                        if self.labels[*id].dominates(&label, DominanceMode::DurRC) {
                            self.cut_in += 1;
                            return false; // If the new label is dominated by any of the previous labels, it's not an improvement
                        }
                    }

                    for (request_id, request_amount) in label.coverset.to_vec().iter().enumerate() {
                        if *request_amount == 0 {
                            continue;
                        }
                        let mut coverset: CoverSet = label.coverset;
                        coverset.uncover(request_id).unwrap();
                        let result = bucket.get(&coverset.covered);
                        if let Some(label_ids) = result {
                            for id in label_ids.iter() {
                                if self.labels[*id].dominates(&label, DominanceMode::DurRC) {
                                    self.cut_in += 1;
                                    return false; // If the new label is dominated by any of the previous labels, it's not an improvement
                                }
                            }
                        }
                    }
                }

                let result = bucket.get_mut(&label.coverset.covered);

                if let Some(label_ids) = result {
                    let len_before = label_ids.len();
                    label_ids.retain(|id| {
                        // Remove dominated labels
                        !label.dominates(&self.labels[*id], DominanceMode::DurRC)
                    });
                    let len_after = label_ids.len();
                    self.cut_out += len_before - len_after;
                    self.better += 1;
                    label_ids.push(label.id);
                } else {
                    // This cover set has not been found before;
                    bucket.insert(label.coverset.covered, vec![label.id]);
                }
            },
        }
        self.labels.push(label); // Add the label to the labels vector
        true
    }

    fn get_label_ids_by_node(&self, node_id: usize) -> Vec<usize> {
        self.visited[node_id].values().flatten().map(|id| *id).collect()
    }

    // fn get_label_ids_by_node_covered(&self, node_id: usize, covered: &CoverSet) -> Vec<usize> {
    //     let result = self.visited[node_id].get(&covered.covered);
    //     if result.is_some() {
    //         result.unwrap().clone()
    //     } else {    
    //         vec![]
    //     }
    // }

    // fn get_finished_labels(&self) -> Vec<Label> {
    //     self.depot_labels.iter().map(|id| self.labels[*id]).collect()
    // }
    
    fn print_visited_info(&self) {
        println!("Visited info:");
        println!("  Cut in: {}", self.cut_in);
        println!("  Cut out: {}", self.cut_out);
        println!("  Better labels: {}", self.better);
        println!("  Total labels: {}", self.labels.len());
    }

    fn num_labels(&self) -> usize {
        self.labels.len()
    }
}

pub trait Pricer {
    fn solve_pricing_problem(&mut self, verbose: bool, objective: f64) -> Vec<Vec<Label>>;
}



struct BucketGraph {
    buckets: Vec<Vec<Vec<usize>>>,
    bucket_length: usize,
    finished_bucket: Vec<usize>,    
    current_bucket: usize,
    current_node: usize,
}

impl BucketGraph {
    pub fn new(time_limit: usize, smallest_arc_duration: usize, num_nodes: usize) -> Self {
        let num_buckets = time_limit / smallest_arc_duration + 1;
        let buckets: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new(); num_nodes]; num_buckets];
        let finished_bucket = vec![];
        BucketGraph { buckets, bucket_length: smallest_arc_duration, finished_bucket, current_bucket: 0, current_node: 0 }
    }

    fn get_bucket_by_duration(&mut self, node_id: usize, time: usize) -> &mut Vec<usize> {
        let bucket_index = time / self.bucket_length;
        assert!(bucket_index < self.buckets.len(), "Bucket index out of bounds: {}, time: {}", bucket_index, time);
        assert!(node_id < self.buckets[0].len(), "Node ID out of bounds: {}", node_id);
        &mut self.buckets[bucket_index][node_id]
    }

    fn push(&mut self, label: Label, max: usize) {  
        if label.id >= max {
            println!("Label: {}", label);
            panic!("Label ID {} exceeds maximum allowed {}", label.id, max);
        }
        if label.node_id == 0 {
            if label.reduced_cost <= -EPS {
                self.finished_bucket.push(label.id);
            }
        } else {
            self.get_bucket_by_duration(label.node_id, label.duration).push(label.id);
        }
    }

    fn get_next_bucket(&mut self) -> Option<&mut Vec<usize>> {
        if self.current_node < self.buckets[self.current_bucket].len() - 1 {
            self.current_node += 1;
        } else {
            self.current_node = 0;
            self.current_bucket += 1;
        }
        if self.current_bucket < self.buckets.len() {
            Some(&mut self.buckets[self.current_bucket][self.current_node])
        } else {
            // Reset the iterator
            self.reset();
            None
        }
    }

    fn reset(&mut self) {
        self.current_bucket = 0;
        self.current_node = 0;
    }
}

impl Display for BucketGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for bucket in &self.buckets {
            for node in bucket {
                write!(f, "{:?} ", node.len())?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}


pub struct BucketPricer<'a> {
    graph: BucketGraph,
    cover_rc: Vec<f64>,
    vehicle_rc: Option<f64>,
    ssi_duals: Vec<f64>,
    phase: LPSolvePhase,
    data: &'a SPDPData,
    arcs: &'a ArcContainer,
    nodes: &'a NodeContainer,
    visited: VisitedData,
    pub lbs: Vec<f64>,
    filter: &'a Option<Vec<bool>>,
}

impl<'a> BucketPricer<'a> {
    pub fn new(data: &'a SPDPData, nodes: &'a NodeContainer, arcs: &'a ArcContainer, cover_rc: Vec<f64>, vehicle_rc: Option<f64>, ssi_duals: Vec<f64>, phase: LPSolvePhase, filter: &'a Option<Vec<bool>>) -> Self {
        let graph = BucketGraph::new(data.t_limit, arcs.min_fragment_length, nodes.nodes.len());
        let visited = VisitedData::new(nodes.nodes.len());
        let lbs = vec![0.0; arcs.num_arcs()];
        BucketPricer {
            graph,
            cover_rc,
            vehicle_rc,
            ssi_duals,
            phase,
            data,
            arcs,
            nodes,
            visited,
            lbs,
            filter,
        }
    }

    pub fn get_lbs(&self) -> Vec<f64> {
        self.lbs.clone()
    }

    fn initialise_forward_labels(&mut self) {
        for arc in self.arcs.arcs_from.get(&self.nodes.depot).unwrap() {
            let serviced = match arc.done.len() {
                0 => vec![],
                1 => vec![arc.done.left()],
                _ => vec![arc.done.left(), arc.done.right()],
            };
            assert!(serviced.len() == 0);
            let coverset = CoverSet::new(get_manager());
            assert!(arc.time == 0);
            let new_reduced_cost = if self.vehicle_rc.is_none() {
                    1.0
                } 
                else {
                    arc.cost as f64 - self.vehicle_rc.unwrap()
                };
            let new_label = Label::new(self.visited.num_labels(), new_reduced_cost, arc.time, None, arc.cost, coverset, arc.end.id, arc.id);
            if self.visited.add_if_improvement(new_label, self.phase) {
                self.graph.push(new_label, self.visited.labels.len());
            }
        }
    }

    fn initialise_backward_labels(&mut self) {
        for arc in self.arcs.arcs_to.get(&self.nodes.depot).unwrap() {
            let serviced = match arc.done.len() {
                0 => vec![],
                1 => vec![arc.done.left()],
                _ => vec![arc.done.left(), arc.done.right()],
            };
            let mut coverset = CoverSet::new(get_manager());
            let mut new_reduced_cost = if self.vehicle_rc.is_none() {
                    0.0
                } else {
                    arc.cost as f64
                };
            for r_id in serviced.iter() {
                match coverset.cover(*r_id) {
                    Ok(_) => {},
                    Err(_) => break, // Skip if the request is already serviced
                }
                new_reduced_cost -= self.cover_rc[*r_id];
            }
            // println!("Initialising backward label with rc {:?}", new_reduced_cost);

            let new_label = Label::new(self.visited.num_labels(), new_reduced_cost, arc.time, None, arc.cost, coverset, arc.start.id, arc.id);
            if self.visited.add_if_improvement(new_label, self.phase) {
                self.graph.push(new_label, self.visited.labels.len());
            }
        }
    }

    /// Extends a label by a given arc, creating a new label if the extension is valid.
    fn calculate_next_label(&self, label: &Label, arc: &Arc) -> Option<Label> {
        let mut next_node_id = arc.end.id;
        if next_node_id == label.node_id {
            next_node_id = arc.start.id; // Make sure we go in the correct direction
        }

        let new_duration = label.duration + arc.time;

        if new_duration > self.data.t_limit {
            return None; // Skip if the new duration exceeds the time limit
        }

        let new_cost = label.cost + arc.cost;

        let serviced = match arc.done.len() {
            0 => vec![],
            1 => vec![arc.done.left()],
            _ => vec![arc.done.left(), arc.done.right()],
        };

        let mut new_covered = label.coverset;
        let mut new_reduced_cost = label.reduced_cost;

        for request_id in serviced.iter() {
            let reduction = self.cover_rc[*request_id];
            // if reduction < EPS {
            //     return None; // Skip if the request does not contribute to the reduced cost
            // }
            match new_covered.cover(*request_id) {
                Ok(_) => {},
                Err(_) => {return None;}, // Skip if the request is already serviced enough
            }
            new_reduced_cost -= reduction;
        }

        new_reduced_cost += if self.vehicle_rc.is_some() {arc.cost as f64} else {0.0};

        Some(Label::new(
            self.visited.num_labels(),
            new_reduced_cost,
            new_duration,
            Some(label.id),
            new_cost,
            new_covered,
            next_node_id,
            arc.id,
        ))
    }

    /// Extends a label by a given arc, creating a new label if the extension is valid.
    /// Set time limit to be the route time limit in the full forward modes, and half the timelimit for half passes.
    fn extend_label(&mut self, label: &Label, arc: &Arc, backward_pass_limit: usize) {   
        if let Some(filter) = &self.filter {
            if filter[arc.id] {
                return; // Skip if the arc is filtered
            }
        }
        // Used for the half forward pass, dont extend labels past the time limit
        if let Some(next_label) = self.calculate_next_label(label, arc) {
            // Used for half backward pass, dont consider labels that exceed the backward pass limit
            if next_label.duration > backward_pass_limit {
                return;    
            }
            if self.visited.add_if_improvement(next_label, self.phase) {
                self.graph.push(next_label, self.visited.labels.len());
            }
        } 
        // println!("Skipping label extension: {}", label.id);
    }
    
    fn forward_pass(&mut self, time_limit: usize) {
        loop {
            let result = self.graph.get_next_bucket();
            if result.is_none() {
                break; // No more buckets to process
            }
            let bucket = std::mem::take(result.unwrap());
            for &label_id in bucket.iter() {
                let label = self.visited.labels[label_id];
                // println!("Processing label: {}", label.reduced_cost);
                if !self.visited.contains_label(&label, self.phase) {
                    // If the label is not in the visited set, it has been pruned
                    // println!("Label not visited, skipping");
                    continue;
                }

                if label.duration > time_limit {
                    // Reached the point where we should stop
                    self.graph.reset();
                    return; // Stop processing further labels
                }
                let node = &self.nodes.nodes[label.node_id];
                let forward_arcs = self.arcs.arcs_from.get(node).unwrap();
                for arc in forward_arcs {
                    self.extend_label(&label, arc, self.data.t_limit);
                }
            }
        }
    }

    fn backward_pass(&mut self, time_limit: usize) {
        loop {
            let result = self.graph.get_next_bucket();
            if result.is_none() {
                break; // No more buckets to process
            }
            let bucket = std::mem::take(result.unwrap());
            for &label_id in bucket.iter() {
                let label = self.visited.labels[label_id];
                if !self.visited.contains_label(&label, self.phase) {
                    // If the label is not in the visited set, it has been pruned
                    continue;
                }
                let node = &self.nodes.nodes[label.node_id];
                let backward_arcs = self.arcs.arcs_to.get(node).unwrap();
                for arc in backward_arcs {
                    self.extend_label(&label, arc, time_limit);
                }
            }
        }
    }

    fn finished_label(&self, forward_label: &Label, backward_label: Option<&Label>, arc: Option<&Arc>) -> Option<Label> {
        // if arc.is_none() && backward_label.is_none() {
        //     let mut new_label = forward_label.clone();
        //     for (idx, &amount) in new_label.coverset.to_vec().iter().enumerate() {
        //         if 2 * amount > self.data.requests[idx].quantity {
        //             new_label.reduced_cost -= self.ssi_duals[idx];
        //         }
        //     }
        // }

        let mut new_label = if let Some(arc) = arc {
            self.calculate_next_label(forward_label, arc)
        } else {
            Some(forward_label.clone())
        }?;

        if backward_label.is_none() {
            // Look at the ssi duals
            for (idx, &amount) in new_label.coverset.to_vec().iter().enumerate() {
                if 2 * amount > self.data.requests[idx].quantity {
                    new_label.reduced_cost -= self.ssi_duals[idx];
                }
            }
            return Some(new_label);
        }

        let backward_label = backward_label.unwrap();

        // Combine the coversets
        let combined_coverset = new_label.coverset.combine(&backward_label.coverset);
        if combined_coverset.is_err() {
            return None; // Skip if the coversets are not compatible
        }

        new_label.coverset = combined_coverset.unwrap();

        // Adjust the reduced cost
        new_label.reduced_cost += backward_label.reduced_cost;
        for (idx, &amount) in new_label.coverset.to_vec().iter().enumerate() {
            if 2 * amount > self.data.requests[idx].quantity {
                new_label.reduced_cost -= self.ssi_duals[idx];
            }
        }

        new_label.cost += backward_label.cost;
        new_label.duration += backward_label.duration;
        
        if new_label.duration > self.data.t_limit {
            return None; // Skip if the new duration exceeds the time limit
        }

        Some(new_label)
    }

    fn forward_backward_pass(&mut self, k: usize, _obj: f64, verbose: bool) -> Vec<Vec<Label>> {
        // Create the initial labels
        self.initialise_forward_labels();
        self.forward_pass((self.data.t_limit as f64 * FORWARD_BACKWARD_PASS_MARK) as usize);

        if verbose {
            println!("Forward pass finished");
            self.visited.print_visited_info();
        }

        let forward_pass_info = std::mem::replace(&mut self.visited, VisitedData::new(self.nodes.nodes.len()));
        self.graph = BucketGraph::new(self.data.t_limit, self.arcs.min_fragment_length, self.nodes.nodes.len());

        // Backward pass
        self.initialise_backward_labels();
        self.backward_pass((self.data.t_limit as f64 * (1.0-FORWARD_BACKWARD_PASS_MARK)) as usize);

        if verbose {
            println!("Backward pass finished");
            self.visited.print_visited_info();
        }

        // Combine forward and backward labels
        let mut candidate_labels: Vec<Vec<Label>> = vec![Vec::new(); self.nodes.nodes.len()];
        let depot_id = self.nodes.depot.id;

        for label_id in forward_pass_info.get_label_ids_by_node(depot_id) {
            if let Some(forward_label) = self.finished_label(&forward_pass_info.labels[label_id], None, None) {
                if forward_label.reduced_cost < -EPS {
                    candidate_labels[depot_id].push(forward_label);
                }
            }
        }
  
        if candidate_labels[depot_id].len() > k {
            candidate_labels[depot_id].sort_by(|a, b| a.reduced_cost.partial_cmp(&b.reduced_cost).unwrap());
            candidate_labels[depot_id].truncate(k); // Keep only the best k labels
        }

        for node_id in 1..self.nodes.nodes.len() {
            let mut worst_rc = INFINITY;
            let mut worst_idx = 0;

            let mut forward_bucket = forward_pass_info.get_label_ids_by_node(node_id).iter().map(|id| &forward_pass_info.labels[*id]).collect::<Vec<_>>();
            let mut backward_bucket = self.visited.get_label_ids_by_node(node_id).iter().map(|id| &self.visited.labels[*id]).collect::<Vec<_>>();

            forward_bucket.sort_by(|a, b| a.reduced_cost.partial_cmp(&b.reduced_cost).unwrap());
            backward_bucket.sort_by(|a, b| a.reduced_cost.partial_cmp(&b.reduced_cost).unwrap());

            for forward_label in forward_bucket.iter() {
                for backward_label in backward_bucket.iter() {
                    if let Some(candidate_label) = self.finished_label(forward_label, Some(backward_label), None) {
                        if candidate_label.reduced_cost >= worst_rc {
                            break; // Skip if the combined reduced cost is not better
                        }
                        if candidate_label.duration <= self.data.t_limit {
                            if !candidate_labels[node_id].iter().any(|l: &Label| l.dominates(&candidate_label, DominanceMode::DurRCCover)) {
                                if candidate_labels[node_id].len() < k {
                                    candidate_labels[node_id].push(candidate_label);
                                    if candidate_label.reduced_cost < worst_rc {
                                        worst_rc = candidate_label.reduced_cost;
                                        worst_idx = candidate_labels[node_id].len() - 1;
                                    }
                                } else {
                                    candidate_labels[node_id][worst_idx] = candidate_label; // Replace the worst label with the new one
                                    let mut idx = 0;
                                    let mut worst_cost = candidate_labels[node_id][0].reduced_cost;
                                    // Find the index of the worst label
                                    for (i, label) in candidate_labels[node_id].iter().enumerate() {
                                        if label.reduced_cost > worst_cost {
                                            idx = i;
                                            worst_cost = label.reduced_cost;
                                        }
                                    }
                                    worst_idx = idx; // Update the worst index
                                    worst_rc = worst_cost; // Update the worst reduced cost
                                }
                            }
                        }
                    }
                }
            }
        }

        if candidate_labels.iter().flatten().all(|label| label.reduced_cost > -EPS) && (self.phase == LPSolvePhase::VehicleCover || self.phase == LPSolvePhase::CostCover) {
            if verbose { println!("No candidate labels found in full coverage forward-backward pass"); }
            // Store the lower bounds at this point to grab again later
            self.lbs = self.calculate_lower_rc_bounds_route_method(verbose, &forward_pass_info, &self.visited);
        }
        candidate_labels
    }

    fn calculate_lower_rc_bounds_route_method(&self, _verbose: bool, forward_pass_info: &VisitedData, backward_pass_info: &VisitedData) -> Vec<f64> {

        println!("\n\nCalculating lower RC bounds using route method\n\n");

        let maximum_reduced_cost: f64 = 1e6;
        let mut lbs = vec![maximum_reduced_cost; self.arcs.num_arcs()];
        let mut routes = 0;
        let forward_labels_by_node: Vec<Vec<Label>> = self.nodes.nodes.iter()
            .map(|node| {
                let mut labels = forward_pass_info.get_label_ids_by_node(node.id).iter().map(|id| forward_pass_info.labels[*id].clone()).collect::<Vec<_>>();
                labels.sort_by(|a, b| a.reduced_cost.partial_cmp(&b.reduced_cost).unwrap());
                labels
            }).collect();

        let backward_labels_by_node: Vec<Vec<Label>> = self.nodes.nodes.iter()
            .map(|node| {
                let mut labels = backward_pass_info.get_label_ids_by_node(node.id).iter().map(|id| backward_pass_info.labels[*id].clone()).collect::<Vec<_>>();
                labels.sort_by(|a, b| a.reduced_cost.partial_cmp(&b.reduced_cost).unwrap());
                labels
            }).collect();
        
        // Depot nodes
        for label in forward_labels_by_node[self.nodes.depot.id].iter() {
            if let Some(finished_label) = self.finished_label(label, None, None) {
                routes += 1;
                    
                let lb = finished_label.reduced_cost;
                let mut curr = label;
                loop {
                    lbs[curr.in_arc] = lbs[curr.in_arc].min(lb);
                    if curr.predecessor.is_none() {
                        break;
                    }
                    curr = &forward_pass_info.labels[curr.predecessor.unwrap()];
                }
            }
        }

        for node in 1..self.nodes.nodes.len() {
            for forward_label in forward_labels_by_node[node].iter() {
                for backward_label in backward_labels_by_node[node].iter() {
                    if let Some(finished_label) = self.finished_label(forward_label, Some(backward_label), None) {
                        routes += 1;
                        let lb = finished_label.reduced_cost;
                        let mut curr = forward_label;
                        loop {
                            lbs[curr.in_arc] = lbs[curr.in_arc].min(lb);
                            if curr.predecessor.is_none() {
                                break;
                            }
                            curr = &forward_pass_info.labels[curr.predecessor.unwrap()];
                        }
                        let mut curr = backward_label;
                        loop {
                            lbs[curr.in_arc] = lbs[curr.in_arc].min(lb);
                            if curr.predecessor.is_none() {
                                break;
                            }
                            curr = &backward_pass_info.labels[curr.predecessor.unwrap()];
                        }
                    } else {
                        continue;
                    }
                }
            }
        }
        println!("Routes found: {}", routes);
        lbs
    }

    pub fn calculate_lower_rc_bounds(&mut self, verbose: bool) -> Vec<f64> {
        // Create the initial labels
        self.initialise_forward_labels();
        self.forward_pass((self.data.t_limit as f64 * FORWARD_BACKWARD_PASS_MARK) as usize);

        if verbose {
            println!("Forward pass finished");
            self.visited.print_visited_info();
        }

        let forward_pass_info = std::mem::replace(&mut self.visited, VisitedData::new(self.nodes.nodes.len()));
        self.graph = BucketGraph::new(self.data.t_limit, self.arcs.min_fragment_length, self.nodes.nodes.len());

        // Backward pass
        self.initialise_backward_labels();
        self.backward_pass((self.data.t_limit as f64 * (1.0-FORWARD_BACKWARD_PASS_MARK)) as usize);
        let backward_pass_info = std::mem::replace(&mut self.visited, VisitedData::new(self.nodes.nodes.len()));

        if verbose {
            println!("Backward pass finished");
            backward_pass_info.print_visited_info();
        }

        self.calculate_lower_rc_bounds_route_method(verbose, &forward_pass_info, &backward_pass_info)
    }

    pub fn _calculate_lower_rc_bounds_full_passes(&mut self, verbose: bool) -> HashMap<usize, f64> {
        // Create the initial labels
        self.initialise_forward_labels();
        self.forward_pass(self.data.t_limit);

        if verbose {
            println!("Full forward pass finished");
            self.visited.print_visited_info();
        }

        let forward_pass_info = std::mem::replace(&mut self.visited, VisitedData::new(self.nodes.nodes.len()));
        self.graph = BucketGraph::new(self.data.t_limit, self.arcs.min_fragment_length, self.nodes.nodes.len());

        // Backward pass
        self.initialise_backward_labels();
        self.backward_pass(self.data.t_limit);

        let backward_pass_info = std::mem::replace(&mut self.visited, VisitedData::new(self.nodes.nodes.len()));

        if verbose {
            println!("Full backward pass finished");
            backward_pass_info.print_visited_info();
        }

        let mut lower_bounds = HashMap::new();

        // Organise the backward labels into buckets based on duration and cost
        let backward_buckets_by_node: Vec<Vec<Vec<Label>>> = self.nodes.nodes.iter()
            .map(|node| {
                // This is the labels for a node
                let mut labels = backward_pass_info.get_label_ids_by_node(node.id).iter().map(|id| &backward_pass_info.labels[*id]).collect::<Vec<_>>();
                labels.sort_by(|a, b| a.reduced_cost.partial_cmp(&b.reduced_cost).unwrap());
                // Want to split this into duration buckets
                let width = self.arcs.min_fragment_length;
                let num_buckets = self.data.t_limit / width;
                let mut buckets = vec![Vec::new(); num_buckets];
                for label in labels.iter() {
                    // If a forward label plus an arc time lands me in bucket x, I want to look at all buckets >= x for valid durations
                    // (forward_dur + arc_dur) + backward_dur <= time_limit
                    let index = num_buckets - (label.duration / width);
                    buckets[index].push(**label);
                }
                buckets            
            }).collect();
        
        let forward_labels_by_node: Vec<Vec<Label>> = self.nodes.nodes.iter()
            .map(|node| {
                let mut labels = forward_pass_info.get_label_ids_by_node(node.id).iter().map(|id| forward_pass_info.labels[*id].clone()).collect::<Vec<_>>();
                labels.sort_by(|a, b| a.reduced_cost.partial_cmp(&b.reduced_cost).unwrap());
                labels
            }).collect();

        println!("Calculating lower bounds for {} arcs", self.arcs.num_arcs());
        let mut now = Instant::now();
        // Calculate the lowest reduced cost for every arc
        let (tx, rx) = std::sync::mpsc::channel();
        let mut handles = vec![];
        for arc in self.arcs.get_arcs() {
            if Instant::now() - now > Duration::from_secs(10) {
                println!("Processed {} arcs", lower_bounds.len());
                now = Instant::now();
            }

            let forward_labels: Vec<Label> = forward_labels_by_node[arc.start.id].clone();
            let backward_buckets = backward_buckets_by_node[arc.end.id].clone();

            // if verbose {
            //     println!("Calculating lower bound for arc {} -> {}", arc.start.id, arc.end.id);
            //     println!("  Forward labels: {}", forward_labels.len());
            //     println!("  Backward buckets: {}", backward_buckets.len());
            // }


            let mut covered = CoverSet::new(get_manager());
            let mut arc_reduced_cost = if self.vehicle_rc.is_some() {arc.cost as f64} else {0.0};

            let serviced = match arc.done.len() {
                0 => vec![],
                1 => vec![arc.done.left()],
                _ => vec![arc.done.left(), arc.done.right()],
            };

            for request_id in serviced.iter() {
                let reduction = self.cover_rc[*request_id];
                match covered.cover(*request_id) {
                    Ok(_) => {},
                    Err(_) => {continue;}, // Skip if the request is already serviced enough
                }
                arc_reduced_cost -= reduction;
            }

            let t_limit = self.data.t_limit;
            let covered = covered;
            let tx1 = tx.clone();
            let min_fragment_length = self.arcs.min_fragment_length;
            let vehicle_rc = self.vehicle_rc;
            let arc_copy = arc.clone();

            let handle = thread::spawn(move || {
                let bound = BucketPricer::calculate_lower_bound_for_arc(&arc_copy, forward_labels, backward_buckets, arc_reduced_cost, covered, t_limit, min_fragment_length, vehicle_rc);
                tx1.send((arc_copy.id, bound)).unwrap();
            });
            handles.push(handle);

        }
        for handle in handles {
            handle.join().unwrap();
        }
        drop(tx); // Close the channel
        for (arc_id, bound) in rx {
            lower_bounds.insert(arc_id, bound);
        }
        lower_bounds
    }

    fn calculate_lower_bound_for_arc(arc: &Arc, forward_labels: Vec<Label>, backward_buckets: Vec<Vec<Label>>, arc_reduced_cost: f64, covered: CoverSet, t_limit: usize, min_fragment_length: usize, vehicle_rc: Option<f64>) -> f64 {
        let mut lowest_reduced_cost = INFINITY;
        if arc.start.id == 0 {
            'backward: for (_idx, backward_label) in backward_buckets.iter().flatten().enumerate() {
                if arc.time + backward_label.duration > t_limit {
                    continue; // Skip if the backward label exceeds the time limit
                }
                let reduced_cost = backward_label.reduced_cost + arc_reduced_cost - vehicle_rc.unwrap_or(1.0) as f64;
                if reduced_cost >= lowest_reduced_cost {

                    break; // Skip if the reduced cost is not lower
                }
                let result = backward_label.coverset.combine(&covered);
                if result.is_err() {
                    continue 'backward; // Skip if the cover sets cannot be combined
                }

                if reduced_cost < lowest_reduced_cost {
                    lowest_reduced_cost = reduced_cost;
                }
                break;
            }
        }

        else if arc.end.id == 0 {
            'forward: for (_idx, forward_label) in forward_labels.iter().enumerate() {
                if forward_label.duration + arc.time > t_limit {
                    continue; // Skip if the forward label exceeds the time limit
                }
                let reduced_cost = forward_label.reduced_cost + arc_reduced_cost as f64;
                if reduced_cost >= lowest_reduced_cost {
                    break; // Skip if the reduced cost is not lower
                }

                let result = forward_label.coverset.combine(&covered);
                if result.is_err() {
                    continue 'forward; // Skip if the cover sets cannot be combined
                }

                if reduced_cost < lowest_reduced_cost {
                    lowest_reduced_cost = reduced_cost;
                }
                break;
            }
        }

        else {
            'forward: for forward_label in forward_labels {
                if forward_label.duration + arc.time > t_limit {
                    continue 'forward; // Skip if the forward label exceeds the time limit
                }
                let arrive_time = arc.time + forward_label.duration;
                let earliest_bucket = arrive_time / min_fragment_length;
                'bucket: for backwards_bucket in backward_buckets.iter().skip(earliest_bucket) {
                    'backward: for (_bucket_idx, backward_label) in backwards_bucket.iter().enumerate() {
                        let reduced_cost = forward_label.reduced_cost + backward_label.reduced_cost + arc_reduced_cost as f64;
                        if reduced_cost >= lowest_reduced_cost {
                            continue 'bucket; // Skip if the reduced cost is not lower
                        }

                        if arc.time + backward_label.duration + forward_label.duration > t_limit {
                            continue 'backward; // Skip if the backward label exceeds the time limit
                        }
                        
                        let result = forward_label.coverset.combine(&backward_label.coverset);
                        if result.is_err() {
                            continue 'backward; // Skip if the cover sets cannot be combined
                        }

                        let result = result.unwrap().combine(&covered);
                        if result.is_err() {
                            continue 'backward; // Skip if the cover sets cannot be combined
                        }

                        if reduced_cost < lowest_reduced_cost {
                            lowest_reduced_cost = reduced_cost;
                        }

                        continue 'bucket;
                    }
                }
            }
        }
        lowest_reduced_cost
    }
}

impl Pricer for BucketPricer<'_> {  
    fn solve_pricing_problem(&mut self, verbose: bool, objective: f64) -> Vec<Vec<Label>> {
        init_manager(self.data); // Cursed but necessary? It is what it is

        let candidate_labels = self.forward_backward_pass(NUM_ROUTES_PER_NODE_CALCULATED, objective, verbose);

        candidate_labels
    }
}


// struct PriorityStructure {
//     pub queue: BinaryHeap<Label>,
//     pub visited: VisitedData, // Label ids
//     pub mode: LPSolvePhase,
//     pub finished_labels: VisitedData, // Finished labels for each node
// }

// impl PriorityStructure {
//     fn new(mode: LPSolvePhase, num_nodes: usize) -> Self {
//         PriorityStructure {
//             queue: BinaryHeap::new(),
//             visited: VisitedData::new(num_nodes),
//             mode,
//             finished_labels: VisitedData::new(num_nodes), // Finished labels for each node
//         }
//     }

//     fn add_label(&mut self, node_id: usize, reduced_cost: f64, duration: usize, predecessor: usize, cost: usize, coverset: CoverSet) {  
//         let label = Label::new(self.visited.num_labels(), reduced_cost, duration, predecessor, cost, coverset, node_id);
//         self.push(label);
//     }

//     fn add_finished_label(&mut self, label: Label) {
//         // for prev_label in self.finished_labels[label.node_id].iter() {
//         //     if prev_label.dominates(&label, DominanceMode::DurRCCover) {
//         //         return;
//         //     }
//         // }
//         // self.finished_labels[label.node_id].push(label); // If the label is for the depot, add it to the depot bucket
//         self.finished_labels.add_with_dominance_mode(label, DominanceMode::DurRC);
//     }

//     fn push(&mut self, label: Label) {
//         if label.node_id == 0 {
//             self.add_finished_label(label); // If the label is for the depot, add it to the depot bucket
//         } else {
//             if self.visited.add_if_improvement(label, self.mode) {
//                 self.queue.push(label); // Add the label to the priority queue
//             }
//         }
//     }

//     fn pop(&mut self, iter: usize) -> Option<Label> {
//         let pop_val = self.queue.pop();
        
//         if let Some(ref label) = pop_val {
//             for candidate_best_label_id in self.visited.get_label_ids_by_node_covered(label.node_id, &label.coverset).iter() {
//                 if *candidate_best_label_id == label.id {
//                     return pop_val;
//                 }
//             }
//             self.visited.cut_out += 1;
//             return self.pop(iter+1);
//         }
//         None
//     }
// }



// pub struct QueuePricer<'a> {
//     priority_queue: PriorityStructure,
//     forward_queue: PriorityStructure,
//     backward_queue: PriorityStructure,
//     nodes: &'a NodeContainer,
//     arcs: &'a ArcContainer,
//     data: &'a SPDPData,
//     cover_rc: Vec<f64>,
//     vehicle_rc: Option<f64>,
// }

// impl<'a> QueuePricer<'a>  {
//     pub fn new(data: &'a SPDPData, nodes: &'a NodeContainer, arcs: &'a ArcContainer, cover_rc: Vec<f64>, vehicle_rc: Option<f64>, phase: LPSolvePhase) -> Self {
//         let priority_queue = PriorityStructure::new(phase, nodes.nodes.len());

//         QueuePricer {
//             priority_queue,
//             forward_queue: PriorityStructure::new(phase, nodes.nodes.len()),
//             backward_queue: PriorityStructure::new(phase, nodes.nodes.len()),
//             nodes,
//             arcs,
//             data,
//             cover_rc,
//             vehicle_rc,
//         }
//     }

//     fn extend_label(&mut self, label: &Label, arc: &Arc) {     
//         let next_node_id = arc.end.id;
//         let new_duration = label.duration + arc.time;
//         let new_cost = label.cost + arc.cost;

//         let serviced = match arc.done.len() {
//             0 => vec![],
//             1 => vec![arc.done.left()],
//             _ => vec![arc.done.left(), arc.done.right()],
//         };

//         let mut new_covered = label.coverset;
//         let mut new_reduced_cost = label.reduced_cost;

//         for request_id in serviced.iter() {
//             match new_covered.cover(*request_id) {
//                 Ok(_) => {},
//                 Err(_) => return, // Skip if the request is already serviced
//             }
//             new_reduced_cost -= self.cover_rc[*request_id];
//         }

//         new_reduced_cost += if self.vehicle_rc.is_some() {arc.cost as f64} else {0.0};

//         if new_duration > self.data.t_limit {
//             return; // Skip if the new duration exceeds the time limit
//         }
        
//         self.priority_queue.add_label(next_node_id, new_reduced_cost, new_duration, label.id, new_cost, new_covered);
//     }

//     fn extend_label_forward(&mut self, label: &Label, arc: &Arc) {     
//         let next_node_id = arc.end.id;
//         let new_duration = label.duration + arc.time;
//         let new_cost = label.cost + arc.cost;

//         let serviced = match arc.done.len() {
//             0 => vec![],
//             1 => vec![arc.done.left()],
//             _ => vec![arc.done.left(), arc.done.right()],
//         };

//         let mut new_covered = label.coverset;
//         let mut new_reduced_cost = label.reduced_cost;

//         for request_id in serviced.iter() {
//             match new_covered.cover(*request_id) {
//                 Ok(_) => {},
//                 Err(_) => return, // Skip if the request is already serviced
//             }
//             new_reduced_cost -= self.cover_rc[*request_id];
//         }

//         new_reduced_cost += if self.vehicle_rc.is_some() {arc.cost as f64} else {0.0};

//         if new_duration > self.data.t_limit {
//             return; // Skip if the new duration exceeds the time limit
//         }

//         self.forward_queue.add_label(next_node_id, new_reduced_cost, new_duration, label.id, new_cost, new_covered);
//     }

//     fn extend_label_backward(&mut self, label: &Label, arc: &Arc) {    
//         let next_node_id = arc.start.id;
//         let new_duration = label.duration + arc.time;
//         let new_cost = label.cost + arc.cost;

//         let serviced = match arc.done.len() {
//             0 => vec![],
//             1 => vec![arc.done.left()],
//             _ => vec![arc.done.left(), arc.done.right()],
//         };

//         let mut new_covered = label.coverset;
//         let mut new_reduced_cost = label.reduced_cost;

//         for request_id in serviced.iter() {
//             match new_covered.cover(*request_id) {
//                 Ok(_) => {},
//                 Err(_) => return, // Skip if the request is already serviced
//             }
//             new_reduced_cost -= self.cover_rc[*request_id];
//         }

//         new_reduced_cost += if self.vehicle_rc.is_some() {arc.cost as f64} else {0.0};

//         if new_duration > self.data.t_limit {
//             return; // Skip if the new duration exceeds the time limit
//         }

//         if new_duration > self.data.t_limit {
//             return; // Skip if the new duration exceeds the time limit
//         }
        
//         self.backward_queue.add_label(next_node_id, new_reduced_cost, new_duration, label.id, new_cost, new_covered);
//     }

//     fn forward_pass(&mut self) { 
//         // Create the initial labels
//         for arc in self.arcs.arcs_from.get(&self.nodes.depot).unwrap() {
//             let serviced = match arc.done.len() {
//                 0 => vec![],
//                 1 => vec![arc.done.left()],
//                 _ => vec![arc.done.left(), arc.done.right()],
//             };
//             assert!(serviced.len() == 0);
//             let coverset = CoverSet::new(get_manager());
//             assert!(arc.time == 0);
//             let new_reduced_cost = if self.vehicle_rc.is_none() {
//                     1.0
//                 } 
//                 else {
//                     arc.cost as f64 - self.vehicle_rc.unwrap()
//                 };
//             self.priority_queue.add_label(arc.end.id, new_reduced_cost, arc.time, 0, arc.cost, coverset);
//         }

//         while let Some(curr) = self.priority_queue.pop(0) {
//             let raw_ptr: *mut QueuePricer = self;  
//             unsafe {
//                 if let Some(pricer_ref) = raw_ptr.as_mut() {
//                     let forward_arcs = self.arcs.arcs_from.get(&self.nodes.nodes[curr.node_id]).unwrap();
//                     for arc in forward_arcs {
//                         pricer_ref.extend_label(&curr, arc);
//                     }
//                 }
//             }
//         }
//         self.priority_queue.finished_labels.get_label_ids_by_node(0).sort_by(|a, b| { 
//             self.priority_queue.finished_labels.labels[*a].reduced_cost.partial_cmp(&self.priority_queue.finished_labels.labels[*b].reduced_cost).unwrap()
//         });
//     }

//     fn forward_backward_pass(&mut self, k: usize, verbose: bool) -> Vec<Label> {
//         // Create the initial labels
//         for arc in self.arcs.arcs_from.get(&self.nodes.depot).unwrap() {
//             let serviced = match arc.done.len() {
//                 0 => vec![],
//                 1 => vec![arc.done.left()],
//                 _ => vec![arc.done.left(), arc.done.right()],
//             };
//             assert!(serviced.len() == 0);
//             let coverset = CoverSet::new(get_manager());

//             assert!(arc.time == 0);
//             let new_reduced_cost = if self.vehicle_rc.is_none() {
//                     1.0
//                 } 
//                 else {
//                     arc.cost as f64 - self.vehicle_rc.unwrap()
//                 };
//             if !arc.end.is_depot() {
//                 self.forward_queue.add_label(arc.end.id, new_reduced_cost, arc.time, 0, arc.cost, coverset);
//             }
//         }

//         // Forward pass to halfway point
//         while let Some(curr) = self.forward_queue.pop(0) {
//             if curr.duration > self.data.t_limit / 2 {
//                 // Reached the halfway point
//                 self.forward_queue.add_finished_label(curr);
//                 continue;
//             }
//             let raw_ptr: *mut QueuePricer = self;  
//             unsafe {
//                 if let Some(pricer_ref) = raw_ptr.as_mut() {
//                     let forward_arcs = self.arcs.arcs_from.get(&self.nodes.nodes[curr.node_id]).unwrap();
//                     for arc in forward_arcs {
//                         pricer_ref.extend_label_forward(&curr, arc);
//                     }
//                 }
//             }
//             self.forward_queue.add_finished_label(curr);
//         }

//         if verbose {
//             println!("Forward pass finished");
//             self.forward_queue.visited.print_visited_info();
//         }

//         // Backward pass

//         for arc in self.arcs.arcs_to.get(&self.nodes.depot).unwrap() {
//             let serviced = match arc.done.len() {
//                 0 => vec![],
//                 1 => vec![arc.done.left()],
//                 _ => vec![arc.done.left(), arc.done.right()],
//             };
//             let mut coverset = CoverSet::new(get_manager());
//             for idx in serviced.iter() {
//                 match coverset.cover(*idx) {
//                     Ok(_) => {},
//                     Err(_) => continue, // Skip if the request is already serviced
//                 }
//             }
//             let new_reduced_cost = if self.vehicle_rc.is_none() {
//                     0.0
//                 } 
//                 else {
//                     arc.cost as f64
//                 } - serviced.iter().map(|idx| self.cover_rc[*idx]).sum::<f64>();
//             if !arc.start.is_depot() {
//                 self.backward_queue.add_label(arc.start.id, new_reduced_cost, arc.time, 0, arc.cost, coverset);
//             }
//         }

//         while let Some(curr) = self.backward_queue.pop(0) {
//             let raw_ptr: *mut QueuePricer = self;  
//             unsafe {
//                 if let Some(pricer_ref) = raw_ptr.as_mut() {
//                     let backward_arcs = self.arcs.arcs_to.get(&self.nodes.nodes[curr.node_id]).unwrap();
//                     for arc in backward_arcs {
//                         if curr.duration + arc.time > self.data.t_limit / 2 {
//                             // Reached the halfway point
//                             self.backward_queue.add_finished_label(curr);
//                             continue;
//                         }
//                         pricer_ref.extend_label_backward(&curr, arc);
//                     }
//                 }
//             }
//             self.backward_queue.add_finished_label(curr);
//         }

//         if verbose {
//             println!("Backward pass finished");
//             self.backward_queue.visited.print_visited_info();
//         }

//         // Combine forward and backward labels
//         let forward_labels = &self.forward_queue.finished_labels;
//         let backward_labels = &self.backward_queue.finished_labels;
//         let mut candidate_labels = Vec::new();

//         let mut worst_rc = -EPS;
//         let mut worst_idx = 0;

//         for node_id in 0..self.nodes.nodes.len() {
//             let mut forward_bucket = forward_labels.get_label_ids_by_node(node_id).iter().map(|id| &forward_labels.labels[*id]).collect::<Vec<_>>();
//             let mut backward_bucket = backward_labels.get_label_ids_by_node(node_id).iter().map(|id| &backward_labels.labels[*id]).collect::<Vec<_>>();
//             // println!("Combining forward and backward labels at node {}", node_id);
//             // println!("Forward labels: {}", forward_bucket.len());
//             // println!("Backward labels: {}", backward_bucket.len());

//             forward_bucket.sort_by(|a, b| a.reduced_cost.partial_cmp(&b.reduced_cost).unwrap());
//             backward_bucket.sort_by(|a, b| a.reduced_cost.partial_cmp(&b.reduced_cost).unwrap());

//             for forward_label in forward_bucket.iter() {
//                 for backward_label in backward_bucket.iter() {
//                     if forward_label.reduced_cost + backward_label.reduced_cost >= worst_rc {
//                         break; // Skip if the combined reduced cost is not negative
//                     }
//                     if forward_label.duration + backward_label.duration <= self.data.t_limit {
//                         let new_covered = forward_label.coverset.combine(&backward_label.coverset);
//                         if new_covered.is_ok() {
//                             // Create a new label combining forward and backward labels
//                             // Ensure the new label is not dominated by any existing label
//                             let candidate_label = Label::new(
//                             0,
//                                 forward_label.reduced_cost + backward_label.reduced_cost,
//                                 forward_label.duration + backward_label.duration,
//                                 0,
//                                 forward_label.cost + backward_label.cost,
//                                 new_covered.unwrap(),
//                                 0,
//                             );

//                             if !candidate_labels.iter().any(|l: &Label| l.dominates(&candidate_label, DominanceMode::DurRCCover)) {
//                                 if candidate_labels.len() < k {
//                                     candidate_labels.push(candidate_label);
//                                     if candidate_label.reduced_cost > worst_rc {
//                                         worst_rc = candidate_label.reduced_cost;
//                                         worst_idx = candidate_labels.len() - 1;
//                                     }
//                                 } else {
//                                     candidate_labels[worst_idx] = candidate_label; // Replace the worst label with the new one
//                                     let mut idx = 0;
//                                     let mut worst_cost = candidate_labels[0].reduced_cost;
//                                     // Find the index of the worst label
//                                     for (i, label) in candidate_labels.iter().enumerate() {
//                                         if label.reduced_cost > worst_cost {
//                                             idx = i;
//                                             worst_cost = label.reduced_cost;
//                                         }
//                                     }
//                                     worst_idx = idx; // Update the worst index
//                                     worst_rc = worst_cost; // Update the worst reduced cost
//                                 }
//                             }
//                         }  
//                     }
//                 }
//             }
//         }
//         // Sort the candidate labels by reduced cost
//         candidate_labels.sort_by(|a, b| a.reduced_cost.partial_cmp(&b.reduced_cost).unwrap());

//         candidate_labels
//     }
// }

// impl Pricer for QueuePricer<'_> {
//     fn solve_pricing_problem(&mut self, verbose: bool) -> Vec<Vec<Label>> {
//         // self.forward_pass();
//         // let depot_labels = &self.priority_queue.finished_labels[0];

//         init_manager(self.data);

//         let depot_labels = self.forward_backward_pass(NUM_ROUTES_PER_NODE_CALCULATED, verbose);

//         let mut labels: Vec<Label> = Vec::new();

//         for idx in 0..min(NUM_ROUTES_PER_NODE_ADDED, depot_labels.len()) {
//             let label = &depot_labels[idx];
//             if label.reduced_cost < -EPS {
//                 labels.retain(|l| !label.dominates(l, DominanceMode::DurRCCover));
//                 labels.push(*label);
//             }
//         }

//         labels
//     }
// }


// pub struct BucketPricer<'a> {
//     bucket_graph: BucketGraph,
//     nodes: &'a NodeContainer,
//     arcs: &'a ArcContainer,
//     data: &'a SPDPData,
//     cover_rc: &'a Vec<f64>,
//     vehicle_rc: &'a Option<f64>,
// }

// impl<'a> BucketPricer<'a>  {
//     pub fn new(nodes: &'a NodeContainer, arcs: &'a ArcContainer, data: &'a SPDPData, cover_rc: &'a Vec<f64>, vehicle_rc: &'a Option<f64>, mode: DominanceMode) -> Self {
//         let bucket_graph = BucketGraph::new(data.t_limit, arcs.min_fragment_length, nodes.clone(), mode);

//         BucketPricer {
//             bucket_graph,
//             nodes,
//             arcs,
//             data,
//             cover_rc,
//             vehicle_rc,
//         }
//     }

//     fn extend_label(&mut self, label: &Label) {
//         let curr = self.nodes.nodes[label.node_id];
//         let forward_arcs = self.arcs.arcs_from.get(&curr).unwrap();

//         'arc_iteration: for arc in forward_arcs {
//             let next_node_id = arc.end.id;
//             let new_duration = label.duration + arc.time;
//             let new_cost = label.cost + arc.cost;

//             if new_duration > self.data.t_limit {
//                 continue 'arc_iteration; // Skip if the new duration exceeds the time limit
//             }

//             let serviced = match arc.done.len() {
//                 0 => vec![],
//                 1 => vec![arc.done.left()],
//                 _ => vec![arc.done.left(), arc.done.right()],
//             };

//             let mut new_covered = label.coverset.covered.clone();

//             for request_id in serviced.iter() {
//                 new_covered[*request_id] += 1;
//                 if new_covered[*request_id] > self.data.requests[*request_id].quantity {
//                     continue 'arc_iteration; // Skip if the request is already serviced
//                 }
//             }

//             let new_reduced_cost = (if self.vehicle_rc.is_some() {new_cost as f64 } else {1.0})
//                 - new_covered.iter().enumerate().map(|(r_id, q)| *q as f64 * self.cover_rc[r_id]).sum::<f64>();
            
//             self.bucket_graph.add_label(next_node_id, new_reduced_cost, new_duration, label.id, new_cost, new_covered);
//         }
//     }

//     fn forward_pass(&mut self) { 
//         // Create the initial labels
//         for arc in self.arcs.arcs_from.get(&self.nodes.depot).unwrap() {
//             let serviced = match arc.done.len() {
//                 0 => vec![],
//                 1 => vec![arc.done.left()],
//                 _ => vec![arc.done.left(), arc.done.right()],
//             };
//             assert!(serviced.len() == 0);
//             let covered = vec![0; self.data.num_requests];

//             assert!(arc.time == 0);
//             let new_reduced_cost = if self.vehicle_rc.is_none() {
//                     1.0
//                 } 
//                 else {
//                     arc.cost as f64 - self.vehicle_rc.unwrap()
//                 };
//             self.bucket_graph.add_label(arc.end.id, new_reduced_cost, arc.time, 0, arc.cost, covered);
//         }

//         print!("Bucket: ");
//         io::stdout().flush().unwrap();
//         let bar_inc = self.bucket_graph.num_buckets() / 10;
//         for bucket_idx in 0..self.bucket_graph.num_buckets() {
//             if bucket_idx % bar_inc == 0 {
//                 print!("|");
//                 io::stdout().flush().unwrap();
//             }
//             for node_idx in 0..self.nodes.nodes.len() {
//                 let raw_ptr: *mut BucketPricer = self;
//                 for label in self.bucket_graph.buckets[bucket_idx][node_idx].iter() {                    
//                     unsafe {
//                         if let Some(pricer_ref) = raw_ptr.as_mut() {
//                             pricer_ref.extend_label(label);
//                         }
//                     }
//                     // let prev_best = self.bucket_graph.previous_reduced_costs.get(&label.covered).unwrap_or(&label);
//                     // if prev_best.reduced_cost >= label.reduced_cost {
//                     //     self.bucket_graph.previous_reduced_costs.insert(label.covered.clone(), label.clone());
//                     // }
//                 }
//             }
//         }
//         println!("|");

//         self.bucket_graph.depot_bucket.sort_by(|a, b| a.reduced_cost.partial_cmp(&b.reduced_cost).unwrap());
//     }
// }

// impl Pricer for BucketPricer<'_> {
//     fn solve_pricing_problem(&mut self, k: usize, verbose: bool) -> Vec<Label> {
//         self.forward_pass();

//         let mut labels = Vec::new();

//         for idx in 0..min(k, self.bucket_graph.depot_bucket.len()) {
//             let label = &self.bucket_graph.depot_bucket[idx];
//             if label.reduced_cost < -EPS {
//                 labels.push(label.clone());
//             }
//         }
//         labels
//     }
// }

// struct BucketGraph {
//     buckets: Vec<Vec<Vec<Label>>>,
//     depot_bucket: Vec<Label>,
//     labels: Vec<Label>,
//     bucket_length: usize,
//     nodes: NodeContainer,
//     mode: DominanceMode,
//     previous_best: HashMap<Key, Vec<Label>>,
//     // cut: usize,
//     // miss: usize,
//     // better_rc: usize,
//     // better_duration: usize,
// }


// impl BucketGraph {
//     fn new(time_limit: usize, smallest_arc_duration: usize, nodes: NodeContainer, mode: DominanceMode) -> Self {
//         let num_buckets = time_limit / smallest_arc_duration + 1;
//         let buckets: Vec<Vec<Vec<Label>>> = vec![vec![Vec::new(); nodes.nodes.len()]; num_buckets];
//         let depot_bucket = vec![];
//         let labels = Vec::new();
//         let previous_best = HashMap::new();
//         BucketGraph { buckets, labels, bucket_length: smallest_arc_duration, depot_bucket, nodes, mode, previous_best} //data}
//     }  

//     fn get_bucket_by_duration(&mut self, node_id: usize, time: usize) -> &mut Vec<Label> {
//         let bucket_index = time / self.bucket_length;
//         if bucket_index >= self.buckets.len() {
//             panic!("Bucket index out of bounds: {}", bucket_index);
//         }
//         if node_id >= self.buckets[0].len() {
//             panic!("Node ID out of bounds: {}", node_id);
//         }
//         &mut self.buckets[bucket_index][node_id]
//     }

//     fn get_labels_iter_by_index(&self, node_id: usize, index: usize) -> std::slice::Iter<'_, Label> {
//         self.buckets[node_id][index].iter()
//     }

//     fn num_labels(&self) -> usize {
//         self.labels.len()
//     }

//     fn num_buckets(&self) -> usize {
//         self.buckets.len()
//     }

//     fn add_label(&mut self, node_id: usize, reduced_cost: f64, duration: usize, predecessor: usize, cost: usize, covered: Vec<usize>) {  
//         let label = Label::new(self.num_labels(), reduced_cost, duration, predecessor, cost, covered, node_id);

//         let key = Key {
//             covered: label.coverset.covered.clone(),
//             node_id: label.node_id,
//         };

//         if let Some(labels) = self.previous_best.get_mut(&key) {
//             if labels.iter().any(|l| l.dominates(&label, self.mode)) {
//                 return; // Skip if a label dominates the new label
//             }
//             labels.retain(|l| !label.dominates(l, self.mode)); // Remove dominated labels
//             labels.push(label.clone()); // Add the new label to the bucket
//         } else {
//             self.previous_best.insert(key, vec![label.clone()]);
//         }

//         let mode = self.mode;
//         let bucket = if self.nodes.nodes[node_id].is_depot() {&mut self.depot_bucket} else {self.get_bucket_by_duration(node_id, duration)};
//         if bucket.iter().any(|l| l.dominates(&label, mode)) {
//             return; // Skip if a label dominates the new label
//         }

//         bucket.retain(|l| !label.dominates(l, mode)); // Remove dominated labels
//         bucket.push(label.clone()); // Add the new label to the bucket


//         self.labels.push(label); // Store the label in the labels vector
//     }
// }