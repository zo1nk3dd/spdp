use std::cmp::{min, Ordering};
use std::collections::{BinaryHeap, HashMap};
use std::io::Write;
use std::{f64, fmt, io, panic, result, vec};

use crate::fragment::{Arc, ArcContainer, NodeContainer};
use crate::utils::{SPDPData};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DominanceMode {
    RC,
    DurRC,
    DurRCCover,
    Dur,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LPSolvePhase {
    VehicleNoCover,
    VehicleCover,
    CostNoCover,
    CostCover,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct Key {
    covered: Vec<usize>,
    node_id: usize,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CoverSet {
    pub covered: Vec<usize>,
}

impl CoverSet {
    fn visits_leq_than(&self, other: &CoverSet) -> bool {
        self.covered.iter().zip(other.covered.iter()).all(|(a, b)| a <= b)
    }
}

impl PartialOrd for CoverSet {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CoverSet {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare based on the number of covered requests
        for (a, b) in self.covered.iter().zip(&other.covered) {
            if a < b {
                return Ordering::Less;
            }
            else if a > b {
                return Ordering::Greater;
            }
        }
        Ordering::Equal
    }
}

#[derive(Debug, Clone)]
struct VisitedData {
    pub visited: Vec<Vec<Vec<usize>>>,
    pub labels: Vec<Label>,
    pub cut_in: usize,
    pub cut_out: usize,
    pub better: usize,
}

impl VisitedData {
    fn new(num_nodes: usize) -> Self {
        VisitedData {
            visited: vec![vec![]; num_nodes],
            labels: Vec::new(),
            cut_in: 0,
            cut_out: 0,
            better: 0,
        }
    }

    fn add_if_improvement(&mut self, label: Label, phase: LPSolvePhase) -> bool {        
        let bucket = &mut self.visited[label.node_id];
        // assert!(bucket.iter().all(|l| !l.is_empty()), "Bucket should not contain empty labels");

        if bucket.is_empty() {
            assert!(self.labels.len() == label.id);
            bucket.push(vec![label.id]);
            self.labels.push(label);
            return true; // If no labels visited, this is an improvement
        }

        // Vehicle No Cover, we want to store only the best reduced cost label at each node
        if phase == LPSolvePhase::VehicleNoCover || phase == LPSolvePhase::CostNoCover {
            let prev_label = &self.labels[bucket[0][0]];
            if label.dominates(prev_label, DominanceMode::DurRC) {
                bucket[0][0] = label.id;
                self.cut_out += 1;
                self.better += 1; // Count this as a better label

            } else {
                self.cut_in += 1;
                return false; // If the new label does not dominate the best label, it's not an improvement
            }
        }
        // We care about the best label for each node, covered set
        else if phase == LPSolvePhase::VehicleCover {
            let result = bucket.binary_search_by(|l| {
                let curr_coverset = &self.labels[l[0]].coverset;
                curr_coverset.cmp(&label.coverset)
            });

            if result.is_ok() {
                // This cover set has been previously found
                let prev_label = &self.labels[bucket[result.unwrap()][0]];
                assert!(bucket[result.unwrap()].len() == 1);
                let mode = DominanceMode::Dur;
                if label.dominates(&prev_label, mode) {
                    bucket[result.unwrap()][0] = label.id;
                    self.cut_out += 1;
                    self.better += 1; // Count this as a better label
                } else {
                    self.cut_in += 1;
                    return false; // If the new label does not dominate the best label, it's not an improvement
                }
            }

            else {
                // This cover set has not been found before
                let idx = result.unwrap_err();
                bucket.insert(idx, vec![label.id]);
                assert!(self.labels.len() == label.id);
            }
        }

        else if phase == LPSolvePhase::CostCover {
            // We care about the best label for each node, cover
            // Labels can be dominated by smaller coversets with better reduced costs
            // let mut is_dominated: bool = false;
            // for prev_label_ids in bucket.iter_mut() {
            //     let prev_cover_set = &self.labels[prev_label_ids[0]].coverset;
            //     if prev_cover_set.visits_leq_than(&label.coverset) {
            //         // We can only be dominated if the previous label vists less than or equal to the new label
            //         for id in prev_label_ids.iter() {
            //             if self.labels[*id].dominates(&label, DominanceMode::DurRC) {
            //                 self.cut_in += 1;
            //                 is_dominated = true;
            //                 break; // If the new label is dominated by any of the previous labels, it's not an improvement
            //             }
            //         }
            //     }
            //     // else {
            //     //     // We can only dominate if the new label visits greater than or equal to the previous label
            //     //     // This is the greater than case
            //     //     prev_label_ids.retain(|id| {
            //     //         // Remove dominated labels
            //     //         !label.dominates(&self.labels[*id], DominanceMode::DurRC)
            //     //     });
            //     // }
            // }
            // // bucket.retain(|l| !l.is_empty());
            // if is_dominated {
            //     return false; // If the new label is dominated by any of the previous labels, it's not an improvement
            // }

            // If we reach here, the label is an improvement
            let result = bucket.binary_search_by(|l| {
                let curr_coverset = &self.labels[l[0]].coverset;
                curr_coverset.cmp(&label.coverset)
            });

            if result.is_ok() {
                for id in bucket[result.unwrap()].iter() {
                    if self.labels[*id].dominates(&label, DominanceMode::DurRC) {
                        self.cut_in += 1;
                        return false; // If the new label is dominated by any of the previous labels, it's not an improvement
                    }
                }
                let len_before = bucket[result.unwrap()].len();
                bucket[result.unwrap()].retain(|id| {
                    // Remove dominated labels
                    !label.dominates(&self.labels[*id], DominanceMode::DurRC)
                });
                let len_after = bucket[result.unwrap()].len();
                self.cut_out += len_before - len_after;
                self.better += 1;
                bucket[result.unwrap()].push(label.id);
            }
            else {
                // This cover set has not been found before
                let idx = result.unwrap_err();
                bucket.insert(idx, vec![label.id]);
            }
        }
        self.labels.push(label); // Add the label to the labels vector
        true
    }

    fn get_label_ids_by_node(&self, node_id: usize) -> Vec<usize> {
        self.visited[node_id].iter().flatten().map(|id| *id).collect()
    }

    fn get_label_ids_by_node_covered(&self, node_id: usize, covered: &CoverSet) -> Vec<usize> {
        let result = self.visited[node_id].binary_search_by(|l| {
            let curr_coverset = &self.labels[l[0]].coverset;
            curr_coverset.cmp(covered)
        });
        if result.is_ok() {
            self.visited[node_id][result.unwrap()].iter().map(|id| *id).collect()
        } else {    
            vec![]
        }
    }
    
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

struct PriorityStructure {
    pub queue: BinaryHeap<Label>,
    pub visited: VisitedData, // Label ids
    pub mode: LPSolvePhase,
    pub finished_labels: Vec<Vec<Label>>,
}

impl PriorityStructure {
    fn new(mode: LPSolvePhase, num_nodes: usize) -> Self {
        PriorityStructure {
            queue: BinaryHeap::new(),
            visited: VisitedData::new(num_nodes),
            mode,
            finished_labels: vec![Vec::new(); num_nodes], // Finished labels for each node
        }
    }

    fn add_label(&mut self, node_id: usize, reduced_cost: f64, duration: usize, predecessor: usize, cost: usize, covered: Vec<usize>) {  
        let label = Label::new(self.visited.num_labels(), reduced_cost, duration, predecessor, cost, covered.clone(), node_id);
        self.push(label);
    }

    fn add_finished_label(&mut self, label: Label) {
        for prev_label in self.finished_labels[label.node_id].iter() {
            if prev_label.dominates(&label, DominanceMode::DurRCCover) {
                return;
            }
        }
        self.finished_labels[label.node_id].push(label); // If the label is for the depot, add it to the depot bucket
    }

    fn push(&mut self, label: Label) {
        if label.node_id == 0 {
            self.add_finished_label(label); // If the label is for the depot, add it to the depot bucket
        } else {
            if self.visited.add_if_improvement(label.clone(), self.mode) {
                self.queue.push(label); // Add the label to the priority queue
            }
        }
    }

    fn pop(&mut self, iter: usize) -> Option<Label> {
        let pop_val = self.queue.pop();
        
        if let Some(ref label) = pop_val {
            for candidate_best_label_id in self.visited.get_label_ids_by_node_covered(label.node_id, &label.coverset).iter() {
                if *candidate_best_label_id == label.id {
                    return pop_val;
                }
            }
            self.visited.cut_out += 1;
            return self.pop(iter+1);
        }
        None
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Label {
    id: usize,
    pub reduced_cost: f64,
    duration: usize,
    predecessor: usize,
    pub cost: usize,
    pub coverset: CoverSet,
    pub node_id: usize,
}

impl Eq for Label {}

impl Ord for Label {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.duration.cmp(&self.duration)
    }
}

impl PartialOrd for Label {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Label {
    fn new(id: usize, reduced_cost: f64, duration: usize, predecessor: usize, cost: usize, covered: Vec<usize>, node_id: usize,) -> Self {       
        let coverset: CoverSet = CoverSet { covered }; 
        Label {
            id,
            reduced_cost,
            duration,
            predecessor,
            cost,
            coverset,
            node_id,
        }
    }

    fn dominates(&self, other: &Label, mode: DominanceMode) -> bool {
        match mode {
            DominanceMode::RC => {
                self.reduced_cost - other.reduced_cost < 1e-6
            },
            DominanceMode::DurRC => {
                self.reduced_cost - other.reduced_cost < 1e-6 && self.duration <= other.duration
            },
            DominanceMode::DurRCCover => {
                self.reduced_cost - other.reduced_cost < 1e-6 && self.duration <= other.duration && self.coverset.visits_leq_than(&other.coverset)
            },
            DominanceMode::Dur => {
                self.duration <= other.duration
            },
        }
    }

    fn visits_less_eq_than(&self, other: &Label) -> bool {
        self.coverset.visits_leq_than(&other.coverset)
    }
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rc: {:.2}, d: {}, covered: {:?}", 
            self.reduced_cost, self.duration, self.coverset)
    }
}
pub trait Pricer {
    fn solve_pricing_problem(&mut self, k: usize, verbose: bool) -> Vec<Label>;
}

pub struct QueuePricer<'a> {
    priority_queue: PriorityStructure,
    forward_queue: PriorityStructure,
    backward_queue: PriorityStructure,
    nodes: &'a NodeContainer,
    arcs: &'a ArcContainer,
    data: &'a SPDPData,
    cover_rc: &'a Vec<f64>,
    vehicle_rc: &'a Option<f64>,
}

impl<'a> QueuePricer<'a>  {
    pub fn new(nodes: &'a NodeContainer, arcs: &'a ArcContainer, data: &'a SPDPData, cover_rc: &'a Vec<f64>, vehicle_rc: &'a Option<f64>, mode: LPSolvePhase) -> Self {
        let priority_queue = PriorityStructure::new(mode, nodes.nodes.len());

        QueuePricer {
            priority_queue,
            forward_queue: PriorityStructure::new(mode, nodes.nodes.len()),
            backward_queue: PriorityStructure::new(mode, nodes.nodes.len()),
            nodes,
            arcs,
            data,
            cover_rc,
            vehicle_rc,
        }
    }

    fn extend_label(&mut self, label: &Label, arc: &Arc) {     
        let next_node_id = arc.end.id;
        let new_duration = label.duration + arc.time;
        let new_cost = label.cost + arc.cost;

        let serviced = match arc.done.len() {
            0 => vec![],
            1 => vec![arc.done.left()],
            _ => vec![arc.done.left(), arc.done.right()],
        };

        let mut new_covered = label.coverset.covered.clone();

        for request_id in serviced.iter() {
            new_covered[*request_id] += 1;
            if new_covered[*request_id] > self.data.requests[*request_id].quantity {
                return; // Skip if the request is already serviced
            }
        }

        let new_reduced_cost = (if self.vehicle_rc.is_some() {new_cost as f64 - self.vehicle_rc.unwrap()} else {1.0})
            - new_covered.iter().enumerate().map(|(r_id, q)| *q as f64 * self.cover_rc[r_id]).sum::<f64>();

        if new_duration > self.data.t_limit {
            return; // Skip if the new duration exceeds the time limit
        }
        
        self.priority_queue.add_label(next_node_id, new_reduced_cost, new_duration, label.id, new_cost, new_covered);
    }

    fn extend_label_forward(&mut self, label: &Label, arc: &Arc) {     
        let next_node_id = arc.end.id;
        let new_duration = label.duration + arc.time;
        let new_cost = label.cost + arc.cost;

        let serviced = match arc.done.len() {
            0 => vec![],
            1 => vec![arc.done.left()],
            _ => vec![arc.done.left(), arc.done.right()],
        };

        let mut new_covered = label.coverset.covered.clone();

        for request_id in serviced.iter() {
            new_covered[*request_id] += 1;
            if new_covered[*request_id] > self.data.requests[*request_id].quantity {
                return; // Skip if the request is already serviced
            }
        }

        let new_reduced_cost = (if self.vehicle_rc.is_some() {new_cost as f64 - self.vehicle_rc.unwrap()} else {1.0})
            - new_covered.iter().enumerate().map(|(r_id, q)| *q as f64 * self.cover_rc[r_id]).sum::<f64>();

        if new_duration > self.data.t_limit {
            return; // Skip if the new duration exceeds the time limit
        }
        
        self.forward_queue.add_label(next_node_id, new_reduced_cost, new_duration, label.id, new_cost, new_covered);
    }

    fn extend_label_backward(&mut self, label: &Label, arc: &Arc) {    
        let next_node_id = arc.start.id;
        let new_duration = label.duration + arc.time;
        let new_cost = label.cost + arc.cost;

        let serviced = match arc.done.len() {
            0 => vec![],
            1 => vec![arc.done.left()],
            _ => vec![arc.done.left(), arc.done.right()],
        };

        let mut new_covered = label.coverset.covered.clone();

        for request_id in serviced.iter() {
            new_covered[*request_id] += 1;
            if new_covered[*request_id] > self.data.requests[*request_id].quantity {
                return; // Skip if the request is already serviced
            }
        }

        let new_reduced_cost = label.reduced_cost + arc.cost as f64 - serviced.iter().map(|r_id| self.cover_rc[*r_id]).sum::<f64>();

        if new_duration > self.data.t_limit {
            return; // Skip if the new duration exceeds the time limit
        }
        
        self.backward_queue.add_label(next_node_id, new_reduced_cost, new_duration, label.id, new_cost, new_covered);
    }

    fn forward_pass(&mut self) { 
        // Create the initial labels
        for arc in self.arcs.arcs_from.get(&self.nodes.depot).unwrap() {
            let serviced = match arc.done.len() {
                0 => vec![],
                1 => vec![arc.done.left()],
                _ => vec![arc.done.left(), arc.done.right()],
            };
            assert!(serviced.len() == 0);
            let covered = vec![0; self.data.num_requests];

            assert!(arc.time == 0);
            let new_reduced_cost = if self.vehicle_rc.is_none() {
                    1.0
                } 
                else {
                    arc.cost as f64 - self.vehicle_rc.unwrap()
                };
            self.priority_queue.add_label(arc.end.id, new_reduced_cost, arc.time, 0, arc.cost, covered);
        }

        while let Some(curr) = self.priority_queue.pop(0) {
            let raw_ptr: *mut QueuePricer = self;  
            unsafe {
                if let Some(pricer_ref) = raw_ptr.as_mut() {
                    let forward_arcs = self.arcs.arcs_from.get(&self.nodes.nodes[curr.node_id]).unwrap();
                    for arc in forward_arcs {
                        pricer_ref.extend_label(&curr, arc);
                    }
                }
            }
        }
        self.priority_queue.finished_labels[0].sort_by(|a, b| a.reduced_cost.partial_cmp(&b.reduced_cost).unwrap());
    }

    fn forward_backward_pass(&mut self, verbose: bool) -> Vec<Label> {
        // Create the initial labels
        for arc in self.arcs.arcs_from.get(&self.nodes.depot).unwrap() {
            let serviced = match arc.done.len() {
                0 => vec![],
                1 => vec![arc.done.left()],
                _ => vec![arc.done.left(), arc.done.right()],
            };
            assert!(serviced.len() == 0);
            let covered = vec![0; self.data.num_requests];

            assert!(arc.time == 0);
            let new_reduced_cost = if self.vehicle_rc.is_none() {
                    1.0
                } 
                else {
                    arc.cost as f64 - self.vehicle_rc.unwrap()
                };
            if !arc.end.is_depot() {
                self.forward_queue.add_label(arc.end.id, new_reduced_cost, arc.time, 0, arc.cost, covered);
            }
        }

        // Forward pass to halfway point
        while let Some(curr) = self.forward_queue.pop(0) {
            if curr.duration > self.data.t_limit / 2 {
                // Reached the halfway point
                self.forward_queue.add_finished_label(curr);
                continue;
            }
            let raw_ptr: *mut QueuePricer = self;  
            unsafe {
                if let Some(pricer_ref) = raw_ptr.as_mut() {
                    let forward_arcs = self.arcs.arcs_from.get(&self.nodes.nodes[curr.node_id]).unwrap();
                    for arc in forward_arcs {
                        pricer_ref.extend_label_forward(&curr, arc);
                    }
                }
            }
            self.forward_queue.add_finished_label(curr);
        }

        if verbose {
            println!("Forward pass finished");
            self.forward_queue.visited.print_visited_info();
        }

        // Backward pass

        for arc in self.arcs.arcs_to.get(&self.nodes.depot).unwrap() {
            let serviced = match arc.done.len() {
                0 => vec![],
                1 => vec![arc.done.left()],
                _ => vec![arc.done.left(), arc.done.right()],
            };
            let mut covered = vec![0; self.data.num_requests];
            for idx in serviced.iter() {
                covered[*idx] += 1;
            }
            let new_reduced_cost = if self.vehicle_rc.is_none() {
                    0.0
                } 
                else {
                    arc.cost as f64
                } - serviced.iter().map(|idx| self.cover_rc[*idx]).sum::<f64>();
            if !arc.start.is_depot() {
                self.backward_queue.add_label(arc.start.id, new_reduced_cost, arc.time, 0, arc.cost, covered);
            }
        }

        while let Some(curr) = self.backward_queue.pop(0) {
            let raw_ptr: *mut QueuePricer = self;  
            unsafe {
                if let Some(pricer_ref) = raw_ptr.as_mut() {
                    let backward_arcs = self.arcs.arcs_to.get(&self.nodes.nodes[curr.node_id]).unwrap();
                    for arc in backward_arcs {
                        if curr.duration + arc.time > self.data.t_limit / 2 {
                            // Reached the halfway point
                            self.backward_queue.add_finished_label(curr.clone());
                            continue;
                        }
                        pricer_ref.extend_label_backward(&curr, arc);
                    }
                }
            }
            self.backward_queue.add_finished_label(curr.clone());
        }

        if verbose {
            println!("Backward pass finished");
            self.backward_queue.visited.print_visited_info();
        }

        // Combine forward and backward labels
        let forward_labels = &self.forward_queue.finished_labels;
        let backward_labels = &self.backward_queue.finished_labels;
        let mut candidate_labels = Vec::new();

        for node_id in 0..self.nodes.nodes.len() {
            let forward_bucket = &forward_labels[node_id];
            let backward_bucket = &backward_labels[node_id];

            for forward_label in forward_bucket.iter() {
                for backward_label in backward_bucket.iter() {
                    if forward_label.duration + backward_label.duration <= self.data.t_limit && forward_label.reduced_cost + backward_label.reduced_cost < -1e-6 {
                        let mut new_covered = forward_label.coverset.covered.clone();
                        for (i, &count) in backward_label.coverset.covered.iter().enumerate() {
                            new_covered[i] += count;
                        }
                        if new_covered.iter().zip(self.data.requests.iter()).all(|(c, r)| c <= &r.quantity) {
                            // Create a new label combining forward and backward labels
                            // Ensure the new label is not dominated by any existing label
                            let candidate_label = Label::new(
                            0,
                                forward_label.reduced_cost + backward_label.reduced_cost,
                                forward_label.duration + backward_label.duration,
                                0,
                                forward_label.cost + backward_label.cost,
                                new_covered,
                                0,
                            );

                            if !candidate_labels.iter().any(|l: &Label| l.dominates(&candidate_label, DominanceMode::DurRCCover)) {
                                candidate_labels.push(candidate_label);
                            }
                        }  
                    }
                }
            }
        }
        // Sort the candidate labels by reduced cost
        candidate_labels.sort_by(|a, b| a.reduced_cost.partial_cmp(&b.reduced_cost).unwrap());

        candidate_labels
    }
}

impl Pricer for QueuePricer<'_> {
    fn solve_pricing_problem(&mut self, k: usize, verbose: bool) -> Vec<Label> {
        // self.forward_pass();
        // let depot_labels = &self.priority_queue.finished_labels[0];

        let depot_labels = self.forward_backward_pass(verbose);

        let mut labels: Vec<Label> = Vec::new();

        for idx in 0..min(k, depot_labels.len()) {
            let label = &depot_labels[idx];
            if label.reduced_cost < -1e-6 {
                labels.retain(|l| !label.dominates(l, DominanceMode::DurRCCover));
                labels.push(label.clone());
            }
        }

        labels
    }
}


pub struct BucketPricer<'a> {
    bucket_graph: BucketGraph,
    nodes: &'a NodeContainer,
    arcs: &'a ArcContainer,
    data: &'a SPDPData,
    cover_rc: &'a Vec<f64>,
    vehicle_rc: &'a Option<f64>,
}

impl<'a> BucketPricer<'a>  {
    pub fn new(nodes: &'a NodeContainer, arcs: &'a ArcContainer, data: &'a SPDPData, cover_rc: &'a Vec<f64>, vehicle_rc: &'a Option<f64>, mode: DominanceMode) -> Self {
        let bucket_graph = BucketGraph::new(data.t_limit, arcs.min_fragment_length, nodes.clone(), mode);

        BucketPricer {
            bucket_graph,
            nodes,
            arcs,
            data,
            cover_rc,
            vehicle_rc,
        }
    }

    fn extend_label(&mut self, label: &Label) {
        let curr = self.nodes.nodes[label.node_id];
        let forward_arcs = self.arcs.arcs_from.get(&curr).unwrap();

        'arc_iteration: for arc in forward_arcs {
            let next_node_id = arc.end.id;
            let new_duration = label.duration + arc.time;
            let new_cost = label.cost + arc.cost;

            if new_duration > self.data.t_limit {
                continue 'arc_iteration; // Skip if the new duration exceeds the time limit
            }

            let serviced = match arc.done.len() {
                0 => vec![],
                1 => vec![arc.done.left()],
                _ => vec![arc.done.left(), arc.done.right()],
            };

            let mut new_covered = label.coverset.covered.clone();

            for request_id in serviced.iter() {
                new_covered[*request_id] += 1;
                if new_covered[*request_id] > self.data.requests[*request_id].quantity {
                    continue 'arc_iteration; // Skip if the request is already serviced
                }
            }

            let new_reduced_cost = (if self.vehicle_rc.is_some() {new_cost as f64 } else {1.0})
                - new_covered.iter().enumerate().map(|(r_id, q)| *q as f64 * self.cover_rc[r_id]).sum::<f64>();
            
            self.bucket_graph.add_label(next_node_id, new_reduced_cost, new_duration, label.id, new_cost, new_covered);
        }
    }

    fn forward_pass(&mut self) { 
        // Create the initial labels
        for arc in self.arcs.arcs_from.get(&self.nodes.depot).unwrap() {
            let serviced = match arc.done.len() {
                0 => vec![],
                1 => vec![arc.done.left()],
                _ => vec![arc.done.left(), arc.done.right()],
            };
            assert!(serviced.len() == 0);
            let covered = vec![0; self.data.num_requests];

            assert!(arc.time == 0);
            let new_reduced_cost = if self.vehicle_rc.is_none() {
                    1.0
                } 
                else {
                    arc.cost as f64 - self.vehicle_rc.unwrap()
                };
            self.bucket_graph.add_label(arc.end.id, new_reduced_cost, arc.time, 0, arc.cost, covered);
        }

        print!("Bucket: ");
        io::stdout().flush().unwrap();
        let bar_inc = self.bucket_graph.num_buckets() / 10;
        for bucket_idx in 0..self.bucket_graph.num_buckets() {
            if bucket_idx % bar_inc == 0 {
                print!("|");
                io::stdout().flush().unwrap();
            }
            for node_idx in 0..self.nodes.nodes.len() {
                let raw_ptr: *mut BucketPricer = self;
                for label in self.bucket_graph.buckets[bucket_idx][node_idx].iter() {                    
                    unsafe {
                        if let Some(pricer_ref) = raw_ptr.as_mut() {
                            pricer_ref.extend_label(label);
                        }
                    }
                    // let prev_best = self.bucket_graph.previous_reduced_costs.get(&label.covered).unwrap_or(&label);
                    // if prev_best.reduced_cost >= label.reduced_cost {
                    //     self.bucket_graph.previous_reduced_costs.insert(label.covered.clone(), label.clone());
                    // }
                }
            }
        }
        println!("|");

        self.bucket_graph.depot_bucket.sort_by(|a, b| a.reduced_cost.partial_cmp(&b.reduced_cost).unwrap());
    }
}

impl Pricer for BucketPricer<'_> {
    fn solve_pricing_problem(&mut self, k: usize, verbose: bool) -> Vec<Label> {
        self.forward_pass();

        let mut labels = Vec::new();

        for idx in 0..min(k, self.bucket_graph.depot_bucket.len()) {
            let label = &self.bucket_graph.depot_bucket[idx];
            if label.reduced_cost < -1e-6 {
                labels.push(label.clone());
            }
        }
        labels
    }
}

struct BucketGraph {
    buckets: Vec<Vec<Vec<Label>>>,
    depot_bucket: Vec<Label>,
    labels: Vec<Label>,
    bucket_length: usize,
    nodes: NodeContainer,
    mode: DominanceMode,
    previous_best: HashMap<Key, Vec<Label>>,
    // cut: usize,
    // miss: usize,
    // better_rc: usize,
    // better_duration: usize,
}


impl BucketGraph {
    fn new(time_limit: usize, smallest_arc_duration: usize, nodes: NodeContainer, mode: DominanceMode) -> Self {
        let num_buckets = time_limit / smallest_arc_duration + 1;
        let buckets: Vec<Vec<Vec<Label>>> = vec![vec![Vec::new(); nodes.nodes.len()]; num_buckets];
        let depot_bucket = vec![];
        let labels = Vec::new();
        let previous_best = HashMap::new();
        BucketGraph { buckets, labels, bucket_length: smallest_arc_duration, depot_bucket, nodes, mode, previous_best} //data}
    }  

    fn get_bucket_by_duration(&mut self, node_id: usize, time: usize) -> &mut Vec<Label> {
        let bucket_index = time / self.bucket_length;
        if bucket_index >= self.buckets.len() {
            panic!("Bucket index out of bounds: {}", bucket_index);
        }
        if node_id >= self.buckets[0].len() {
            panic!("Node ID out of bounds: {}", node_id);
        }
        &mut self.buckets[bucket_index][node_id]
    }

    fn get_labels_iter_by_index(&self, node_id: usize, index: usize) -> std::slice::Iter<'_, Label> {
        self.buckets[node_id][index].iter()
    }

    fn num_labels(&self) -> usize {
        self.labels.len()
    }

    fn num_buckets(&self) -> usize {
        self.buckets.len()
    }

    fn add_label(&mut self, node_id: usize, reduced_cost: f64, duration: usize, predecessor: usize, cost: usize, covered: Vec<usize>) {  
        let label = Label::new(self.num_labels(), reduced_cost, duration, predecessor, cost, covered, node_id);

        let key = Key {
            covered: label.coverset.covered.clone(),
            node_id: label.node_id,
        };

        if let Some(labels) = self.previous_best.get_mut(&key) {
            if labels.iter().any(|l| l.dominates(&label, self.mode)) {
                return; // Skip if a label dominates the new label
            }
            labels.retain(|l| !label.dominates(l, self.mode)); // Remove dominated labels
            labels.push(label.clone()); // Add the new label to the bucket
        } else {
            self.previous_best.insert(key, vec![label.clone()]);
        }

        let mode = self.mode;
        let bucket = if self.nodes.nodes[node_id].is_depot() {&mut self.depot_bucket} else {self.get_bucket_by_duration(node_id, duration)};
        if bucket.iter().any(|l| l.dominates(&label, mode)) {
            return; // Skip if a label dominates the new label
        }

        bucket.retain(|l| !label.dominates(l, mode)); // Remove dominated labels
        bucket.push(label.clone()); // Add the new label to the bucket


        self.labels.push(label); // Store the label in the labels vector
    }
}

