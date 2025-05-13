use core::panic;
use std::cmp::min;
use std::collections::{BinaryHeap, HashMap};
use std::io::{self, Write};

use grb::attribute::ConstrIntAttr::Lazy;

use crate::fragment::{NodeContainer, ArcContainer};
use crate::utils::{Node, SPDPData};

#[derive(Debug, Clone, Copy)]
pub struct DominanceMode {
    pub rc: bool,
    pub duration: bool,
    pub coverage: bool,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct Key {
    covered: Vec<usize>,
    node_id: usize,
}

struct PriorityStructure {
    pub queue: BinaryHeap<Label>,
    pub visited: HashMap<Key, Vec<Label>>,
    pub mode: DominanceMode,
    pub labels: Vec<Label>,
    pub depot_labels: Vec<Label>,
    pub cut: usize,
    pub miss: usize,
    pub better_rc: usize,
}

impl PriorityStructure {
    fn new(mode: DominanceMode) -> Self {
        PriorityStructure {
            queue: BinaryHeap::new(),
            visited: HashMap::new(),
            mode,
            labels: Vec::new(),
            depot_labels: Vec::new(),
            cut: 0,
            miss: 0,
            better_rc: 0,
        }
    }

    fn add_label(&mut self, node_id: usize, reduced_cost: f64, duration: usize, predecessor: usize, cost: usize, covered: Vec<usize>, unfinished: usize) {  
        let label = Label::new(self.labels.len(), reduced_cost, duration, predecessor, cost, covered, node_id, unfinished);
        self.push(label);
    }

    fn push(&mut self, label: Label) {
        if label.node_id == 0 {
            self.depot_labels.push(label.clone());
        } else {
            let key = Key {
                covered: label.covered.clone(),
                node_id: label.node_id,
            };
            if let Some(labels) = self.visited.get_mut(&key) {
                if labels.iter().any(|l| l.dominates(&label, self.mode)) {
                    self.cut += 1;
                    return; // Skip if a label dominates the new label
                }

                labels.retain(|l| !label.dominates(l, self.mode)); // Remove dominated labels
                self.better_rc += 1;
                labels.push(label.clone()); // Add the new label to the bucket
            } else {
                self.miss += 1;
                self.visited.insert(key, vec![label.clone()]);
                self.queue.push(label.clone());
            }
            self.labels.push(label.clone()); // Store the label in the labels vector
        }
    }

    fn pop(&mut self) -> Option<Label> {
        self.queue.pop()
    }
}

struct BucketGraph {
    buckets: Vec<Vec<Vec<Label>>>,
    depot_bucket: Vec<Label>,
    labels: Vec<Label>,
    bucket_length: usize,
    nodes: NodeContainer,
    mode: DominanceMode,
    cut: usize,
    miss: usize,
    better_rc: usize,
    better_duration: usize,
}


impl BucketGraph {
    fn new(time_limit: usize, smallest_arc_duration: usize, nodes: NodeContainer, mode: DominanceMode) -> Self {
        let num_buckets = time_limit / smallest_arc_duration + 1;
        let buckets: Vec<Vec<Vec<Label>>> = vec![vec![Vec::new(); nodes.nodes.len()]; num_buckets];
        let depot_bucket = vec![];
        let labels = Vec::new();
        BucketGraph { buckets, labels, bucket_length: smallest_arc_duration, depot_bucket, nodes, mode, cut: 0, miss: 0, better_rc: 0, better_duration: 0} //data}
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

    fn add_label(&mut self, node_id: usize, reduced_cost: f64, duration: usize, predecessor: usize, cost: usize, covered: Vec<usize>, unfinished: usize) {  
        let label = Label::new(self.num_labels(), reduced_cost, duration, predecessor, cost, covered, node_id, unfinished);

        let mode = self.mode;
        let bucket = if self.nodes.nodes[node_id].is_depot() {&mut self.depot_bucket} else {self.get_bucket_by_duration(node_id, duration)};
        if bucket.iter().any(|l| l.dominates(&label, mode)) {
            return; // Skip if a label dominates the new label
        }

        bucket.retain(|l| !label.dominates(l, mode)); // Remove dominated labels
        bucket.push(label.clone()); // Add the new label to the bucket

        // if predecessor != 0 {
        //     let from_node = self.nodes.nodes[self.labels[predecessor].node_id];
        //     let to_node = self.nodes.nodes[node_id];
        // }

        self.labels.push(label); // Store the label in the labels vector

    }

    fn print_path(&self, idx: usize, nodes: Vec<Node>) {
        if idx == 0 {
            let curr_label = self.labels[idx].clone();
            println!("Node: {:?}", nodes[curr_label.node_id]);
            println!("Label: {:?}", curr_label);
            // let curr = nodes[curr_label.node_id];
            // print!("At {:?}, {:?} with cost :{:?}\n", curr, self.labels[idx].covered, curr_label.cost);
            // panic!("Reached depot");
            return;
        }

        let curr_label = self.labels[idx].clone();
        println!("Node: {:?}", nodes[curr_label.node_id]);
        println!("Label: {:?}", curr_label);
        // let curr = nodes[curr_label.node_id];

        // print!("At {:?}, {:?} with cost :{:?}\n", curr, self.labels[idx].covered, curr_label.cost);
        self.print_path(self.labels[idx].predecessor, nodes);
    }
}


#[derive(Debug, Clone, PartialEq)]
pub struct Label {
    id: usize,
    pub reduced_cost: f64,
    duration: usize,
    predecessor: usize,
    pub cost: usize,
    pub covered: Vec<usize>,
    pub node_id: usize,
    unfinished: usize,
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
    fn new(id: usize, reduced_cost: f64, duration: usize, predecessor: usize, cost: usize, covered: Vec<usize>, node_id: usize, unfinished: usize) -> Self {        Label {
            id,
            reduced_cost,
            duration,
            predecessor,
            cost,
            covered,
            node_id,
            unfinished,
        }
    }

    fn dominates(&self, other: &Label, mode: DominanceMode) -> bool {
        let mut result = true;
        if mode.rc {
            result &= self.reduced_cost <= other.reduced_cost;
        }
        if mode.duration {
            result &= self.duration <= other.duration;
        }
        if mode.coverage {
            result &= self.visits_less_than(other);
        }
        result
    }

    fn visits_less_than(&self, other: &Label) -> bool {
        self.covered.iter().zip(other.covered.iter()).all(|(a, b)| a <= b)
    }
}


pub struct QueuePricer<'a> {
    priority_queue: PriorityStructure,
    nodes: &'a NodeContainer,
    arcs: &'a ArcContainer,
    data: &'a SPDPData,
    cover_rc: &'a Vec<f64>,
    vehicle_rc: &'a Option<f64>,
}

impl<'a> QueuePricer<'a>  {
    pub fn new(nodes: &'a NodeContainer, arcs: &'a ArcContainer, data: &'a SPDPData, cover_rc: &'a Vec<f64>, vehicle_rc: &'a Option<f64>, mode: DominanceMode) -> Self {
        let priority_queue = PriorityStructure::new(mode);

        QueuePricer {
            priority_queue,
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

            let next = arc.end;

            if new_duration > self.data.t_limit {
                continue 'arc_iteration; // Skip if the new duration exceeds the time limit
            }

            let serviced = match arc.done.len() {
                0 => vec![],
                1 => vec![arc.done.left()],
                _ => vec![arc.done.left(), arc.done.right()],
            };

            let mut unfinished = label.unfinished;

            if self.vehicle_rc.is_some() {
                if unfinished == self.data.num_requests {
                    if !next.is_pickup() && !next.is_depot() {
                        if next.to_treat.is_none() {
                            assert!(serviced.len() == 2);
                            if self.data.requests[serviced[0]].from_id == next.to_empty.unwrap() {
                                unfinished = serviced[0];
                            } else {
                                unfinished = serviced[1];
                            }
                        } else {
                            // Have a treat and deliver
                            if self.data.requests[serviced[0]].from_id == next.to_empty.unwrap()
                                && self.data.requests[serviced[0]].to_id == next.to_treat.unwrap() {
                                unfinished = serviced[0];
                            } else {
                                unfinished = serviced[1];
                            }
                        }
                    }

                } else {
                    // Had something onboard
                    assert!(serviced.len() == 1);

                    // Do we still have the same thing on board?
                    if next.is_pickup() {
                        unfinished = self.data.num_requests;
                    }

                    else {
                        let old_to_deliver = self.data.requests[unfinished].from_id;
                        if next.to_empty.unwrap() != old_to_deliver {
                            unfinished = serviced[0];
                        }
                    }
                }
            }

            // if self.vehicle_rc.is_some() {
            //     let valid = vec![
            //         vec![0, 1],
            //         vec![3, 4],
            //         vec![2, 6],
            //         vec![5]
            //     ];
                
            //     if !valid.iter().any(|v| {
            //         v.iter().zip(serviced.iter()).all(|(a, b)| a == b)
            //     }) {
            //         continue 'arc_iteration; // Skip if the serviced requests do not match the valid combinations
            //     }
            // }

            let mut new_covered = label.covered.clone();

            for request_id in serviced.iter() {
                new_covered[*request_id] += 1;
                if new_covered[*request_id] > self.data.requests[*request_id].quantity {
                    continue 'arc_iteration; // Skip if the request is already serviced
                }
            }

            let mut new_reduced_cost = (if self.vehicle_rc.is_some() {new_cost as f64 } else {1.0})
                - new_covered.iter().enumerate().map(|(r_id, q)| *q as f64 * self.cover_rc[r_id]).sum::<f64>();


            if unfinished != self.data.num_requests {
                // new_reduced_cost += self.cover_rc[unfinished];
            }

            
            self.priority_queue.add_label(next_node_id, new_reduced_cost, new_duration, label.id, new_cost, new_covered, unfinished);
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
            self.priority_queue.add_label(arc.end.id, new_reduced_cost, arc.time, 0, arc.cost, covered, self.data.num_requests);
        }

        while let Some(curr) = self.priority_queue.pop() {
            let raw_ptr: *mut QueuePricer = self;  
            unsafe {
                if let Some(pricer_ref) = raw_ptr.as_mut() {
                    pricer_ref.extend_label(&curr);
                }
            }
        }
        self.priority_queue.depot_labels.sort_by(|a, b| a.reduced_cost.partial_cmp(&b.reduced_cost).unwrap());
    }

    pub fn solve_pricing_problem(&mut self, k: usize) -> Vec<Label> {
        self.forward_pass();

        let mut labels = Vec::new();

        for idx in 0..min(k, self.priority_queue.depot_labels.len()) {
            let label = &self.priority_queue.depot_labels[idx];

            // let mut expected_rc = 0.0;
            // for (r_id, q) in label.covered.iter().enumerate() {
            //     expected_rc -= self.cover_rc[r_id] * (*q as f64);
            // } 
            // if self.vehicle_rc.is_some() {
            //     expected_rc -= self.vehicle_rc.unwrap();
            // }
            // if self.vehicle_rc.is_some() {
            //     expected_rc += label.cost as f64;
            // } else {
            //     expected_rc += 1.0;
            // }

            // print!("Expected reduced cost: {} vs. Actual reduced cost: {}", expected_rc, label.reduced_cost);

            // assert!(expected_rc == label.reduced_cost);


            if label.reduced_cost < -1e-6 {
                labels.push(label.clone());
                // if label.covered.iter().sum::<usize>() == self.data.num_requests {
                //     self.bucket_graph.print_path(label.id, self.nodes.nodes.clone());
                // }
            }   
        }
        println!("Labels: {}", self.priority_queue.labels.len());
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

            let next = arc.end;

            if new_duration > self.data.t_limit {
                continue 'arc_iteration; // Skip if the new duration exceeds the time limit
            }

            let serviced = match arc.done.len() {
                0 => vec![],
                1 => vec![arc.done.left()],
                _ => vec![arc.done.left(), arc.done.right()],
            };

            let mut unfinished = label.unfinished;

            if self.vehicle_rc.is_some() {
                if unfinished == self.data.num_requests {
                    if !next.is_pickup() && !next.is_depot() {
                        if next.to_treat.is_none() {
                            assert!(serviced.len() == 2);
                            if self.data.requests[serviced[0]].from_id == next.to_empty.unwrap() {
                                unfinished = serviced[0];
                            } else {
                                unfinished = serviced[1];
                            }
                        } else {
                            // Have a treat and deliver
                            if self.data.requests[serviced[0]].from_id == next.to_empty.unwrap()
                                && self.data.requests[serviced[0]].to_id == next.to_treat.unwrap() {
                                unfinished = serviced[0];
                            } else {
                                unfinished = serviced[1];
                            }
                        }
                    }

                } else {
                    // Had something onboard
                    assert!(serviced.len() == 1);

                    // Do we still have the same thing on board?
                    if next.is_pickup() {
                        unfinished = self.data.num_requests;
                    }

                    else {
                        let old_to_deliver = self.data.requests[unfinished].from_id;
                        if next.to_empty.unwrap() != old_to_deliver {
                            unfinished = serviced[0];
                        }
                    }
                }
            }

            // if self.vehicle_rc.is_some() {
            //     let valid = vec![
            //         vec![0, 1],
            //         vec![3, 4],
            //         vec![2, 6],
            //         vec![5]
            //     ];
                
            //     if !valid.iter().any(|v| {
            //         v.iter().zip(serviced.iter()).all(|(a, b)| a == b)
            //     }) {
            //         continue 'arc_iteration; // Skip if the serviced requests do not match the valid combinations
            //     }
            // }

            let mut new_covered = label.covered.clone();

            for request_id in serviced.iter() {
                new_covered[*request_id] += 1;
                if new_covered[*request_id] > self.data.requests[*request_id].quantity {
                    continue 'arc_iteration; // Skip if the request is already serviced
                }
            }

            let mut new_reduced_cost = (if self.vehicle_rc.is_some() {new_cost as f64 } else {1.0})
                - new_covered.iter().enumerate().map(|(r_id, q)| *q as f64 * self.cover_rc[r_id]).sum::<f64>();


            if unfinished != self.data.num_requests {
                // new_reduced_cost += self.cover_rc[unfinished];
            }

            self.bucket_graph.add_label(next_node_id, new_reduced_cost, new_duration, label.id, new_cost, new_covered, unfinished);
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
            self.bucket_graph.add_label(arc.end.id, new_reduced_cost, arc.time, 0, arc.cost, covered, self.data.num_requests);
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

    pub fn solve_pricing_problem(&mut self, k: usize) -> Vec<Label> {
        self.forward_pass();

        let mut labels = Vec::new();

        for idx in 0..min(k, self.bucket_graph.depot_bucket.len()) {
            let label = &self.bucket_graph.depot_bucket[idx];

            // let mut expected_rc = 0.0;
            // for (r_id, q) in label.covered.iter().enumerate() {
            //     expected_rc -= self.cover_rc[r_id] * (*q as f64);
            // } 
            // if self.vehicle_rc.is_some() {
            //     expected_rc -= self.vehicle_rc.unwrap();
            // }
            // if self.vehicle_rc.is_some() {
            //     expected_rc += label.cost as f64;
            // } else {
            //     expected_rc += 1.0;
            // }

            // print!("Expected reduced cost: {} vs. Actual reduced cost: {}", expected_rc, label.reduced_cost);

            // assert!(expected_rc == label.reduced_cost);


            if label.reduced_cost < -1e-6 {
                labels.push(label.clone());
                // if label.covered.iter().sum::<usize>() == self.data.num_requests {
                //     self.bucket_graph.print_path(label.id, self.nodes.nodes.clone());
                // }
            }   
        }

        labels
    }
}