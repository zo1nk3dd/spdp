use std::cmp::min;
use std::num::ParseIntError;

use crate::fragment::{NodeContainer, ArcContainer};
use crate::utils::{Node, SPDPData};

struct BucketGraph {
    buckets: Vec<Vec<Vec<Label>>>,
    depot_bucket: Vec<Label>,
    labels: Vec<Label>,
    bucket_length: usize,
    // data: SPDPData,
}


impl BucketGraph {
    fn new(time_limit: usize, smallest_arc_duration: usize, num_nodes: usize) -> Self {
        let num_buckets = time_limit / smallest_arc_duration + 1;
        let buckets: Vec<Vec<Vec<Label>>> = vec![vec![Vec::new(); num_nodes]; num_buckets];
        let depot_bucket = vec![];
        let labels = Vec::new();
        BucketGraph { buckets, labels, bucket_length: smallest_arc_duration, depot_bucket} //data}
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
        let label = Label::new(self.num_labels(), reduced_cost, duration, predecessor, cost, covered);
        let bucket = if node_id == 0 {&mut self.depot_bucket} else {self.get_bucket_by_duration(node_id, duration)};
        if bucket.iter().any(|l| l.dominates(&label)) {
            return; // Skip if a label dominates the new label
        }
        bucket.retain(|l| !label.dominates(l)); // Remove dominated labels
        bucket.push(label.clone()); // Add the new label to the bucket
    }
}


#[derive(Debug, Clone)]
pub struct Label {
    id: usize,
    reduced_cost: f64,
    duration: usize,
    predecessor: usize,
    pub cost: usize,
    pub covered: Vec<usize>,
}

impl Label {
    fn new(id: usize, reduced_cost: f64, duration: usize, predecessor: usize, cost: usize, covered: Vec<usize>) -> Self {
        Label {
            id,
            reduced_cost,
            duration,
            predecessor,
            cost,
            covered,
        }
    }

    fn dominates(&self, other: &Label) -> bool {
        self.reduced_cost <= other.reduced_cost && self.duration <= other.duration
    }
}

pub struct Pricer<'a> {
    bucket_graph: BucketGraph,
    nodes: &'a NodeContainer,
    arcs: &'a ArcContainer,
    data: &'a SPDPData,
    cover_rc: &'a Vec<f64>,
}

impl<'a> Pricer<'a>  {
    pub fn new(nodes: &'a NodeContainer, arcs: &'a ArcContainer, data: &'a SPDPData, cover_rc: &'a Vec<f64>) -> Self {
        let bucket_graph = BucketGraph::new(data.t_limit, arcs.min_fragment_length, nodes.nodes.len());

        Pricer {
            bucket_graph,
            nodes,
            arcs,
            data,
            cover_rc,
        }
    }

    fn extend_label(&mut self, label: &Label, curr: &Node) {
        let forward_arcs = self.arcs.arcs_from.get(curr).unwrap();

        for arc in forward_arcs {
            let next_node_id = arc.end.id;
            let new_duration = label.duration + arc.time;
            let new_cost = label.cost + arc.cost;

            if new_duration > self.data.t_limit {
                continue; // Skip if the new duration exceeds the time limit
            }

            let serviced = match arc.done.len() {
                0 => vec![],
                1 => vec![arc.done.left()],
                _ => vec![arc.done.left(), arc.done.right()],
            };

            let new_reduced_cost = label.reduced_cost - serviced.iter().map(|r_id| self.cover_rc[*r_id]).sum::<f64>();

            
            let mut new_serviced = label.covered.clone();
            new_serviced.extend(serviced);

            self.bucket_graph.add_label(next_node_id, new_reduced_cost, new_duration, curr.id, new_cost, new_serviced);
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
            assert!(arc.time == 0);
            let new_reduced_cost = 1.0;
            self.bucket_graph.add_label(arc.end.id, new_reduced_cost, arc.time, 0, arc.cost, serviced);
        }


        for bucket_idx in 0..self.bucket_graph.num_buckets() {
            for node_idx in 0..self.nodes.nodes.len() {
                let curr = self.nodes.nodes[node_idx];
                let raw_ptr: *mut Pricer = self;
                for label in self.bucket_graph.buckets[bucket_idx][node_idx].iter() {                    
                    unsafe {
                        if let Some(pricer_ref) = raw_ptr.as_mut() {
                            pricer_ref.extend_label(label, &curr);
                        }
                    }
                }
            }
        }

        self.bucket_graph.depot_bucket.sort_by(|a, b| a.reduced_cost.partial_cmp(&b.reduced_cost).unwrap());
    }

    pub fn solve_pricing_problem(&mut self, k: usize) -> Vec<Label> {
        self.forward_pass();

        let mut labels = Vec::new();

        for idx in 0..min(k, self.bucket_graph.depot_bucket.len()) {
            let label = &self.bucket_graph.depot_bucket[idx];
            if label.reduced_cost < 0.0 {
                labels.push(label.clone()) // Return the first negative reduced cost label
            }
        }

        labels
    }

}

