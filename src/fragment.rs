use std::collections::{HashMap, HashSet};
use std::hash::{Hasher, Hash};

use super::utils::*;
use super::locset::Locset;

pub struct Generator {
    data: SPDPData,
}

impl Generator {
    pub fn new(data: SPDPData) -> Self {
        Generator {
            data,
        }
    }

    // pub fn generate_adj_lists(&self, fragments: &Vec<Fragment>) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    //     let mut outgoing_fragments_from_request_start: Vec<Vec<usize>> = vec![Vec::new(); self.data.locations];
    // }

    pub fn generate_nodes(&self) -> NodeContainer {
        let depot = Node { 
            id: 0,
            location: None,
            to_treat: None,
            to_empty: None,
        };

        let p_locs: HashSet<usize> = self.data.requests.iter()
            .map(|x| x.from_id)
            .collect::<HashSet<usize>>();

        let mut offset = 1;
        let mut empty_state_nodes: Vec<Node> = p_locs.iter().enumerate()
            .map(|(idx, l)| Node {
                id: idx + offset,
                location: Some(*l),
                to_treat: None,
                to_empty: None,
            })
            .collect();

        empty_state_nodes.push(depot.clone());
        
        offset = empty_state_nodes.len();

        let mut deliver_nodes: Vec<Node> = p_locs.iter()
            .map(|l| {
                p_locs.iter()
                    .filter_map(|l2| {
                        if *l != *l2 {
                            Some(Node {
                                id: 0,
                                location: Some(*l),
                                to_treat: None,
                                to_empty: Some(*l2),
                            })
                        } else {
                            None
                        }
                    })
            })
            .flatten()
            .collect();

        deliver_nodes = deliver_nodes.iter()
            .enumerate()
            .map(|(idx, l)| Node {
                id: idx + offset,
                location: l.location,
                to_treat: l.to_treat,
                to_empty: l.to_empty,
            })
            .collect();

        offset = deliver_nodes.len() + empty_state_nodes.len();

        let mut treat_nodes: Vec<Node> = p_locs.iter()
            .map(|l| {
                self.data.requests.iter()
                    .filter_map(|r| {
                        if *l != r.from_id {
                            Some(Node {
                                id: 0, 
                                location: Some(*l),
                                to_treat: Some(r.to_id),
                                to_empty: Some(r.from_id),
                            })
                        } else {
                            None
                        }
                    })
            })
            .flatten()
            .collect();   

        treat_nodes = treat_nodes.iter()
            .enumerate()
            .map(|(idx, l)| Node {
                id: idx + offset,
                location: l.location,
                to_treat: l.to_treat,
                to_empty: l.to_empty,
            })
            .collect();        

        let mut nodes = Vec::new();

        for node in empty_state_nodes.iter() {
            nodes.push(node.clone());
        }
        for node in deliver_nodes.iter() {
            nodes.push(node.clone());
        }
        for node in treat_nodes.iter() {
            nodes.push(node.clone());
        }

        NodeContainer { 
            nodes, 
            pickup_nodes: empty_state_nodes,
            treat_nodes,
            deliver_nodes,
            depot,
        }
    }

    pub fn generate_arcs(&self, nodes: &NodeContainer) -> ArcContainer {
        let fragments = self.generate_fragments();

        let mut arc_container = ArcContainer::new(self.data.clone());

        for f in fragments {
            let mut done = f.done.iter();
            let done_set = DoneSet::new(Some(done.next().unwrap()), done.next());

            for pickup in nodes.pickup_nodes.iter() {
                let mut extended = f.events.clone();
                extended.push(Event { request_id: pickup.location, action: Action::PP });

                let start = nodes.pickup_nodes.iter()
                    .find(|n| n.location == Some(self.data.requests[f.events[0].request_id.unwrap()].from_id))
                    .unwrap();
                
                arc_container.create_arc(
                    *start,
                    *pickup,
                    done_set,
                    extended,      
                );   
            }

            if done_set.len() == 1 {
                if self.data.requests[done_set.left()].quantity >= 2 {
                    let done_set_dup = DoneSet {
                        r1: Some(done_set.left()),
                        r2: Some(done_set.left()),
                    };

                    for pickup in nodes.pickup_nodes.iter() {
                        let mut extended = f.events.clone();
                        extended.push(Event { request_id: pickup.location, action: Action::PP });
                        
                        let start = nodes.pickup_nodes.iter()
                            .find(|n| n.location == Some(self.data.requests[f.events[0].request_id.unwrap()].from_id))
                            .unwrap();

                        arc_container.create_arc(
                            *start,
                            *pickup,
                            done_set_dup,
                            extended,      
                        );   
                    } 
                }
                continue;
            }


            let second_pickup_index = f.events.iter().skip(1) // The first pickup
                .position(|e| e.action == Action::Pickup)
                .unwrap() + 1;

            assert!(second_pickup_index <= 2);

            let late_start_node = nodes.nodes.iter()
                .find(|n| {
                    n.location == Some(self.data.requests[f.events[second_pickup_index].request_id.unwrap()].from_id) &&
                    n.to_treat == (if second_pickup_index == 1 { Some(self.data.requests[f.events[0].request_id.unwrap()].to_id)} else { None }) &&
                    n.to_empty == Some(self.data.requests[f.events[0].request_id.unwrap()].from_id)
                });

            let done_set_no_first_pickup = DoneSet::new(Some(f.events[second_pickup_index].request_id.unwrap()), None);

            // assert!(nodes.nodes.contains(&late_start_node));

            for p in nodes.pickup_nodes.iter() {
                let mut extended: Vec<Event> = f.events.clone()[second_pickup_index..].to_vec();
                extended.push(Event { request_id: p.location, action: Action::PP });

                if late_start_node.is_some() && late_start_node.unwrap().location != late_start_node.unwrap().to_empty {
                    arc_container.create_arc(
                        *late_start_node.unwrap(),
                        *p,
                        done_set_no_first_pickup,
                        extended,
                    );
                }
            }

            let first_delivery_index = f.events.iter() 
                .position(|e| e.action == Action::Deliver)
                .unwrap();

            let onboard_transfer_location = if f.events[first_delivery_index + 1].action == Action::Treat {
                    Some(self.data.requests[f.events[first_delivery_index + 1].request_id.unwrap()].to_id)
                } else {
                    None
                };

            let onboard_deliver_location = Some(self.data.requests[f.events[first_delivery_index + 1].request_id.unwrap()].from_id);

            if onboard_transfer_location.is_some() {
                for pickup in nodes.pickup_nodes.iter() {
                    if pickup.is_depot() {
                        continue;
                    }
                    if onboard_deliver_location != pickup.location {
                        let mut extended = f.events.clone()[..(first_delivery_index+1)].to_vec();
                        extended.push(Event { request_id: pickup.location, action: Action::PP });

                        let early_finish_node = nodes.treat_nodes.iter()
                            .find(|n| n.location == pickup.location && n.to_treat == onboard_transfer_location && n.to_empty == onboard_deliver_location)
                            .unwrap();

                        let start = nodes.pickup_nodes.iter()
                            .find(|n| n.location == Some(self.data.requests[f.events[0].request_id.unwrap()].from_id))
                            .unwrap();

                        arc_container.create_arc(
                            *start,
                            *early_finish_node,
                            done_set,
                            extended,
                        );

                        // Potential issue, no if fnode in nodes but i dont see any node pruning

                        let mut extended = f.events.clone()[(second_pickup_index)..(first_delivery_index+1)].to_vec();
                        extended.push(Event { request_id: pickup.location, action: Action::PP });

                        if late_start_node.is_some() && late_start_node.unwrap().location != late_start_node.unwrap().to_empty {
                            arc_container.create_arc(
                                *late_start_node.unwrap(),
                                *early_finish_node, 
                                done_set_no_first_pickup, 
                                extended);
                        }
                    }
                }
            }

            let only_deliver_left = first_delivery_index + 
                (if onboard_transfer_location.is_some() { 1 } else { 0 });

            for pickup in nodes.pickup_nodes.iter() {
                if pickup.is_depot() {
                    continue;
                }
                if onboard_deliver_location != pickup.location {
                    let mut extended = f.events.clone()[..(only_deliver_left+1)].to_vec();
                    extended.push(Event { request_id: pickup.location, action: Action::PP });

                    let start = nodes.pickup_nodes.iter()
                        .find(|n| n.location == Some(self.data.requests[f.events[0].request_id.unwrap()].from_id))
                        .unwrap();

                    let end = nodes.deliver_nodes.iter()
                        .find(|n| n.location == pickup.location && n.to_empty == onboard_deliver_location)
                        .unwrap();

                    arc_container.create_arc(
                        *start,
                        *end,
                        done_set,
                        extended,
                    );

                    let mut extended = f.events.clone()[(second_pickup_index)..(only_deliver_left+1)].to_vec();
                    extended.push(Event { request_id: pickup.location, action: Action::PP });

                    if late_start_node.is_some() && late_start_node.unwrap().location != late_start_node.unwrap().to_empty {
                        arc_container.create_arc(
                            *late_start_node.unwrap(),
                            *end,
                            done_set_no_first_pickup,
                            extended,
                        );
                    }
                }
            }
        }

        for node in nodes.pickup_nodes.iter() {
            if !node.is_depot() {
                arc_container.create_arc(
                    nodes.depot, 
                    *node, 
                    DoneSet::new(None, None),
                    vec!(),
                );
            }
        }

        arc_container.process_once_populated();
        
        arc_container
    }

    pub fn generate_fragments(&self) -> Vec<Fragment> {
        let mut tree: Vec<(State, usize)> = Vec::new();
        let root = State {
            event: Event { request_id: Some(0), action: Action::Pickup },
            time: 0,
            cost: 0,
            treat: Locset::new(),
            empty: Locset::new(),
            done: Locset::new(),
        };

        tree.push((root, 0)); // Placeholder for root node, ends the recursion when generating fragment paths

        let mut fragments: Vec<Fragment> = Vec::new();

        for request_id in 0..self.data.requests.len() {
            let event = Event { request_id: Some(request_id), action: Action::Pickup };
            let time = self.data.t_pickup + self.data.t_empty + self.data.t_delivery;
            let cost = 0;
            let mut to_treat = Locset::new();
            to_treat.insert(request_id);
            let to_empty = Locset::new();
            let done = Locset::new();

            let state = State {
                event,
                time,
                cost,
                treat: to_treat,
                empty: to_empty,
                done,
            };

            tree.push((state, 0));
            let next = tree.len() - 1;

            self.generate_fragments_from(&mut fragments, &mut tree, next);
        }

        fragments
    }
    
    fn generate_fragments_from(
        &self,
        fragments: &mut Vec<Fragment>,
        tree: &mut Vec<(State, usize)>, // State and parent index
        curr: usize,
    ) -> bool { 
        let (state, parent_idx) = tree[curr];

        let last_event = state.event;
        let time = state.time;
        let cost = state.cost;
        let to_treat = state.treat;
        let to_empty = state.empty;
        let done = state.done;
        
        if time > self.data.t_limit {
            return false;
        }

        if to_treat.len() + to_empty.len() == 0 {
            let fragment = Fragment::from_tree(tree, last_event, parent_idx, time, cost, done);
            fragments.push(fragment);
            return true;
        }

        let mut found_true = false;

        for i in to_treat.iter() {
            let next_event = Event { request_id: Some(i), action: Action::Treat };
            let new_time = time + self.data.time_between(&last_event, &next_event);
            let new_cost = cost + self.data.cost_between(&last_event, &next_event);
            let mut new_to_treat = to_treat;
            new_to_treat.remove(i);
            let mut new_to_empty = to_empty;
            new_to_empty.insert(i);

            let new_state = State {
                event: next_event,
                time: new_time,
                cost: new_cost,
                treat: new_to_treat,
                empty: new_to_empty,
                done: done,
            };

            tree.push((new_state, curr));
            let next = tree.len() - 1;

            let result = self.generate_fragments_from(fragments, tree, next);

            if result {
                found_true = true;
                // todo add the optimization, how does return true work?
                if last_event.action == Action::Treat &&
                        self.data.requests[last_event.request_id.unwrap()].to_id == self.data.requests[i].to_id {
                    return true;
                }
            }
        }

        for i in to_empty.iter() {
            let next_event = Event { request_id: Some(i), action: Action::Deliver };
            let new_time = time + self.data.time_between(&last_event, &next_event);
            let new_cost = cost + self.data.cost_between(&last_event, &next_event);
            let mut new_to_empty = to_empty;
            new_to_empty.remove(i);
            let mut new_done = done;
            new_done.insert(i);

            let new_state = State {
                event: next_event,
                time: new_time,
                cost: new_cost,
                treat: to_treat,
                empty: new_to_empty,
                done: new_done,
            };
            
            tree.push((new_state, curr));
            let next = tree.len() - 1;

            let result = self.generate_fragments_from(fragments, tree, next);

            if result {
                found_true = true;
                // todo add the optimization, how does return true work?
                // ig it just skips extending to pickup in this scenario by returning
                if last_event.action == Action::Deliver &&
                        self.data.requests[last_event.request_id.unwrap()].from_id == self.data.requests[i].from_id {
                    return true;
                }
            }
        }

        if !found_true {
            return false;
        }
        
        // Extending to the next pickup node
        if to_treat.len() + to_empty.len() + done.len() < 2 {
            let mut total = Locset::new();
            total.union_inplace(&to_treat);
            total.union_inplace(&to_empty);
            total.union_inplace(&done);

            // If there is a request to empty, assign that
            let mut to_empty_location = None;
            if to_empty.len() > 0 {
                to_empty_location = Some(self.data.requests[to_empty.iter().nth(0).unwrap()].from_id);
            }

            for request_id in 0..self.data.requests.len() {
                if !total.contains(&request_id) {
                    let request = &self.data.requests[request_id];
                    if to_empty_location.is_some() && to_empty_location.unwrap() == request.from_id {
                        continue;
                    }

                    if last_event.action == Action::Pickup && 
                            last_event.request_id.unwrap() > request_id &&
                            self.data.requests[last_event.request_id.unwrap()].from_id == request.from_id {
                        continue;
                    }

                    let next_event = Event { request_id: Some(request_id), action: Action::Pickup };
                    let new_time = time + self.data.time_between(&last_event, &next_event);
                    let new_cost = cost + self.data.cost_between(&last_event, &next_event);
                    let mut new_to_treat = to_treat;
                    new_to_treat.insert(request_id);
                    let new_to_empty = to_empty;
                    let new_done = done;

                    let new_state = State {
                        event: next_event,
                        time: new_time,
                        cost: new_cost,
                        treat: new_to_treat,
                        empty: new_to_empty,
                        done: new_done,
                    };

                    tree.push((new_state, curr));
                    let next = tree.len() - 1;

                    self.generate_fragments_from(fragments, tree, next);
                }
            }
        }
        return true;
    }
}



#[derive(Debug, Clone)]
pub struct Fragment {
    pub events: Vec<Event>,
    pub time: usize,
    pub cost: usize,
    pub done: Locset,
}

impl Fragment {
    fn from_tree(tree: &mut Vec<(State, usize)>, last_event: Event, parent_idx: usize, time: usize, cost: usize, done: Locset) -> Self {
        let mut events = vec![last_event];
        let mut curr = parent_idx;
        while curr != 0 {
            let (state, parent) = tree[curr];
            events.push(state.event);
            curr = parent;
        }
        events.reverse();

        Fragment {
            events,
            time,
            cost,
            done,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Arc {
    pub id: usize,
    pub start: Node,
    pub end: Node,
    pub done: DoneSet,
    pub time: usize,
    pub cost: usize,
    pub path: Vec<Event>,
}

impl Arc {
    fn dominates(&self, other: &Arc) -> bool {
        if (self.cost <= other.cost && self.time <= other.time) == true {
            // println!("{:?} dominates {:?}", self, other);
            return true;
        }
        false
    }
}

impl Hash for Arc {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.start.hash(state);
        self.end.hash(state);
        self.done.hash(state);
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct DoneSet {
    r1: Option<usize>,
    r2: Option<usize>,
}

impl DoneSet{
    fn new(r1: Option<usize>, r2: Option<usize>) -> Self {
        DoneSet {
            r1,
            r2,
        }
    }

    pub fn len(&self) -> usize {
        if self.r2 == None {
            if self.r1 == None {
                0
            } else {
                1
            }
        } else {
            2
        }
    }

    pub fn left(&self) -> usize {
        if self.r1.is_none() {
            panic!("No left element in ArcDone")
        } else {
            self.r1.unwrap()
        }
    }

    pub fn right(&self) -> usize {
        if self.r2.is_none() {
            panic!("No right element in ArcDone")
        } else {
            self.r2.unwrap()
        }
    }
}

pub struct NodeContainer {
    pub nodes: Vec<Node>,
    pub pickup_nodes: Vec<Node>,
    pub treat_nodes: Vec<Node>,
    pub deliver_nodes: Vec<Node>,
    pub depot: Node,
}

pub struct ArcContainer {
    pub data: SPDPData,
    pub container: HashMap<(Node, Node, DoneSet), Vec<Arc>>,
    pub arcs: Vec<Arc>,
    pub arcs_from: HashMap<Node, Vec<Arc>>,
    pub arcs_to: HashMap<Node, Vec<Arc>>,
    pub min_fragment_length: usize,
}

impl ArcContainer {
    fn new(data: SPDPData) -> Self {
        let min_fragment_length = data.t_limit;
        ArcContainer {
            data,
            container: HashMap::new(),
            arcs: Vec::new(),
            arcs_from: HashMap::new(),
            arcs_to: HashMap::new(),
            min_fragment_length,
        }
    }

    pub fn num_keys(&self) -> usize {
        self.container.len()
    }

    pub fn num_arcs(&self) -> usize {
        self.arcs.len()
    }
    
    fn create_arc(&mut self, start: Node, end: Node, done: DoneSet, path: Vec<Event>) {
        let mut time = path.iter()
            .zip(path.iter().skip(1))
            .map(|(a, b)| self.data.time_between(a, b))
            .sum();

        let mut cost = path.iter()
            .zip(path.iter().skip(1))
            .map(|(a, b)| self.data.cost_between(a, b))
            .sum();

        if start.is_depot() {
            cost += self.data.fixed_vehicle_cost;
        }

        if done.len() == 2 && done.left() == done.right() {
            time += self.data.t_pickup + self.data.t_empty + self.data.t_delivery;
        }

        if start.is_depot() && end.is_depot() {
            time = self.data.t_limit;
        }

        let arc = Arc {
            id: 0,
            start,
            end,
            done,
            time,
            cost,
            path,
        };

        let key = (start, end, done);
        if self.container.contains_key(&key) {
            if !(self.container.get_mut(&key).unwrap().iter().any(|a| a.dominates(&arc))) {
                let mut new_arcs: Vec<Arc> = self.container.get(&key).unwrap().iter().filter_map(|a| {
                    if !arc.dominates(a) {
                        return Some(a.clone());
                    }
                    return None;
                })
                .collect();
                new_arcs.push(arc.clone());
                if arc.time < self.min_fragment_length && arc.time != 0 {
                    self.min_fragment_length = arc.time;
                }
                self.container.insert(key, new_arcs);
            }
        } else {
            if arc.time < self.min_fragment_length && arc.time != 0 {
                self.min_fragment_length = arc.time;
            }
            self.container.insert(key, vec![arc.clone()]);

        }
    }

    fn process_once_populated(&mut self) {
        for (id, arc) in self.container.values().flatten().enumerate() {
            let mut new_arc = arc.clone();
            new_arc.id = id;
            self.arcs.push(new_arc.clone());

            if self.arcs_from.contains_key(&arc.start) {
                self.arcs_from.get_mut(&arc.start).unwrap().push(new_arc.clone());
            } else {
                self.arcs_from.insert(arc.start, vec![new_arc.clone()]);
            }

            if self.arcs_to.contains_key(&arc.end) {
                self.arcs_to.get_mut(&arc.end).unwrap().push(new_arc.clone());
            } else {
                self.arcs_to.insert(arc.end, vec![new_arc.clone()]);
            }
        }
    }
}


// Short tests to confirm the data loading is functional
#[cfg(test)]

mod tests {
    use super::*;
    
    #[test]

    fn test_fragment() {
        let spdp = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_A1.dat");
        let fragments = Generator::new(spdp).generate_fragments();
        assert_eq!(fragments.len(), 145);

        let spdp = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_D20.dat");
        let fragments = Generator::new(spdp).generate_fragments();
        assert_eq!(fragments.len(), 130444);       

        let spdp = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_C6.dat");
        let fragments = Generator::new(spdp).generate_fragments();
        assert_eq!(fragments.len(), 1920);
    }

    #[test]
    fn test_node_generation() {
        let spdp = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_A1.dat");
        let nodes = Generator::new(spdp).generate_nodes();
        assert_eq!(nodes.nodes.len(), 20);

        let spdp = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_B8.dat");
        let nodes = Generator::new(spdp).generate_nodes();
        assert_eq!(nodes.nodes.len(), 184);

        let spdp = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_C1.dat");
        let nodes = Generator::new(spdp).generate_nodes();
        assert_eq!(nodes.nodes.len(), 67);
    }

    #[test]
    fn test_arc_generation() {
        let spdp = SPDPData::from_file("./SkipData/Testing/small.dat");
        let gen = Generator::new(spdp);
        let nodes = gen.generate_nodes();
        let arccontainer = gen.generate_arcs(&nodes);
        assert_eq!(arccontainer.num_keys(), 48);

        let spdp = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_A1.dat");
        let gen = Generator::new(spdp);
        let nodes = gen.generate_nodes();
        let arccontainer = gen.generate_arcs(&nodes);
        assert_eq!(arccontainer.num_arcs(), 579);

        let spdp = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_B8.dat");
        let gen = Generator::new(spdp);
        let nodes = gen.generate_nodes();
        let arccontainer = gen.generate_arcs(&nodes);
        assert_eq!(arccontainer.num_arcs(), 22609);

        let spdp = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_C10.dat");
        let gen = Generator::new(spdp);
        let nodes = gen.generate_nodes();
        let arccontainer = gen.generate_arcs(&nodes);
        assert_eq!(arccontainer.num_arcs(), 59019);
    }
}