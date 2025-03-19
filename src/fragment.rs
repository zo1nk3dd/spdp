use std::collections::HashSet;

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

    fn generate_nodes(self) -> HashSet<Node> {
        let depot = Node { 
            location: None,
            to_treat: None,
            to_empty: None,
        };

        let p_locs: HashSet<usize> = self.data.requests.iter()
            .map(|x| x.from_id)
            .collect::<HashSet<usize>>();

        let empty_state_nodes: HashSet<Node> = p_locs.iter()
            .map(|l| Node {
                location: Some(*l),
                to_treat: None,
                to_empty: None,
            })
            .collect();

        let deliver_nodes: HashSet<Node> = p_locs.iter()
            .map(|l| {
                p_locs.iter()
                    .filter_map(|l2| {
                        if *l != *l2 {
                            Some(Node {
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

        let treat_nodes: HashSet<Node> = p_locs.iter()
            .map(|l| {
                self.data.requests.iter()
                    .filter_map(|r| {
                        if *l != r.from_id {
                            Some(Node {
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
        
        let mut nodes = HashSet::new();

        nodes.extend(empty_state_nodes);
        nodes.extend(deliver_nodes);
        nodes.extend(treat_nodes);
        nodes.insert(depot);

        nodes
    }

    pub fn generate_naive_fragments(self) -> Vec<Fragment> {
        let mut tree: Vec<(State, usize)> = Vec::new();
        let root = State {
            event: Event { request_id: 0, action: Action::Pickup },
            time: 0,
            cost: 0,
            treat: Locset::new(),
            empty: Locset::new(),
            done: Locset::new(),
        };

        tree.push((root, 0)); // Placeholder for root node, ends the recursion when generating fragment paths

        let mut fragments: Vec<Fragment> = Vec::new();

        for request_id in 0..self.data.requests.len() {
            let event = Event { request_id, action: Action::Pickup };
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
            let next_event = Event { request_id: i, action: Action::Treat };
            let new_time = time + self.time_between(&last_event, &next_event);
            let new_cost = cost + self.cost_between(&last_event, &next_event);
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
                        self.data.requests[last_event.request_id].to_id == self.data.requests[i].to_id {
                    return true;
                }
            }
        }

        for i in to_empty.iter() {
            let next_event = Event { request_id: i, action: Action::Deliver };
            let new_time = time + self.time_between(&last_event, &next_event);
            let new_cost = cost + self.cost_between(&last_event, &next_event);
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
                        self.data.requests[last_event.request_id].from_id == self.data.requests[i].from_id {
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
                            last_event.request_id > request_id &&
                            self.data.requests[last_event.request_id].from_id == request.from_id {
                        continue;
                    }

                    let next_event = Event { request_id, action: Action::Pickup };
                    let new_time = time + self.time_between(&last_event, &next_event);
                    let new_cost = cost + self.cost_between(&last_event, &next_event);
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

    fn time_between(&self, a: &Event, b: &Event) -> usize {
        let a_loc = match a.action {
            Action::Pickup => self.data.requests[a.request_id].from_id,
            Action::Treat => self.data.requests[a.request_id].to_id,
            Action::Deliver => self.data.requests[a.request_id].from_id,
            Action::_PP => self.data.requests[a.request_id].from_id,
        };
        let b_loc = match b.action {
            Action::Pickup => self.data.requests[b.request_id].from_id,
            Action::Treat => self.data.requests[b.request_id].to_id,
            Action::Deliver => self.data.requests[b.request_id].from_id,
            Action::_PP => self.data.requests[b.request_id].from_id,
        };
        self.data.time[a_loc][b_loc]
    }

    // pub fn generate_routes(&self) -> Vec<Fragment> {
    //     // Use the fragment struct to store valid routes. Similar to the other code
    //     // A route is any path through the network with state nodes and fragment arcs
    //     // Can use DFS to find these
    // }

    fn cost_between(&self, a: &Event, b: &Event) -> usize {
        let a_loc = match a.action {
            Action::Pickup => self.data.requests[a.request_id].from_id,
            Action::Treat => self.data.requests[a.request_id].to_id,
            Action::Deliver => self.data.requests[a.request_id].from_id,
            Action::_PP => self.data.requests[a.request_id].from_id,
        };
        let b_loc = match b.action {
            Action::Pickup => self.data.requests[b.request_id].from_id,
            Action::Treat => self.data.requests[b.request_id].to_id,
            Action::Deliver => self.data.requests[b.request_id].from_id,
            Action::_PP => self.data.requests[b.request_id].from_id,
        };
        self.data.distance[a_loc][b_loc]
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


// Short tests to confirm the data loading is functional
#[cfg(test)]

mod tests {
    use super::*;
    
    #[test]

    fn test_fragment() {
        let spdp = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_A1.dat");
        let fragments = Generator::new(spdp).generate_naive_fragments();
        assert_eq!(fragments.len(), 145);

        let spdp = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_D20.dat");
        let fragments = Generator::new(spdp).generate_naive_fragments();
        assert_eq!(fragments.len(), 130444);       

        let spdp = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_C6.dat");
        let fragments = Generator::new(spdp).generate_naive_fragments();
        assert_eq!(fragments.len(), 1920);
    }

    #[test]
    fn test_node_generation() {
        let spdp = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_A1.dat");
        let nodes = Generator::new(spdp).generate_nodes();
        assert_eq!(nodes.len(), 20);

        let spdp = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_B8.dat");
        let nodes = Generator::new(spdp).generate_nodes();
        assert_eq!(nodes.len(), 184);

        let spdp = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_C1.dat");
        let nodes = Generator::new(spdp).generate_nodes();
        assert_eq!(nodes.len(), 67);
    }
}