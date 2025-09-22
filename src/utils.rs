use std::{fmt, fs};
use crate::{coverset::CoverSet, pricing::DominanceMode};

use super::locset::Locset;
use super::constants::EPS;

#[derive(Debug, Clone)]
pub struct SPDPData {
    pub container_types: usize,
    pub waste_types: usize,
    pub locations: usize,
    pub fixed_vehicle_cost: usize,
    pub t_pickup: usize,
    pub t_empty: usize,
    pub t_delivery: usize,
    pub t_limit: usize,
    pub num_requests: usize,
    pub requests: Vec<Request>,
    pub distance: Vec<Vec<usize>>,
    pub time: Vec<Vec<usize>>, 
}

impl SPDPData {
    pub fn from_file(filename: &str) -> Self {
        let result = fs::read_to_string(filename);


        let file = match result {
            Ok(file) => file,
            Err(_) => panic!("Error reading file: {}. Probably due to invalid instance provided.", filename),
        };

        let mut lines = file.lines();

        lines.next();
        lines.next();
        lines.next();
        lines.next();
        lines.next();

        let (name, num) = lines.next().unwrap().split_once("\t").unwrap();
        assert_eq!(name, "CONTAINER_TYPES");
        let container_types: usize = num.trim().parse().unwrap();

        let (name, num) = lines.next().unwrap().split_once("\t").unwrap();
        assert_eq!(name, "WASTE_TYPES");
        let waste_types: usize = num.trim().parse().unwrap();

        let (_name, num) = lines.next().unwrap().split_once("\t").unwrap();
        // assert_eq!(name, "LOCATIONS"); // This is a bug in the data file
        let locations: usize = num.trim().parse().unwrap();

        let (name, num) = lines.next().unwrap().split_once("\t").unwrap();
        assert_eq!(name, "FixedVehicleCost");
        let fixed_vehicle_cost: usize = num.trim().parse().unwrap();

        let (name, num) = lines.next().unwrap().split_once("\t").unwrap();
        assert_eq!(name, "TimePickUp");
        let t_pickup: usize = num.trim().parse().unwrap();

        let (name, num) = lines.next().unwrap().split_once("\t").unwrap();
        assert_eq!(name, "TimeEmpty");
        let t_empty: usize = num.trim().parse().unwrap();

        let (name, num) = lines.next().unwrap().split_once("\t").unwrap();
        assert_eq!(name, "TimeDelivery");
        let t_delivery: usize = num.trim().parse().unwrap();

        let (name, num) = lines.next().unwrap().split_once("\t").unwrap();
        assert_eq!(name, "TimeLimit");
        let t_limit: usize = num.trim().parse().unwrap();

        let (name, num) = lines.next().unwrap().split_once("\t").unwrap();
        assert_eq!(name, "REQUESTS");
        let num_requests: usize = num.trim().parse().unwrap();

        lines.next();

        let mut requests: Vec<Request> = (0..num_requests)
            .map(|_| {
                let data = lines.next().unwrap();
                Request::from_line(data)
                }
            ).collect();

        println!("Requests: {:?}", requests.len());

        let mut new_requests = Vec::new();

        for r in &requests {
            if new_requests.iter().any(|x: &Request| x.from_id == r.from_id && x.to_id == r.to_id) {
                continue;
            }
            let mut count = 0;

            for r2 in &requests {
                if r.from_id == r2.from_id && r.to_id == r2.to_id {
                    count += 1;
                }
            }

            let mut new_request = r.clone();
            new_request.quantity = count;

            new_requests.push(new_request);
        }    

        println!("New Requests: {:?}", new_requests.len());
        
        requests = new_requests;
        requests.sort_by(|a, b| a.quantity.cmp(&b.quantity));

        assert!(lines.next().unwrap().starts_with("Distance"));

        let distance: Vec<Vec<usize>> = (0..locations)
            .map(|_| {
                let data = lines.next().unwrap();
                data.split('\t').into_iter().map(|x| x.parse().unwrap()).collect()
            }).collect();

        assert!(lines.next().unwrap().starts_with("Time"));

        let time: Vec<Vec<usize>> = (0..locations)
            .map(|_| {
                let data = lines.next().unwrap();
                data.split('\t').into_iter().map(|x| x.parse().unwrap()).collect()
            }).collect();

        SPDPData {
            container_types,
            waste_types,
            locations,
            fixed_vehicle_cost,
            t_pickup,
            t_empty,
            t_delivery,
            t_limit,
            num_requests: requests.len(),
            requests,
            distance,
            time,
        }
    }

    
    pub fn time_between(&self, a: &Event, b: &Event) -> usize {
        if b.request_id.is_none() {
            return 0;
        }
        let a_loc = match a.action {
            Action::Pickup => self.requests[a.request_id.unwrap()].from_id,
            Action::Treat => self.requests[a.request_id.unwrap()].to_id,
            Action::Deliver => self.requests[a.request_id.unwrap()].from_id,
            Action::PP => a.request_id.unwrap(),
        };
        let b_loc = match b.action {
            Action::Pickup => self.requests[b.request_id.unwrap()].from_id,
            Action::Treat => self.requests[b.request_id.unwrap()].to_id,
            Action::Deliver => self.requests[b.request_id.unwrap()].from_id,
            Action::PP => b.request_id.unwrap(),
        };
        self.time[a_loc][b_loc]
    }

    // pub fn generate_routes(&self) -> Vec<Fragment> {
    //     // Use the fragment struct to store valid routes. Similar to the other code
    //     // A route is any path through the network with state nodes and fragment arcs
    //     // Can use DFS to find these
    // }

    pub fn cost_between(&self, a: &Event, b: &Event) -> usize {
        if b.request_id.is_none() {
            return 0;
        }
        let a_loc = match a.action {
            Action::Pickup => self.requests[a.request_id.unwrap()].from_id,
            Action::Treat => self.requests[a.request_id.unwrap()].to_id,
            Action::Deliver => self.requests[a.request_id.unwrap()].from_id,
            Action::PP => a.request_id.unwrap(),
        };
        let b_loc = match b.action {
            Action::Pickup => self.requests[b.request_id.unwrap()].from_id,
            Action::Treat => self.requests[b.request_id.unwrap()].to_id,
            Action::Deliver => self.requests[b.request_id.unwrap()].from_id,
            Action::PP => b.request_id.unwrap(),
        };
        self.distance[a_loc][b_loc]
    }
}

#[derive(Debug, Clone)]
pub struct Request {
    _id: usize,
    _waste_id: usize,
    pub from_id: usize,
    _container_type: usize,
    _to_num: usize,
    pub to_id: usize,
    pub quantity: usize,
}

impl Request {
    pub fn from_line(line: &str) -> Self {
        let mut data = line.split('\t').into_iter();

        let id = data.next().unwrap().parse().unwrap();
        let waste_id = data.next().unwrap().parse().unwrap();
        let from_id = data.next().unwrap().parse().unwrap();
        let container_type = data.next().unwrap().parse().unwrap();
        let to_num = data.next().unwrap().parse().unwrap();
        let to_id = data.next().unwrap().parse().unwrap();

        Request {
            _id: id,
            _waste_id: waste_id,
            from_id,
            _container_type: container_type,
            _to_num: to_num,
            to_id,
            quantity: 1,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct State {
    pub event: Event,
    pub time: usize,
    pub cost: usize,
    pub treat: Locset,
    pub empty: Locset,
    pub done: Locset,
}

#[derive(Debug, Clone, Copy)]
pub struct Event {
    pub request_id: Option<usize>,
    pub action: Action,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Pickup,
    Treat,
    Deliver,
    PP
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Node {
    pub id: usize,
    pub location: Option<usize>,
    pub to_treat: Option<usize>,
    pub to_empty: Option<usize>,
}

impl Node {
    pub fn is_depot(&self) -> bool {
        self.location.is_none()
    }

    pub fn is_pickup(&self) -> bool {
        self.to_treat.is_none() && self.to_empty.is_none()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Label {
    pub id: usize,
    pub reduced_cost: f64,
    pub duration: usize,
    pub predecessor: Option<usize>,
    pub in_arc: usize,
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
    pub fn new(id: usize, reduced_cost: f64, duration: usize, predecessor: Option<usize>, cost: usize, coverset: CoverSet, node_id: usize, in_arc: usize,) -> Self {    
        Label {
            id,
            reduced_cost,
            duration,
            predecessor,
            cost,
            coverset,
            node_id,
            in_arc,
        }
    }

    pub fn dominates(&self, other: &Label, mode: DominanceMode) -> bool {
        match mode {
            DominanceMode::RC => {
                self.reduced_cost - other.reduced_cost < EPS
            },
            DominanceMode::DurRC => {
                self.reduced_cost - other.reduced_cost < EPS && self.duration <= other.duration
            },
            DominanceMode::DurRCCover => {
                self.reduced_cost - other.reduced_cost < EPS && self.duration <= other.duration && self.visits_less_eq_than(other)
            },
            DominanceMode::Dur => {
                self.duration <= other.duration
            },
            DominanceMode::DurRCQuantity => {
                self.reduced_cost - other.reduced_cost < EPS && self.duration <= other.duration && self.coverset.len <= other.coverset.len
            }
        }
    }

    fn visits_less_eq_than(&self, other: &Label) -> bool {
        self.coverset.visits_leq_than(&other.coverset)
    }
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rc: {:.2}, d: {}, node: {:?}", 
            self.reduced_cost, self.duration, self.node_id)
    }
}

// Short tests to confirm the data loading is functional
#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn test_load_data() {
        let spdp = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_A1.dat");

        assert_eq!(spdp.container_types, 9);
        assert_eq!(spdp.waste_types, 29);
        assert_eq!(spdp.locations, 23);
        assert_eq!(spdp.fixed_vehicle_cost, 500);
        assert_eq!(spdp.t_pickup, 7);
        assert_eq!(spdp.t_empty, 8);
        assert_eq!(spdp.t_delivery, 7);
        assert_eq!(spdp.t_limit, 480);
        assert_eq!(spdp.num_requests, 5);
        assert_eq!(spdp.requests.len(), 5);
        assert_eq!(spdp.distance.len(), 23);
        assert_eq!(spdp.time.len(), 23);
        assert_eq!(spdp.requests[0].from_id, 5);
        assert_eq!(spdp.requests[0].to_id, 14);
    }
}