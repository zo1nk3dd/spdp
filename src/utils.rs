use std::fs;
use super::locset::Locset;

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
        let file = fs::read_to_string(filename).unwrap();

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
            num_requests,
            requests,
            distance,
            time,
        }
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
    pub request_id: usize,
    pub action: Action,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Pickup,
    Treat,
    Deliver,
    _PP
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Node {
    pub location: Option<usize>,
    pub to_treat: Option<usize>,
    pub to_empty: Option<usize>,
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