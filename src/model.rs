use std::fs;

struct ModelRestricted {
    pub data : SPDPData,
    fragments: Vec<Fragment>,
}

impl ModelRestricted {

    pub fn new(data: SPDPData) -> Self {
        let mut model = ModelRestricted {
            data,
            fragments: Vec::new(),
        };

        for request_id in 0..model.data.requests.len() {
            let request = &model.data.requests[request_id];
            let path = vec![Event { request_id, action: Action::Pickup }];
            let time = model.data.t_pickup + model.data.t_empty + model.data.t_delivery;
            let cost = 0;
            let to_treat = vec![request_id];
            let to_empty = Vec::new();
            let done = Vec::new();
            let num_p = 1;

            model.generate_fragments(path, time, cost, to_treat, to_empty, done, num_p);
        }

        model
    }

    fn generate_fragments(
        &mut self,
        path: Vec<Event>,
        time: usize,
        cost: usize,
        to_treat: Vec<usize>,
        to_empty: Vec<usize>,
        done: Vec<usize>,
        num_p: usize,
    ) -> bool { 
        if time > self.data.t_limit {
            return false;
        }

        if to_treat.len() + to_empty.len() == 0 {
            self.fragments.push(Fragment::new(path, time, cost, done));
            return true;
        }

        let mut found_true = false;

        for i in &to_treat {
            let next_event = Event { request_id: *i, action: Action::Treat };
            let new_time = time + self.time_between(path.last().unwrap(), &next_event);
            let new_cost = cost + self.cost_between(path.last().unwrap(), &next_event);

            let mut new_path = path.clone();
            new_path.push(next_event);

            let mut new_to_treat = to_treat.clone();
            new_to_treat.retain(|&x| x != *i);

            let mut new_to_empty = to_empty.clone();
            new_to_empty.push(*i);

            let mut new_done = done.clone();

            let result = self.generate_fragments(new_path, new_time, new_cost, new_to_treat, new_to_empty, new_done, num_p);

            if result {
                found_true = true;
                // todo add the optimization, how does return true work?
            }
        }

        for i in &to_empty {
            let next_event = Event { request_id: *i, action: Action::Deliver };

            let new_time = time + self.time_between(path.last().unwrap(), &next_event);
            let new_cost = cost + self.cost_between(path.last().unwrap(), &next_event);

            let mut new_path = path.clone();
            new_path.push(next_event);

            let new_to_treat = to_treat.clone();

            let mut new_to_empty = to_empty.clone();
            new_to_empty.retain(|&x| x != *i);

            let mut new_done = done.clone();
            new_done.push(*i);

            let result = self.generate_fragments(new_path, new_time, new_cost, new_to_treat, new_to_empty, new_done, num_p);

            if result {
                found_true = true;
                // todo add the optimization, how does return true work?
            }
        }

        if !found_true {
            return false;
        }
        
        // Extending to the next pickup node
        if to_treat.len() + to_empty.len() + done.len() < 2 {
            let mut total = Vec::new();
            total.extend(to_treat.clone());
            total.extend(to_empty.clone());
            total.extend(done.clone());
            // Remove duplicates
            total.sort();
            total.dedup();

            // If there is a request to empty, assign that
            let mut to_empty_location = None;
            if to_empty.len() > 0 {
                to_empty_location = Some(self.data.requests[to_empty[0]].from_id);
            }

            for request_id in 0..self.data.requests.len() {
                if !total.contains(&request_id) {
                    let request = &self.data.requests[request_id];
                    if to_empty_location.is_some() && to_empty_location.unwrap() == request.from_id {
                        continue;
                    }

                    if path.last().unwrap().action == Action::Pickup && 
                            path.last().unwrap().request_id > request_id &&
                            self.data.requests[path.last().unwrap().request_id].from_id == request.from_id {
                        continue;
                    }

                    let next_event = Event { request_id, action: Action::Pickup };
                    let new_time = time + self.time_between(path.last().unwrap(), &next_event);
                    let new_cost = cost + self.cost_between(path.last().unwrap(), &next_event);

                    let mut new_path = path.clone();
                    new_path.push(next_event);

                    let mut new_to_treat = to_treat.clone();
                    new_to_treat.push(request_id);

                    let new_to_empty = to_empty.clone();

                    let mut new_done = done.clone();

                    let result = self.generate_fragments(new_path, new_time, new_cost, new_to_treat, new_to_empty, new_done, num_p + 1);
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
            Action::PP => self.data.requests[a.request_id].from_id,
        };
        let b_loc = match b.action {
            Action::Pickup => self.data.requests[b.request_id].from_id,
            Action::Treat => self.data.requests[b.request_id].to_id,
            Action::Deliver => self.data.requests[b.request_id].from_id,
            Action::PP => self.data.requests[b.request_id].from_id,
        };
        self.data.time[a_loc][b_loc]
    }

    fn cost_between(&self, a: &Event, b: &Event) -> usize {
        let a_loc = match a.action {
            Action::Pickup => self.data.requests[a.request_id].from_id,
            Action::Treat => self.data.requests[a.request_id].to_id,
            Action::Deliver => self.data.requests[a.request_id].from_id,
            Action::PP => self.data.requests[a.request_id].from_id,
        };
        let b_loc = match b.action {
            Action::Pickup => self.data.requests[b.request_id].from_id,
            Action::Treat => self.data.requests[b.request_id].to_id,
            Action::Deliver => self.data.requests[b.request_id].from_id,
            Action::PP => self.data.requests[b.request_id].from_id,
        };
        self.data.distance[a_loc][b_loc]
    }

    fn optimise_requests() {
        // Organise the quantities
    }
}

#[derive(Debug, Clone)]
struct Fragment {
    pub events: Vec<Event>,
    pub time: usize,
    pub cost: usize,
    pub done: Vec<usize>,
}

impl Fragment {
    fn new(events:Vec<Event>, time: usize, cost: usize, done: Vec<usize>) -> Self {
        Fragment {
            events,
            time,
            cost,
            done,
        }
    }
}

#[derive(Debug, Clone)]
struct Event {
    request_id: usize,
    action: Action,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Action {
    Pickup,
    Treat,
    Deliver,
    PP
}

#[derive(Debug, Clone)]
struct SPDPData {
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
        println!("{num}");
        let container_types: usize = num.trim().parse().unwrap();

        let (name, num) = lines.next().unwrap().split_once("\t").unwrap();
        assert_eq!(name, "WASTE_TYPES");
        let waste_types: usize = num.trim().parse().unwrap();

        let (name, num) = lines.next().unwrap().split_once("\t").unwrap();
        assert_eq!(name, "LOCATIONS");
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

        let requests: Vec<Request> = (0..num_requests)
            .map(|_| {
                let data = lines.next().unwrap();
                Request::from_line(data)
                }
            ).collect();

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
struct Request {
    id: usize,
    waste_id: usize,
    from_id: usize,
    container_type: usize,
    to_num: usize,
    to_id: usize,
    quantity: usize,
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
            id,
            waste_id,
            from_id,
            container_type,
            to_num,
            to_id,
            quantity: 1,
        }
    }
}

// Short tests to confirm the data loading is functional
#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn test_load_data() {
        let spdp = SPDPData::from_file("./Data/SkipData/Benchmark/RecDep_day_A1.dat");

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