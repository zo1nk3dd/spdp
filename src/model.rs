use std::fs;

struct ModelRestricted {
    pub data : SPDPData,

}

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