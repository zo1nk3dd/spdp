use std::vec;

use super::utils::*;
use super::fragment::*;

pub struct ModelRestricted {
    pub data : SPDPData,
    pub fragments: Vec<Fragment>,
}

impl ModelRestricted {

    pub fn new(data: SPDPData) -> Self {
        let fragments = Generator::new(data.clone()).generate_naive_fragments();

        let model = ModelRestricted {
            data,
            fragments,
        };

        model
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

    #[test]

    fn test_fragment() {
        let spdp = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_A1.dat");
        let model = ModelRestricted::new(spdp);
        assert_eq!(model.fragments.len(), 145);

        let spdp = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_D20.dat");
        let model = ModelRestricted::new(spdp);
        assert_eq!(model.fragments.len(), 130444);

        let spdp = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_C6.dat");
        let model = ModelRestricted::new(spdp);
        assert_eq!(model.fragments.len(), 1920);
    }
}