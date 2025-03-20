use super::utils::*;
use super::fragment::*;

pub struct ModelRestricted {
    pub data : SPDPData,
    pub fragments: Vec<Fragment>,
}

impl ModelRestricted {

    pub fn new(data: SPDPData) -> Self {
        let fragments = Generator::new(data.clone()).generate_fragments();

        let model = ModelRestricted {
            data,
            fragments,
        };

        model
    }
}