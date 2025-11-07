use std::sync::OnceLock;
use crate::utils::SPDPData;
use crate::constants::SIZE;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoverSetManager {
    offsets: Vec<usize>,
    overflow_flag_mask: SIZE,
    overflow_add: SIZE,
}

impl CoverSetManager {
    fn new(data: &SPDPData) -> Self {
        let mut offsets = vec![0; data.num_requests];
        let mut overflow_flag_mask: SIZE = 0;
        let mut overflow_add: SIZE = 0;
        let mut current_offset = 0;
        for request_id in 0..data.num_requests {
            let request = &data.requests[request_id];
            let q = (request.quantity) as SIZE; // turn this back
            let num_bits = (q as f64 + 0.5).log2().ceil() as usize;
            let overflow_bit = num_bits;

            if current_offset + overflow_bit > 128 {
                panic!("Overflow bit exceeds 128 bits, increase the size of the CoverSetManager");
            }

            overflow_flag_mask |= 1 << overflow_bit + current_offset;
            offsets[request_id] = current_offset;
            overflow_add |= ((1 << overflow_bit) - q - 1) << current_offset;
            current_offset += overflow_bit + 1;
        }

        CoverSetManager {
            offsets,
            overflow_flag_mask,
            overflow_add,
        }
    }

    pub fn print(&self) {
        println!("CoverSetManager:");
        println!("  Offsets: {:?}", self.offsets);
        println!("  Overflow flag mask: {:b}", self.overflow_flag_mask);
        println!("  Overflow add: {:b}", self.overflow_add);
    }
}

#[derive(Debug, Clone, Copy, Eq)]
pub struct CoverSet {
    pub covered: SIZE, // This should be big enough for 64 requests, especially with duplicates
    manager: &'static CoverSetManager,
    pub len: usize,
}

impl CoverSet {
    pub fn new(manager: &'static CoverSetManager) -> Self {
        CoverSet { 
            covered: 0,
            manager,
            len: 0,
        }
    }

    pub fn cover(&mut self, request_id: usize) -> Result<(), ()> {
        self.covered += 1 << self.manager.offsets[request_id];

        if self.is_valid() {
            self.len += 1; // Increment length only if valid
            Ok(())
        } else {
            self.covered -= 1 << self.manager.offsets[request_id]; // Rollback if invalid
            Err(())
        }
    }

    pub fn uncover(&mut self, request_id: usize) -> Result<(), ()> {
        match self.covered.checked_sub(1 << self.manager.offsets[request_id]) {
            Some(new_covered) => {
                self.covered = new_covered;
                self.len -= 1;
            },
            None => {
                return Err(()); // Cannot uncover if it would result in negative coverage
            }
        }

        if self.is_valid() {
            Ok(())
        } else {
            self.len += 1;
            self.covered += 1 << self.manager.offsets[request_id]; // Rollback if invalid
            Err(())
        }
    }

    pub fn combine(self, other: &CoverSet) -> Result<Self, String> {
        let new = CoverSet {
            covered: self.covered + other.covered,
            manager: self.manager,
            len: self.len + other.len,
        };

        if new.is_valid() {
            Ok(new)
        } else {
            Err("Combined CoverSet is invalid due to overflow".to_string())
        }
    }

    fn is_valid(&self) -> bool {
        let overflow_flag = (self.covered + self.manager.overflow_add) & self.manager.overflow_flag_mask;
        overflow_flag == 0
    }

    pub fn visits_leq_than(&self, other: &CoverSet) -> bool {
        0 == !((other.covered | self.manager.overflow_flag_mask) - self.covered) & self.manager.overflow_flag_mask
    }

    pub fn get_cover(&self, request_id: usize) -> usize {
        let offset = self.manager.offsets[request_id];
        let mut result = self.covered >> offset;
        if request_id != self.manager.offsets.len() - 1 {
            result &= (1 << (self.manager.offsets[request_id + 1] - offset)) - 1;   
        }
        result as usize
    }

    pub fn to_vec(&self) -> Vec<usize> {
        assert!(self.is_valid(), "CoverSet is not valid, cannot iterate");
        self.manager.offsets.iter().enumerate().filter_map(|(i, &offset)| {
            let mut result = self.covered >> offset;
            if i != self.manager.offsets.len() - 1 {
                result &= (1 << (self.manager.offsets[i + 1] - offset)) - 1;   
            }
            Some(result as usize)
        }).collect()
    }
}

impl PartialEq for CoverSet {
    fn eq(&self, other: &Self) -> bool {
        self.covered == other.covered
    }
}

impl PartialOrd for CoverSet {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CoverSet {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.covered.cmp(&other.covered)
    }
}

static MANAGER: OnceLock<CoverSetManager> = OnceLock::new();
pub fn get_manager() -> &'static CoverSetManager {
    MANAGER.get().expect("CoverSetManager has not been initialized")
}

pub fn init_manager(data: &SPDPData) -> &'static CoverSetManager {
    MANAGER.get_or_init(|| CoverSetManager::new(data))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() {
        let data = SPDPData::from_file("./SkipData/Benchmark/RecDep_day_A5.dat"); // Assuming a default implementation exists
        init_manager(&data);
    }

    #[test]
    fn test_cover_set_manager() {
        setup();
        let manager = get_manager();
        manager.print();
    }

    #[test]
    fn test_cover_set_valid() {
        setup();
        let manager = get_manager();
        let mut cover_set = CoverSet::new(&manager);
        assert!(cover_set.cover(0).is_ok());
        assert!(cover_set.cover(1).is_ok());
        assert!(cover_set.cover(0).is_err());
        assert!(cover_set.is_valid());
        assert!(cover_set.cover(2).is_ok());
        assert!(!cover_set.cover(2).is_ok());
        assert!(cover_set.cover(2).is_err());
        assert!(cover_set.is_valid());
    }

    #[test]
    fn test_cover_set_uncover() {
        setup();
        let manager = get_manager();
        let mut cover_set = CoverSet::new(&manager);
        assert!(cover_set.cover(0).is_ok());
        assert!(cover_set.cover(1).is_ok());
        assert!(cover_set.uncover(0).is_ok());
        assert!(cover_set.uncover(1).is_ok());
        assert!(cover_set.uncover(2).is_err());
        assert!(cover_set.is_valid());
    }

    #[test]
    fn test_cover_set_leq() {
        setup();
        let manager: &'static CoverSetManager = get_manager();
        let mut cover_set1 = CoverSet::new(&manager);
        assert!(cover_set1.cover(0).is_ok());
        assert!(cover_set1.cover(1).is_ok());
        
        let mut cover_set2 = CoverSet::new(&manager);
        assert!(cover_set2.cover(0).is_ok());
        assert!(cover_set2.cover(1).is_ok());
        assert!(cover_set2.cover(2).is_ok());

        let mut cover_set3 = CoverSet::new(&manager);
        assert!(cover_set3.cover(0).is_ok());
        assert!(cover_set3.cover(3).is_ok());
        assert!(cover_set3.cover(4).is_ok());

        assert!(cover_set1.visits_leq_than(&cover_set2));
        assert!(!cover_set2.visits_leq_than(&cover_set1));
        assert!(!cover_set1.visits_leq_than(&cover_set3));
        assert!(!cover_set2.visits_leq_than(&cover_set3));
    }

    #[test]
    fn test_to_vec() {
        setup();
        let manager: &'static CoverSetManager = get_manager();
        let mut cover_set = CoverSet::new(&manager);
        assert!(cover_set.cover(0).is_ok());
        assert!(cover_set.cover(1).is_ok());
        assert!(cover_set.cover(2).is_ok());
        assert!(cover_set.cover(5).is_ok());
        
        let vec = cover_set.to_vec();
        assert_eq!(vec, vec![1, 1, 1, 0, 0, 1, 0, 0]); // Assuming the first three requests are covered

        assert!(cover_set.get_cover(0) == 1);
        assert!(cover_set.get_cover(1) == 1);
        assert!(cover_set.get_cover(2) == 1);
        assert!(cover_set.get_cover(3) == 0);
        assert!(cover_set.get_cover(5) == 1);

        cover_set.cover(7).unwrap();
        cover_set.cover(7).unwrap();
        assert!(cover_set.get_cover(7) == 2);
    }
}