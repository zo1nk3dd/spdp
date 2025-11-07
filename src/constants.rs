pub const EPS: f64 = 1e-6;
pub type SIZE = u128; // The size of the cover set
pub const M: f64 = 1e4; // A large constant for Big-M constraints

pub const FORWARD_BACKWARD_PASS_MARK: f64 = 0.4; // Percentage that forward pass goes past and backwards pass goes to
pub const NUM_ROUTES_PER_NODE_CALCULATED: usize = 20;
pub const NUM_ROUTES_PER_NODE_ADDED: usize = 3;

pub const SRI_CONSTRAINTS_ENABLED: bool = true; // Whether to use the subset row inequalities
pub const OVER_HALF_GAP_CUTS_ENABLED: bool = true; // Whether to use the over half gap cuts
pub const NUM_ROUTES_FOR_RIP: usize = 10000;