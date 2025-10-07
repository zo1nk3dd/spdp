pub const EPS: f64 = 1e-6;
pub type SIZE = u128; // The size of the cover set

pub const FORWARD_BACKWARD_PASS_MARK: f64 = 0.45; // Percentage that forward pass goes past and backwards pass goes to
pub const NUM_ROUTES_PER_NODE_CALCULATED: usize = 20;
pub const NUM_ROUTES_PER_NODE_ADDED: usize = 3;
pub const GAP_MULTIPLIER: f64 = 1.005; // Multiplier for the gap when calculating routes

pub const SRI_CONSTRAINTS_ENABLED: bool = true; // Whether to use the subset row inequalities