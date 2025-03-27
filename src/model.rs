extern crate grb;

use grb::attribute::ModelModelSenseAttr::ModelSense;
use grb::prelude::*;

use super::utils::*;
use super::fragment::*;

pub struct RouteFormulationModel {
    pub data : SPDPData,
    pub arcs : ArcContainer,
    pub nodes: NodeContainer,
    pub model: Option<Model>,
    pub lambda: Vec<Var>,
    pub cover_constraints: Vec<Constr>,
    pub c: Vec<f64>,
    pub a: Vec<Vec<f64>>,
}

impl RouteFormulationModel {

    pub fn new(data: SPDPData) -> Self {
        let generator = Generator::new(data.clone());

        let rfm = RouteFormulationModel {
            data,
            arcs: generator.generate_arcs(),
            nodes: generator.generate_nodes(),
            model: None,
            lambda: Vec::new(),
            cover_constraints: Vec::new(),
            c: Vec::new(),
            a: Vec::new(),
        };

        rfm
    }

    fn initialise_model(&mut self) {
        // Generate variables, one for each route

        // Queue keeps track of tuples (node, serviced, time, cost)
        let mut queue = vec![(self.nodes.depot, Vec::new(), 0, 0)];

        let mut routes: Vec<(Vec<usize>, usize)> = Vec::new();
        
        while !queue.is_empty() {
            let (current, serviced, time, cost) = queue.pop().unwrap();
            // println!("Current: {:?}, Serviced: {:?}, Time: {}, Cost: {}", current, serviced, time, cost);
            let duplicate_service: bool = (0..self.data.num_requests)
                .map(|req_id| serviced.iter().filter(|&s| *s == req_id).count())
                .enumerate()
                .any(|(r_id, count)| count > self.data.requests[r_id].quantity);
            if time > self.data.t_limit || duplicate_service {
                continue;
            }

            // Check if we have reached the depot
            if current.is_depot() && serviced.len() > 0 {
                routes.push((serviced, cost));
                continue;
            }

            for arc in self.arcs.arcs_from.get(&current).unwrap() {
                let next = arc.end;
                let next_time = time + arc.time;
                let next_cost = cost + arc.cost;
                let mut next_serviced = serviced.clone();

                if arc.done.len() > 0 {
                    next_serviced.push(arc.done.left());
                }

                if arc.done.len() == 2 {
                    next_serviced.push(arc.done.right());
                }

                queue.push((next, next_serviced, next_time, next_cost));
            }
        }

        let num_routes = routes.len();
        println!("Number of routes: {}", num_routes);

        self.a = vec![vec![0.0; num_routes]; self.data.num_requests];

        for (route_id, (serviced, cost)) in routes.iter().enumerate() {
            self.c.push(*cost as f64);

            for request in serviced.iter() {
                self.a[*request][route_id] += 1.0;
            }
        }

        let mut model = Model::new("Route Formulation").unwrap();

        // Variables
        self.lambda = (0..num_routes)
            .map(|r_id| add_binvar!(model, name: &format!("Route_{r_id}"), obj: self.c[r_id]).unwrap())
            .collect();

        // Objective
        let _ = model.set_attr(ModelSense, Minimize);

        // Constraints
        for (request_id, request) in self.data.requests.iter().enumerate() {
            let constr_expr = self.lambda
                .iter()
                .zip((0..num_routes).map(|r_id| self.a[request_id][r_id]))
                .map(|(lambda, a)| (a * *lambda))
                .grb_sum();

            self.cover_constraints.push(
                model.add_constr(
                    &format!("request_{}", request_id),
                    c!( constr_expr == request.quantity)
                ).unwrap()
            );
        }

        self.model = Some(model);
    }

    pub fn solve(&mut self) {
        self.initialise_model();
        let _ = self.model.as_mut().unwrap().optimize();
    }
}