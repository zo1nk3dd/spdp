extern crate grb;

use grb::prelude::*;

use super::utils::*;
use super::fragment::*;
use super::pricing::*;


pub struct MasterProblemModel {
    pub data : SPDPData,
    pub arcs : ArcContainer,
    pub nodes: NodeContainer,
    pub model: Model,
    pub omega: Vec<Var>,
    pub cover: Vec<Constr>,
    pub flow: Vec<Constr>,
    pub c: Vec<f64>,
    pub a: Vec<Vec<f64>>,
}

pub struct ColGenModel {
    pub data: SPDPData,
    pub arcs: ArcContainer,
    pub nodes: NodeContainer,
    pub model: Model,
    pub routes_covering_request: Vec<Vec<usize>>,
    pub lambda: Vec<Var>,
    pub cover_constraints: Vec<Constr>,
}

impl ColGenModel {
    pub fn new(data: SPDPData) -> Self {
        let generator = Generator::new(data.clone());

        let node_container = generator.generate_nodes();
        let arc_container = generator.generate_arcs(&node_container);

        let model = Model::new("COLGEN Problem").unwrap();

        let mut result = ColGenModel {
            data: data.clone(),
            arcs: arc_container,
            nodes: node_container,
            model,
            routes_covering_request: vec![Vec::new(); data.num_requests],
            lambda: Vec::new(),
            cover_constraints: Vec::new(),
        };

        result.first_initialisation();
        result.initialise_model_constraints();
        result
    }

    pub fn add_route_var(&mut self, cost: f64, covered: Vec<usize>) {
        // Add a new route to the model
        let route_id = self.lambda.len();

        // let model = &mut self.model;

        for request_id in 0..self.data.num_requests {
            self.routes_covering_request[request_id].push(0);
        }

        for request_id in covered.iter() {
            self.routes_covering_request[*request_id][route_id] += 1;
        }

        let requests_covered_by_var = self.cover_constraints.iter()
            .map(|c| *c)
            .zip((0..self.data.num_requests).map(|request_id| self.routes_covering_request[request_id][route_id] as f64));

        let lambda = self.model.add_var(
            &format!("route_{route_id}"),
            Continuous,
            cost,
            0.0,
            1.0,
            requests_covered_by_var,
        ).unwrap();

        self.lambda.push(lambda);
        
        
    }

    pub fn initialise_model_constraints(&mut self) {
        self.cover_constraints = (0..self.data.num_requests)
            .map(|request_id| {
                let constr_expr = self.lambda.iter()
                    .zip(self.routes_covering_request[request_id].iter())
                    .map(|(lambda, a)| (*a as f64 * *lambda))
                    .grb_sum();

                self.model.add_constr(
                    &format!("request_{}", request_id),
                    c!(constr_expr == self.data.requests[request_id].quantity as f64)
                ).unwrap()
            })
            .collect::<Vec<_>>();

        self.model.update().unwrap();
    }

    pub fn first_initialisation(&mut self) {
        let mut initial_routes: Vec<(usize, Vec<usize>)> = Vec::new();
        for arc in self.arcs.arcs.iter() {
            if arc.start.is_pickup() && arc.end.is_depot() {
                let mut covered = Vec::new();
                if arc.done.len() > 0 {
                    covered.push(arc.done.left());
                }
                if arc.done.len() == 2 {
                    covered.push(arc.done.right());
                }

                initial_routes.push((arc.cost, covered));
            }
        }

        for (_cost, covered) in initial_routes.iter() {
            self.add_route_var(1.0, covered.clone());
        }
    }

    pub fn solve(&mut self) {
        let mut iter = 1;
        loop {
            self.model.set_attr(attr::ModelSense, Minimize).unwrap();
            let result = self.model.optimize();
            assert!(result.is_ok());

            let duals = self.cover_constraints.iter()
                .map(|c| self.model.get_obj_attr(attr::Pi, c).unwrap())
                .collect::<Vec<_>>();

            let mut pricer = Pricer::new(
                &self.nodes, 
                &self.arcs, 
                &self.data,
                &duals,
            );

            let new_routes = pricer.solve_pricing_problem(1);

            if new_routes.len() == 0 {
                print!("Optimal obj found after {} iterations\n", iter);
                let obj = self.model.get_attr(attr::ObjVal).unwrap();
                println!("Objective value: {:?}", obj);
                break;
            }
            else {             
                for route in new_routes.iter() {
                    let _cost = route.cost;
                    let covered = route.covered.clone();
                    self.add_route_var(1.0 as f64, covered);
                }
                self.model.update().unwrap();
            }
            iter += 1;
        }
    }
}


// impl MasterProblemModel {
//     pub fn new(data: SPDPData) -> Self {
//         let generator = Generator::new(data.clone());

//         let arc_container = generator.generate_arcs();
//         let node_container = generator.generate_nodes();
//         let mut model = Model::new("Master Problem").unwrap();

//         let c: Vec<f64> = arc_container.arcs.iter()
//             .map(|arc| arc.cost as f64)
//             .collect();

//         let mut a = vec![vec![0.0; arc_container.num_arcs()]; data.num_requests];

//         for arc in arc_container.arcs.iter() {
//             if arc.done.len() > 0 {
//                 a[arc.done.left()][arc.id] += 1.0;
//             }
//             if arc.done.len() == 2 {
//                 a[arc.done.right()][arc.id] += 1.0;
//             }
//         }

//         let omega: Vec<Var> = (0..arc_container.num_arcs())
//             .map(|arc_id| {
//                 let arc = &arc_container.arcs[arc_id];
//                 add_var!(model, Continuous, name: &format!("a_{arc_id}"), obj: arc.cost as f64, bounds: 0.0..0.0).unwrap()
//             })
//             .collect();

//         let cover: Vec<Constr> = data.requests.iter()
//             .enumerate()
//             .map(|(r_id, request)| {
//                 let constr_expr = omega.iter()
//                     .zip((0..arc_container.num_arcs()).map(|arc_id| a[r_id][arc_id]))
//                     .map(|(omega, a)| (a * *omega))
//                     .grb_sum();

//                 let quantity = request.quantity as f64;
//                 assert!(quantity == 1.0);

//                 model.add_constr(
//                     &format!("request_{}", r_id),
//                     c!(constr_expr == quantity)
//                 ).unwrap()
//             })
//             .collect();

//         let flow: Vec<Constr> = node_container.nodes
//             .iter()
//             .enumerate()
//             .map(|(node_id, node)| {
//                 let inflow = arc_container.arcs_to.get(node).unwrap()
//                     .iter()
//                     .map(|arc| omega[arc.id])
//                     .grb_sum();

//                 let outflow = arc_container.arcs_from.get(node).unwrap()
//                     .iter()
//                     .map(|arc| omega[arc.id])
//                     .grb_sum();

//                 model.add_constr(
//                     &format!("flow_{}", node_id),
//                     c!(inflow == outflow)
//                 ).unwrap()
//             })
//             .collect();

//         // The number of vehicles leaving the depot is accurate
//         let depot_outflow = arc_container.arcs_from.get(&node_container.depot).unwrap()
//             .iter()
//             .map(|arc| omega[arc.id])
//             .grb_sum();

//         model.add_constr(
//             "depot_outflow",
//             c!(depot_outflow == 1 as f64)
//         ).unwrap();

//         // Set the upper bound of a small subset of variables to be 1.0
//         for arc in arc_container.arcs_from.get(&node_container.depot).unwrap() {
//             let result = model.set_obj_attr(UB, &omega[arc.id], 1.0);
//             assert!(result.is_ok());
//         }    

//         for arc in arc_container.arcs_to.get(&node_container.depot).unwrap() {
//             let result = model.set_obj_attr(UB, &omega[arc.id], 1.0);
//             assert!(result.is_ok());
//         }

//         MasterProblemModel {
//             data,
//             arcs: arc_container,
//             nodes: node_container,
//             model,
//             omega,
//             cover,
//             flow,
//             c,
//             a,
//         }
//     }

//     pub fn solve(&mut self) {
//         let result = self.model.optimize();
//         assert!(result.is_ok());
//     }
// }