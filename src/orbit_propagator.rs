use std::f64::consts::PI;

use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::Vector3;

use crate::{
    kepler_orbit::Orbit,
    lambert_solver::{solve_velocities, LambertSolver},
};

fn minimum_energy_transfer_time(kkv: &Orbit, target: &Orbit, mu: f64) -> f64 {
    let s = 2.0 * (kkv.r.norm() + target.r.norm());
    let c = (kkv.r - target.r).norm();
    let beta_min = 2.0 * ((s - c) / 2.0).sqrt().sin();
    (s.powi(3) / (8.0 * mu)).sqrt() * (PI - beta_min - beta_min.sin())
}

pub fn calculate_transfers(
    kkv: Orbit,
    target: Orbit,
    tstep: f64,
    tmax: Option<f64>,
    dv_max: f64,
    mu: f64,
) -> (Vec<f64>, Vec<Vector3<f64>>) {
    let tmax = if let Some(tmax) = tmax {
        tmax
    } else {
        kkv.period(mu).max(target.period(mu))
    };
    let mut time = minimum_energy_transfer_time(&kkv, &target, mu);
    let lm = LevenbergMarquardt::new();
    let guess = Vector3::new((kkv.r.norm() + target.r.norm()) / 2.0, PI / 2.0, PI / 2.0);
    let mut solver = LambertSolver::new(guess, kkv.r, target.r, time, mu);
    let mut times = vec![];
    let mut velocities = vec![];

    while time <= tmax {
        let target_step = target.propagate_time(time, mu);

        solver.r2 = target_step.r;
        solver.dt = time;
        let (updated_solver, report) = lm.minimize(solver);
        solver = updated_solver;

        let (v1, _) =
            solve_velocities(kkv.r, target_step.r, solver.v.x, solver.v.y, solver.v.z, mu);

        if report.number_of_evaluations <= 15 {
            velocities.push(v1);
            times.push(solver.dt);
        }

        time += tstep;
    }
    time = minimum_energy_transfer_time(&kkv, &target, mu) - tstep;
    let mut dv = 0.0;
    while dv <= dv_max {
        let target_step = target.propagate_time(time, mu);

        solver.r2 = target_step.r;
        solver.dt = time;
        let (updated_solver, report) = lm.minimize(solver);
        solver = updated_solver;

        let (v1, _) =
            solve_velocities(kkv.r, target_step.r, solver.v.x, solver.v.y, solver.v.z, mu);

        velocities.insert(0, v1);
        times.insert(0, solver.dt);
        dv = (v1 - kkv.v).norm();

        if report.objective_function > 1e-20 {
            dbg!(report);
            break;
        }
        time -= tstep;
    }
    (times, velocities)
}
