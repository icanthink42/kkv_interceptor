use core::f64;
use parking_lot::Mutex;
use std::{f64::consts::PI, future, sync::Arc};
use tokio::task::spawn;

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
    kkv: &Orbit,
    target: &Orbit,
    tstep: f64,
    tmax: Option<f64>,
    dv_max: f64,
    kill_velocity: f64,
    mu: f64,
) -> (Vec<f64>, Vec<Vector3<f64>>, Vec<Vector3<f64>>) {
    let tmax = if let Some(tmax) = tmax {
        tmax
    } else {
        kkv.period(mu).max(target.period(mu))
    };
    let start_time = minimum_energy_transfer_time(&kkv, &target, mu);
    let mut time = start_time;
    let lm = LevenbergMarquardt::new();
    let guess = Vector3::new((kkv.r.norm() + target.r.norm()) / 2.0, PI / 2.0, PI / 2.0);
    let mut solver = LambertSolver::new(guess, kkv.r, target.r, time, mu);
    let mut times = vec![];
    let mut start_velocities = vec![];
    let mut end_velocities = vec![];

    while time <= tmax + start_time {
        let target_step = target.propagate_time(time, mu);

        solver.r2 = target_step.r;
        solver.dt = time;
        let (updated_solver, report) = lm.minimize(solver);
        solver = updated_solver;

        let (v1, v2) =
            solve_velocities(kkv.r, target_step.r, solver.v.x, solver.v.y, solver.v.z, mu);

        let dv = (v1 - kkv.v).norm();
        let intercept_v = (v2 - target.v).norm();
        if report.number_of_evaluations <= 15 && dv <= dv_max && intercept_v >= kill_velocity {
            start_velocities.push(v1);
            end_velocities.push(v1);
            times.push(solver.dt);
        }

        time += tstep;
    }

    let (updated_solver, _) = lm.minimize(solver);
    solver = updated_solver;
    let (starting_v1, _) =
        solve_velocities(kkv.r, target.r, solver.v.x, solver.v.y, solver.v.z, mu);

    time = start_time;
    let mut dv = (starting_v1 - kkv.v).norm();
    while dv <= dv_max {
        let target_step = target.propagate_time(time, mu);

        solver.r2 = target_step.r;
        solver.dt = time;
        let (updated_solver, report) = lm.minimize(solver);
        solver = updated_solver;

        let (v1, v2) =
            solve_velocities(kkv.r, target_step.r, solver.v.x, solver.v.y, solver.v.z, mu);

        let intercept_v = (v2 - target.v).norm();
        if intercept_v >= kill_velocity && dv <= dv_max {
            start_velocities.insert(0, v1);
            end_velocities.insert(0, v2);
            times.insert(0, solver.dt);
        }
        dv = (v1 - kkv.v).norm();

        if report.number_of_evaluations > 15 {
            break;
        }
        time -= tstep;
    }
    (times, start_velocities, end_velocities)
}

pub struct PossibleIntercept {
    pub orbit: Arc<Orbit>,
    pub dv: Vector3<f64>,
    pub dt: f64,
    pub burn_time: f64,
}

impl PossibleIntercept {
    fn new(orbit: Orbit, dv: Vector3<f64>, dt: f64, burn_time: f64) -> Self {
        Self {
            orbit: orbit.into(),
            dv,
            dt,
            burn_time,
        }
    }
}

#[derive(Debug)]
pub struct InterceptError<const COUNT: usize> {
    pub ideal_orbit: Arc<Orbit>,
    pub error_orbit: [Orbit; COUNT],
    pub dv: Vector3<f64>,
    pub dt: f64,
    pub burn_time: f64,
}

impl<const COUNT: usize> InterceptError<COUNT> {
    #[allow(dead_code)]
    pub fn from_intercept(
        intercept: &PossibleIntercept,
        r_error: [Vector3<f64>; COUNT],
        v_error: [Vector3<f64>; COUNT],
    ) -> Self {
        let error_orbit = r_error
            .iter()
            .zip(v_error)
            .map(|(r_e, v_e)| Orbit::new(intercept.orbit.r + r_e, intercept.orbit.v + v_e))
            .collect::<Vec<Orbit>>()
            .try_into()
            .expect("Failed unpacking error values.");
        Self {
            error_orbit,
            ideal_orbit: intercept.orbit.clone(),
            dv: intercept.dv,
            dt: intercept.dt,
            burn_time: intercept.burn_time,
        }
    }

    pub fn from_intercept_v(intercept: &PossibleIntercept, v_error: [Vector3<f64>; COUNT]) -> Self {
        let error_orbit = v_error
            .iter()
            .map(|v_e| Orbit::new(intercept.orbit.r, intercept.orbit.v + v_e))
            .collect::<Vec<Orbit>>()
            .try_into()
            .expect("Failed unpacking error values.");
        Self {
            error_orbit,
            ideal_orbit: intercept.orbit.clone(),
            dv: intercept.dv,
            dt: intercept.dt,
            burn_time: intercept.burn_time,
        }
    }

    pub fn dx(&self, mu: f64) -> [Vector3<f64>; COUNT] {
        let (ideal, _) = self.ideal_orbit.propagate_time_xyz(self.dt, mu);
        let ideal_error = self
            .error_orbit
            .iter()
            .map(|orbit| {
                let (error, _) = orbit.propagate_time_xyz(self.dt, mu);
                ideal - error
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("Failed unpacking error values");
        ideal_error
    }
}

pub async fn propagate(
    kkv: &Orbit,
    target: &Orbit,
    tstep: f64,
    tmin: f64,
    tmax: f64,
    dv_max: f64,
    kill_velocity: f64,
    planet_radius: f64,
    mu: f64,
) -> Vec<PossibleIntercept> {
    let mut time = tmin;
    let mut tasks = vec![];
    let mut intercepts = vec![];
    let tasks_completed = Arc::new(Mutex::new(0));
    while time <= tmax {
        let new_kkv = kkv.propagate_time(time, mu);
        let new_target = target.propagate_time(time, mu);
        if new_kkv.r.norm() < planet_radius || new_target.r.norm() < planet_radius {
            break;
        }
        let task = spawn(async_transfer(
            new_kkv,
            new_target,
            tstep,
            tmax,
            dv_max,
            time,
            tasks_completed.clone(),
            kill_velocity,
            planet_radius,
            mu,
        ));
        tasks.push(task);
        time += 1.0;
    }
    for task in tasks {
        intercepts.append(&mut task.await.unwrap());
    }
    intercepts
}

async fn async_transfer(
    new_kkv: Orbit,
    target: Orbit,
    tstep: f64,
    tmax: f64,
    dv_max: f64,
    burn_time: f64,
    tasks_completed: Arc<Mutex<usize>>,
    kill_velocity: f64,
    planet_radius: f64,
    mu: f64,
) -> Vec<PossibleIntercept> {
    let mut intercepts = vec![];
    let (times, start_velocities, _) = calculate_transfers(
        &new_kkv,
        &target,
        tstep,
        Some(tmax),
        dv_max,
        kill_velocity,
        mu,
    );
    for (dt, velocity) in times.into_iter().zip(start_velocities) {
        let dv = new_kkv.v - velocity;
        let orbit = Orbit::new(new_kkv.r, velocity);
        let future_orbit = orbit.propagate_time(dt, mu);
        if future_orbit.r.norm() > planet_radius {
            intercepts.push(PossibleIntercept::new(orbit, dv, dt, burn_time));
        }
    }
    let mut t = tasks_completed.lock();
    *t += 1;
    intercepts
}
