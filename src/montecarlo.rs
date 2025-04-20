use std::{f64::INFINITY, sync::Arc, usize};

use nalgebra::Vector3;
use rand_distr::{Distribution, Normal};
use tokio::spawn;

use crate::{
    kepler_orbit::Orbit,
    orbit_propagator::{propagate, InterceptError, PossibleIntercept},
};

pub async fn minimize_x_error<const COUNT: usize>(
    kkv: Orbit,
    target: Orbit,
    tstep: f64,
    tstepmin: f64,
    tstepdiv: f64,
    tmax: Option<f64>,
    dv_max: f64,
    kill_v: f64,
    dv_error_mean: f64,
    dv_error_stdev: f64,
    mu: f64,
) -> InterceptError<COUNT> {
    let mut tmax = if let Some(tmax) = tmax {
        tmax
    } else {
        kkv.period(mu).max(target.period(mu))
    };
    let normal = Normal::new(dv_error_mean, dv_error_stdev).unwrap();
    let mut tstep = tstep;
    let target = Arc::new(target);
    let mut tmin = 0.0;
    loop {
        dbg!("Completed Iteration");
        let intercepts =
            propagate(&kkv, target.clone(), tstep, tmin, tmax, dv_max, kill_v, mu).await;
        let mut tasks = vec![];
        for intercept in intercepts {
            let task = spawn(montecarlo_async(intercept, normal, mu));
            tasks.push(task);
        }
        let mut errors = vec![];
        let mut intercept_errors: Vec<InterceptError<COUNT>> = Vec::new();
        for task in tasks {
            let (intercept, error) = task.await.unwrap();
            intercept_errors.push(intercept);
            errors.push(error);
        }
        let mut min_error = INFINITY;
        let mut min_index = 0;
        for (i, _) in intercept_errors.iter().enumerate() {
            let avg_error = errors[i];
            if avg_error < min_error {
                min_error = avg_error;
                min_index = i;
            }
        }
        if tmax - tmin <= tstepmin {
            return intercept_errors.remove(min_index);
        }
        tmax = intercept_errors[min_index].burn_time + tstep;
        tmin = intercept_errors[min_index].burn_time - tstep;
        dbg!(intercept_errors[min_index].burn_time);
        tstep /= tstepdiv;
    }
}

async fn montecarlo_async<const COUNT: usize>(
    intercept: PossibleIntercept,
    normal: Normal<f64>,
    mu: f64,
) -> (InterceptError<COUNT>, f64) {
    let error_map = 0..COUNT;
    let unit_dv = intercept.dv / intercept.dv.norm();
    let error = error_map
        .map(|_| unit_dv * normal.sample(&mut rand::rng()))
        .collect::<Vec<Vector3<f64>>>();
    let intercept_error = InterceptError::from_intercept_v(&intercept, error.try_into().unwrap());
    let avg_error = intercept_error.dx(mu).iter().map(|x| x.norm()).sum::<f64>() / COUNT as f64;
    (intercept_error, avg_error)
}
