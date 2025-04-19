use std::sync::Arc;

use nalgebra::Vector3;

use crate::{
    kepler_orbit::Orbit,
    orbit_propagator::{propagate, InterceptError},
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
    mu: f64,
) -> InterceptError<COUNT> {
    let mut tmax = if let Some(tmax) = tmax {
        tmax
    } else {
        kkv.period(mu).max(target.period(mu))
    };
    let mut tstep = tstep;
    let target = Arc::new(target);
    let mut tmin = 0.0;
    loop {
        dbg!("Completed Iteration");
        let intercepts =
            propagate(&kkv, target.clone(), tstep, tmin, tmax, dv_max, kill_v, mu).await;
        let mut intercept_errors: Vec<InterceptError<COUNT>> = vec![];
        for intercept in intercepts {
            let error_map = 0..COUNT;
            let unit_dv = intercept.dv / intercept.dv.norm();
            let error = error_map.map(|_| unit_dv).collect::<Vec<Vector3<f64>>>();
            intercept_errors.push(InterceptError::from_intercept_v(
                &intercept,
                error.try_into().unwrap(),
            ));
        }
        let mut min_error = 0.0;
        let mut min_index = 0;
        for (i, intercept_error) in intercept_errors.iter().enumerate() {
            println!("{}/{}", i, intercept_errors.len());
            let avg_error =
                intercept_error.dx(mu).iter().map(|x| x.norm()).sum::<f64>() / COUNT as f64;
            if avg_error < min_error {
                min_error = avg_error;
                min_index = i;
            }
        }
        if tmax - tmax <= tstepmin {
            return intercept_errors.remove(min_index);
        }
        tmax = intercept_errors[min_index].burn_time + tstep;
        tmin = intercept_errors[min_index].burn_time - tstep;
        tstep /= tstepdiv;
    }
}
