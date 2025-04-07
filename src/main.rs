use std::time::Instant;

use kepler_orbit::Orbit;
use nalgebra::Vector3;
use orbit_propagator::{calculate_transfers, propagate, InterceptError};
use plotly::{Plot, Scatter};

mod kepler_orbit;
mod lambert_solver;
mod orbit_propagator;

#[tokio::main]
async fn main() {
    let start = Instant::now();

    let max_dv = 0.0;
    let mu = 398600e9;
    let tstep = 100.0;
    let tmin = 0.1;

    let r1 = Vector3::new(3500.0e3, 6805.0e3, 2200.0e3);
    let v1 = Vector3::new(-7.511e3, 0.357e3, 4.447e3);
    let kkv = Orbit::new(r1, v1);

    let r2 = Vector3::new(-4069.5e3, 2861.786e3, 4483.608e3);
    let v2 = Vector3::new(-5.114e3, -5.691e3, -1.0e3);
    let target = Orbit::new(r2, v2);

    let intercepts = propagate(kkv, target, tstep, None, max_dv, mu).await;
    let mut intercept_errors: Vec<InterceptError<100>> = vec![];
    for intercept in intercepts {
        let error_map = 0..100;
        let unit_dv = intercept.dv / intercept.dv.norm();
        let error = error_map.map(|_| unit_dv).collect::<Vec<Vector3<f64>>>();
        intercept_errors.push(InterceptError::from_intercept_v(
            &intercept,
            error.try_into().unwrap(),
        ));
    }

    let time = start.elapsed();
    println!("{:.2}ms", time.as_millis());

    //let mut plot = Plot::new();
    //let trace = Scatter::new(dv, time_vec);

    //plot.add_trace(trace);

    //plot.write_html("out.html");
}
