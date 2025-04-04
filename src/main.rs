use std::time::Instant;

use itertools::izip;
use lambert_solver::{calculate_transfers, solve_velocities};
use nalgebra::Vector3;
use plotly::{Plot, Scatter};

mod kepler_orbit;
mod lambert_solver;

fn main() {
    let start = Instant::now();

    let mu = 398600e9;
    let t0 = 15.0 * 60.0;
    let tf = 4.0 * 60.0_f64.powi(2);
    let tstep = 1.0;

    let r1 = Vector3::new(3500.0e3, 6805.0e3, 2200.0e3);
    let v1 = Vector3::new(-7.511e3, 0.357e3, 4.447e3);

    let r2 = Vector3::new(-4069.5e3, 2861.786e3, 4483.608e3);
    let v2 = Vector3::new(-5.114e3, -5.691e3, -1.0e3);

    let (time_vec, a_vec, alpha_vec, beta_vec) = calculate_transfers(r1, r2, t0, tf, tstep, mu);
    let mut dv_vec = Vec::new();
    for it in izip!(&time_vec, &a_vec, &alpha_vec, &beta_vec) {
        let (_, a, alpha, beta) = it;
        let (v1_burn, v2_burn) = solve_velocities(r1, r2, *a, *alpha, *beta, mu);

        let dv = (v2 - v2_burn).norm() + (v1 - v1_burn).norm();
        dv_vec.push(dv);
    }

    let time = start.elapsed();
    println!("{:.2}ms", time.as_millis());

    let mut plot = Plot::new();
    let trace = Scatter::new(dv_vec, time_vec);

    plot.add_trace(trace);

    plot.write_html("out.html");
}
