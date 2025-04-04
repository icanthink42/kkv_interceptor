use std::time::Instant;

use lambert_solver::calculate_transfers;
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
    let r2 = Vector3::new(-4069.5e3, 2861.786e3, 4483.608e3);

    let (times, a) = calculate_transfers(r1, r2, t0, tf, tstep, mu);

    let time = start.elapsed();
    println!("{:.2}ms", time.as_millis());

    let mut plot = Plot::new();
    let trace = Scatter::new(a, times);

    plot.add_trace(trace);

    plot.write_html("out.html");
}
