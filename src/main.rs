use std::time::Instant;

use lambert_solver::calculate_transfers_inside_dv;
use nalgebra::Vector3;
use plotly::{Plot, Scatter};

mod kepler_orbit;
mod lambert_solver;
mod orbit_propagator;

fn main() {
    let start = Instant::now();

    let max_dv = 5100.0;
    let mu = 398600e9;
    let tstep = 1.0;

    let r1 = Vector3::new(3500.0e3, 6805.0e3, 2200.0e3);
    let v1 = Vector3::new(-7.511e3, 0.357e3, 4.447e3);

    let r2 = Vector3::new(-4069.5e3, 2861.786e3, 4483.608e3);
    let v2 = Vector3::new(-5.114e3, -5.691e3, -1.0e3);

    let mut dv = vec![];
    let (time_vec, v1_vec) = calculate_transfers_inside_dv(r1, v1, r2, max_dv, tstep, mu);
    for v in v1_vec {
        dv.push((v1 - v).norm())
    }

    let time = start.elapsed();
    println!("{:.2}ms", time.as_millis());

    let mut plot = Plot::new();
    let trace = Scatter::new(dv, time_vec);

    plot.add_trace(trace);

    plot.write_html("out.html");
}
