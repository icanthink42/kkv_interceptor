use std::time::Instant;

use kepler_orbit::Orbit;
use nalgebra::Vector3;
use orbit_propagator::calculate_transfers;
use plotly::{Plot, Scatter};

mod kepler_orbit;
mod lambert_solver;
mod orbit_propagator;

fn main() {
    let start = Instant::now();

    let max_dv = 0.0;
    let mu = 398600e9;
    let tstep = 1.0;

    let r1 = Vector3::new(3500.0e3, 6805.0e3, 2200.0e3);
    let v1 = Vector3::new(-7.511e3, 0.357e3, 4.447e3);
    let kkv = Orbit::new(r1, v1);

    let r2 = Vector3::new(-4069.5e3, 2861.786e3, 4483.608e3);
    let v2 = Vector3::new(-5.114e3, -5.691e3, -1.0e3);
    let target = Orbit::new(r2, v2);

    let mut dv = vec![];
    let (time_vec, v1_vec) = calculate_transfers(&kkv, &target, tstep, None, max_dv, mu);
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
