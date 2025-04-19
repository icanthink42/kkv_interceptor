use std::time::Instant;

use kepler_orbit::Orbit;
use montecarlo::minimize_x_error;
use nalgebra::Vector3;

mod kepler_orbit;
mod lambert_solver;
mod montecarlo;
mod orbit_propagator;

#[tokio::main]
async fn main() {
    let start = Instant::now();

    let max_dv = 0.0;
    let kill_v = 3.0e3;
    let mu = 398600e9;
    let tstep = 100.0;
    let tstepmin = 1.0;
    let tstepdiv = 10.0;
    const ITER: usize = 10;

    let r1 = Vector3::new(3500.0e3, 6805.0e3, 2200.0e3);
    let v1 = Vector3::new(-7.511e3, 0.357e3, 4.447e3);
    let kkv = Orbit::new(r1, v1);

    let r2 = Vector3::new(-4069.5e3, 2861.786e3, 4483.608e3);
    let v2 = Vector3::new(-5.114e3, -5.691e3, -1.0e3);
    let target = Orbit::new(r2, v2);

    let intercept = minimize_x_error::<ITER>(
        kkv, target, tstep, tstepmin, tstepdiv, None, max_dv, kill_v, mu,
    )
    .await;

    dbg!(intercept.dv.norm());
    dbg!(intercept.burn_time);
    dbg!(intercept.dx(mu).iter().map(|x| x.norm()).sum::<f64>() / ITER as f64);

    let time = start.elapsed();
    println!("{:.2}ms", time.as_millis());

    //let mut plot = Plot::new();
    //let trace = Scatter::new(dv, time_vec);

    //plot.add_trace(trace);

    //plot.write_html("out.html");
}
