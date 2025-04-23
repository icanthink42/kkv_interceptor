use std::{fs::OpenOptions, process::Command, time::Instant};

use kepler_orbit::Orbit;
use montecarlo::minimize_x_error;
use nalgebra::Vector3;
use serde_pickle::SerOptions;

mod kepler_orbit;
mod lambert_solver;
mod montecarlo;
mod orbit_propagator;

#[tokio::main]
async fn main() {
    let start = Instant::now();
    let save_file_path = "data.orbit";

    let max_dv = 5500.0;
    let kill_v = 1.0e3;
    let dv_error_mean = 0.0;
    let dv_error_stdev = 1.0;
    let planet_radius = 6378.0e3;

    let mu = 398600e9;
    let tstep = 1000.0;
    let tstepmin = 10.0;
    let tstepdiv = 10.0;
    const ITER: usize = 100;

    let r1 = Vector3::new(-4069.5e3, 2861.786e3, 4483.608e3);
    let v1 = Vector3::new(-5.114e3, -5.691e3, -1.0e3);
    let kkv = Orbit::new(r1, v1);

    let r2 = Vector3::new(3500.0e3, 6805.0e3, 2200.0e3);
    let v2 = Vector3::new(-0.511e3, 0.357e3, 4.447e3);
    let target = Orbit::new(r2, v2);

    let intercept = minimize_x_error::<ITER>(
        kkv.clone(),
        target.clone(),
        tstep,
        tstepmin,
        tstepdiv,
        None,
        max_dv,
        kill_v,
        dv_error_mean,
        dv_error_stdev,
        planet_radius,
        mu,
    )
    .await;

    let dx = intercept.dx(mu).iter().map(|x| x.norm()).sum::<f64>() / ITER as f64;
    println!("*Optimal Interception*");
    println!("Wait Time: {}s", intercept.burn_time);
    println!("Transfer Time: {}s", intercept.dt);
    println!("Delta V: {}m/s", intercept.dv.norm());
    println!("Average Distance: {}", dx);
    println!("Velocity Change Vector: {}m/s", intercept.dv);

    let time = start.elapsed();
    println!("{:.2}s", time.as_secs());
    let (intercept_t, intercept_r, _) = intercept.ideal_orbit.plot_orbit(1000, mu);
    let (kkv_t, kkv_r, _) = kkv.plot_orbit(1000, mu);
    let (target_t, target_r, _) = target.plot_orbit(1000, mu);
    let mut data_file = OpenOptions::new()
        .create(true)
        .write(true)
        .open(save_file_path)
        .unwrap();
    let serde_options = SerOptions::new();
    let serialized_data = (
        intercept_t,
        intercept_r,
        kkv_t,
        kkv_r,
        target_t,
        target_r,
        kkv.propagate_time_xyz(intercept.burn_time, mu).0,
        intercept.ideal_orbit.propagate_time_xyz(intercept.dt, mu).0,
    );
    serde_pickle::to_writer(&mut data_file, &serialized_data, serde_options).unwrap();
    let _ = Command::new("python")
        .arg("plot.py")
        .arg(save_file_path)
        .output()
        .unwrap();
}
