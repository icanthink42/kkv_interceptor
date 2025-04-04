use std::f64::consts::PI;

use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{ComplexField, Matrix3, Owned, Vector, Vector3, U3};

use crate::kepler_orbit::Orbit;

pub struct LambertSolver {
    pub v: Vector3<f64>,
    r1: Vector3<f64>,
    r2: Vector3<f64>,
    pub dt: f64,
    mu: f64,
}

impl LeastSquaresProblem<f64, U3, U3> for LambertSolver {
    type ResidualStorage = Owned<f64, U3>;
    type JacobianStorage = Owned<f64, U3, U3>;
    type ParameterStorage = Owned<f64, U3>;
    fn residuals(&self) -> Option<nalgebra::Vector<f64, U3, Self::ResidualStorage>> {
        let [a, alpha, beta] = [self.v.x, self.v.y, self.v.z];

        let c = (self.r1 - self.r2).norm();
        let s = (self.r1.norm() + self.r2.norm() + c) / 2.0;

        let eq1 =
            -self.dt + (a.powi(3) / self.mu).sqrt() * ((alpha - beta) - (alpha.sin() - beta.sin()));
        let eq2 = -(alpha / 2.0).sin() + (s / (2.0 * a)).sqrt();
        let eq3 = -(beta / 2.0).sin() + ((s - c) / (2.0 * a)).sqrt();

        Some(Vector3::new(eq1, eq2, eq3))
    }
    fn jacobian(&self) -> Option<nalgebra::Matrix<f64, U3, U3, Self::JacobianStorage>> {
        let [a, alpha, beta] = [self.v.x, self.v.y, self.v.z];

        let c = (self.r1 - self.r2).norm();
        let s = (self.r1.norm() + self.r2.norm() + c) / 2.0;

        let m11 = (3.0 * (a.powi(3) / self.mu).sqrt() * (alpha - beta - alpha.sin() + beta.sin()))
            / (2.0 * a);
        let m21 = -2.0.sqrt() * (s / a).sqrt() / (4.0 * a);
        let m31 = -2.0.sqrt() * ((-c + s) / a).sqrt() / (4.0 * a);
        let m12 = (a.powi(3) / self.mu).sqrt() * (1.0 - alpha.cos());
        let m22 = -(alpha / 2.0).cos() / 2.0;
        let m32 = 0.0;
        let m13 = (a.powi(3) / self.mu).sqrt() * (beta.cos() - 1.0);
        let m23 = 0.0;
        let m33 = -(beta / 2.0).cos() / 2.0;

        Some(Matrix3::new(m11, m12, m13, m21, m22, m23, m31, m32, m33))
    }
    fn set_params(&mut self, x: &nalgebra::Vector<f64, U3, Self::ParameterStorage>) {
        self.v.copy_from(x)
    }
    fn params(&self) -> nalgebra::Vector<f64, U3, Self::ParameterStorage> {
        self.v
    }
}

impl LambertSolver {
    pub fn new(v: Vector3<f64>, r1: Vector3<f64>, r2: Vector3<f64>, dt: f64, mu: f64) -> Self {
        Self { v, r1, r2, dt, mu }
    }
}

pub fn calculate_transfers(
    r1: Vector3<f64>,
    r2: Vector3<f64>,
    t0: f64,
    tf: f64,
    tstep: f64,
    mu: f64,
) -> (Vec<f64>, Vec<f64>) {
    let transfer_orbit = Orbit::new_ap_pe(r1.norm(), r2.norm(), mu);

    let guess = Vector3::new((r1.norm() + r2.norm()) / 2.0, PI / 2.0, PI / 2.0);
    let lm = LevenbergMarquardt::new();
    let tguess = transfer_orbit.period / 2.0;
    let mut dt = tguess;
    let mut solver = LambertSolver::new(guess, r1, r2, dt, mu);

    let mut times1 = vec![];
    let mut a1 = vec![];

    while dt < tf {
        solver.dt = dt;
        let (updated_solver, report) = lm.minimize(solver);
        solver = updated_solver;
        if report.number_of_evaluations > 15 {
            break;
        }
        times1.push(dt);
        a1.push(solver.v.x);
        dt += tstep;
    }

    let mut times2 = vec![];
    let mut a2 = vec![];
    dt = tguess;
    while dt > t0 {
        solver.dt = dt;
        let (updated_solver, report) = lm.minimize(solver);
        solver = updated_solver;
        if report.number_of_evaluations > 15 {
            break;
        }
        times2.push(dt);
        a2.push(solver.v.x);
        dt -= tstep;
    }
    times2.reverse();
    a2.reverse();
    times2.append(&mut times1);
    a2.append(&mut a1);

    (times2, a2)
}

pub fn solve_velocities(
    r1: Vector3<f64>,
    r2: Vector3<f64>,
    a: f64,
    alpha: f64,
    beta: f64,
    mu: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let r_c = r2 - r1;

    let z = (mu / (4.0 * a)).sqrt() / (beta / 2.0).tan();
    let y = (mu / (4.0 * a)).sqrt() / (alpha / 2.0).tan();

    let v1 = (z - y) * r_c + (z - y) * r1 / r1.norm();
    let v2 = (z - y) * r_c + (z - y) * r2 / r2.norm();

    (v1, v2)
}
