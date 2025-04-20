use core::f64;
use std::f64::consts::PI;

use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{Matrix3, Owned, Vector1, Vector3, U1};

#[derive(Debug)]
pub struct Orbit {
    pub r: Vector3<f64>,
    pub v: Vector3<f64>,
}

#[allow(dead_code)]
impl Orbit {
    pub fn new(r: Vector3<f64>, v: Vector3<f64>) -> Self {
        Self { r, v }
    }

    pub fn h(&self) -> Vector3<f64> {
        self.r.cross(&self.v)
    }
    pub fn i(&self) -> f64 {
        let h = self.h();
        (h.dot(&Vector3::new(0.0, 0.0, 1.0)) / h.norm()).acos()
    }

    pub fn node_line(&self) -> Vector3<f64> {
        Vector3::new(0.0, 0.0, 1.0).cross(&self.h())
    }

    pub fn e(&self, mu: f64) -> Vector3<f64> {
        self.v.cross(&self.h()) / mu - self.r / self.r.norm()
    }

    pub fn cap_omega(&self) -> f64 {
        let n = self.node_line();
        if n.y >= 0.0 {
            (n.x / n.norm()).acos()
        } else {
            2.0 * PI - (n.x / n.norm()).acos()
        }
    }

    pub fn omega(&self, mu: f64) -> f64 {
        let e = self.e(mu);
        let n = self.node_line();
        if e.z >= 0.0 {
            (n.dot(&e) / (n.norm() * e.norm())).acos()
        } else {
            2.0 * PI - (n.dot(&e) / (n.norm() * e.norm())).acos()
        }
    }

    pub fn theta(&self, mu: f64) -> f64 {
        let e = self.e(mu);
        if e.dot(&self.v) >= 0.0 {
            (e.dot(&self.r) / (e.norm() * self.r.norm())).acos()
        } else {
            2.0 * PI - (e.dot(&self.r) / (e.norm() * self.r.norm())).acos()
        }
    }
    pub fn v_r(&self) -> f64 {
        self.v.dot(&(self.r / self.r.norm()))
    }

    pub fn period(&self, mu: f64) -> f64 {
        (2.0 * PI / mu.powi(2))
            * (self.h().norm() / (1.0 - self.e(mu).norm().powi(2)).sqrt()).powi(3)
    }

    pub fn time_at(&self, theta: f64, mu: f64) -> f64 {
        let e = self.e(mu).norm();
        let ecc = 2.0 * (((1.0 - e) / (1.0 + e)).sqrt() * (theta / 2.0).tan()).atan();
        self.period(mu) / (2.0 * PI) * (ecc - e * ecc.sin())
    }

    pub fn theta_at(&self, time: f64, mu: f64) -> f64 {
        let e = self.e(mu).norm();
        let lm = LevenbergMarquardt::new();
        let solver = EccentricAnomalySolver {
            e,
            period: self.period(mu),
            time,
            v: Vector1::new(PI),
        };
        let (updated_solver, _) = lm.minimize(solver);
        let ecc = updated_solver.v.x;
        2.0 * (((1.0 + e) / (1.0 - e)).sqrt() * (ecc / 2.0).tan()).atan()
    }

    pub fn time(&self, mu: f64) -> f64 {
        self.time_at(self.theta(mu), mu)
    }

    pub fn r_at(&self, theta: f64, mu: f64) -> f64 {
        let h = self.h().norm();
        let e = self.e(mu).norm();
        h.powi(2) / (mu * (1.0 + e * theta.cos()))
    }

    pub fn propagate_time(&self, time: f64, mu: f64) -> Orbit {
        let (r, v) = self.propagate_time_xyz(time, mu);
        Orbit::new(r, v)
    }
    pub fn propagate_time_xyz(&self, time: f64, mu: f64) -> (Vector3<f64>, Vector3<f64>) {
        let h = self.h();
        let e = self.e(mu);

        let p_hat = e / e.norm();
        let w_hat = h / h.norm();
        let q_hat = w_hat.cross(&p_hat);

        let pqw_ijk = Matrix3::from_columns(&[p_hat, q_hat, w_hat]);

        let time0 = self.time(mu);
        let new_time = time0 + time;
        let new_theta = self.theta_at(new_time, mu);
        let r_new = self.r_at(new_theta, mu);

        #[rustfmt::skip]
        let rthetaz_pqw = Matrix3::new(
        new_theta.cos(), -new_theta.sin(), 0.0,
        new_theta.sin(),  new_theta.cos(), 0.0,
             0.0,              0.0,        1.0);

        let r_newpqw = rthetaz_pqw * Vector3::new(r_new, 0.0, 0.0);

        let v_r_new = (mu / h.norm()) * e.norm() * new_theta.sin();
        let v_theta_new = (mu / h.norm()) * (1.0 + e.norm() * new_theta.cos());

        let v_newpqw = rthetaz_pqw * Vector3::new(v_r_new, v_theta_new, 0.0);
        (pqw_ijk * r_newpqw, pqw_ijk * v_newpqw)
    }
    pub fn plot_orbit(&self, n: i32, mu: f64) -> (Vec<f64>, Vec<Vector3<f64>>, Vec<Vector3<f64>>) {
        let period = self.period(mu);
        let dt = period / n as f64;
        let mut t = 0.0;

        let mut times = vec![];
        let mut rs = vec![];
        let mut vs = vec![];

        while t < period {
            let (r, v) = self.propagate_time_xyz(t, mu);
            times.push(t);
            rs.push(r);
            vs.push(v);
            t += dt;
        }
        (times, rs, vs)
    }
}

pub struct EccentricAnomalySolver {
    pub v: Vector1<f64>,
    period: f64,
    time: f64,
    e: f64,
}

impl LeastSquaresProblem<f64, U1, U1> for EccentricAnomalySolver {
    type ResidualStorage = Owned<f64, U1>;
    type JacobianStorage = Owned<f64, U1, U1>;
    type ParameterStorage = Owned<f64, U1>;

    fn set_params(&mut self, x: &nalgebra::Vector<f64, U1, Self::ParameterStorage>) {
        self.v.copy_from(x)
    }
    fn residuals(&self) -> Option<nalgebra::Vector<f64, U1, Self::ResidualStorage>> {
        let ecc = self.v.x;
        Some(Vector1::new(
            2.0 * PI / self.period * self.time + self.e * ecc.sin() - ecc,
        ))
    }
    fn jacobian(&self) -> Option<nalgebra::Matrix<f64, U1, U1, Self::JacobianStorage>> {
        let ecc = self.v.x;
        Some(Vector1::new(self.e * ecc.cos() - 1.0))
    }
    fn params(&self) -> nalgebra::Vector<f64, U1, Self::ParameterStorage> {
        self.v
    }
}
