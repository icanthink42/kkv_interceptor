use std::f64::{consts::PI, INFINITY};

use nalgebra::Vector3;

pub struct Orbit {
    pub mu: f64,
    pub energy: f64,
    pub h: f64,
    pub e: f64,
    pub pe: f64,
    pub ap: f64,
    pub a: f64,
    pub period: f64,
}

impl Orbit {
    pub fn new(r: Vector3<f64>, v: Vector3<f64>, mu: f64) -> Self {
        let energy = v.norm().powi(2) / 2.0 - mu / r.norm();
        let h = r.cross(&v);
        let e = (v.cross(&h) / mu - r / r.norm()).norm();
        let pe = h.norm().powi(2) / (mu * (1.0 + e));
        let a = h.norm().powi(2) / (mu * (1.0 - e.powi(2)));
        let ap = h.norm().powi(2) / (mu * (1.0 - e));
        let period = if energy > 0.0 {
            INFINITY
        } else {
            2.0 * PI * (a.powi(3) / mu).sqrt()
        };

        Self {
            mu,
            energy,
            h: h.norm(),
            e,
            pe,
            ap,
            a,
            period,
        }
    }

    pub fn new_ap_pe(ap: f64, pe: f64, mu: f64) -> Self {
        let e = (ap - pe) / (ap + pe);
        let a = (ap + pe) / 2.0;
        let h = (a * mu * (1.0 - e.powi(2))).sqrt();
        let energy = (mu / h).powi(2) / 2.0 - mu / pe;
        let period = if energy > 0.0 {
            INFINITY
        } else {
            2.0 * PI * (a.powi(3) / mu).sqrt()
        };
        Self {
            mu,
            energy,
            h,
            e,
            pe,
            ap,
            a,
            period,
        }
    }

    fn hohmann(&self, orbit2: Self) -> Self {
        Orbit::new_ap_pe(self.pe, orbit2.ap, self.mu)
    }
}
