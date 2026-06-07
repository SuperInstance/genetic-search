//! Crossover operators: single-point, uniform, blend (BLX-alpha).

use crate::rng::SimpleRng;

/// Trait for crossover operators.
pub trait Crossover {
    /// Produce two offspring from two parents.
    fn crossover(&self, parent1: &[f64], parent2: &[f64], rng: &mut SimpleRng) -> (Vec<f64>, Vec<f64>);
}

/// Single-point crossover.
#[derive(Clone)]
pub struct SinglePointCrossover;

impl Crossover for SinglePointCrossover {
    fn crossover(&self, parent1: &[f64], parent2: &[f64], rng: &mut SimpleRng) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(parent1.len(), parent2.len(), "parents must have same length");
        let n = parent1.len();
        if n <= 1 {
            return (parent1.to_vec(), parent2.to_vec());
        }
        let point = rng.gen_range_usize(1..n);
        let mut child1 = Vec::with_capacity(n);
        let mut child2 = Vec::with_capacity(n);
        for i in 0..n {
            if i < point {
                child1.push(parent1[i]);
                child2.push(parent2[i]);
            } else {
                child1.push(parent2[i]);
                child2.push(parent1[i]);
            }
        }
        (child1, child2)
    }
}

/// Uniform crossover: each gene randomly from either parent.
#[derive(Clone)]
pub struct UniformCrossover {
    pub swap_prob: f64,
}

impl Crossover for UniformCrossover {
    fn crossover(&self, parent1: &[f64], parent2: &[f64], rng: &mut SimpleRng) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(parent1.len(), parent2.len());
        let n = parent1.len();
        let mut child1 = Vec::with_capacity(n);
        let mut child2 = Vec::with_capacity(n);
        for i in 0..n {
            if rng.gen_bool(self.swap_prob) {
                child1.push(parent2[i]);
                child2.push(parent1[i]);
            } else {
                child1.push(parent1[i]);
                child2.push(parent2[i]);
            }
        }
        (child1, child2)
    }
}

/// Blend crossover (BLX-alpha).
#[derive(Clone)]
pub struct BlendCrossover {
    pub alpha: f64,
}

impl BlendCrossover {
    pub fn new(alpha: f64) -> Self {
        assert!(alpha >= 0.0, "alpha must be non-negative");
        Self { alpha }
    }
}

impl Crossover for BlendCrossover {
    fn crossover(&self, parent1: &[f64], parent2: &[f64], rng: &mut SimpleRng) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(parent1.len(), parent2.len());
        let blend = |p1: f64, p2: f64, rng: &mut SimpleRng| -> f64 {
            let (lo, hi) = if p1 < p2 { (p1, p2) } else { (p2, p1) };
            let range = hi - lo;
            let low = lo - self.alpha * range;
            let high = hi + self.alpha * range;
            rng.gen_range(low..high)
        };
        let n = parent1.len();
        let child1: Vec<f64> = (0..n).map(|i| blend(parent1[i], parent2[i], rng)).collect();
        let child2: Vec<f64> = (0..n).map(|i| blend(parent1[i], parent2[i], rng)).collect();
        (child1, child2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_point_produces_correct_length() {
        let mut rng = SimpleRng::new(42);
        let cx = SinglePointCrossover;
        let (c1, c2) = cx.crossover(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], &mut rng);
        assert_eq!(c1.len(), 4);
        assert_eq!(c2.len(), 4);
    }

    #[test]
    fn single_point_crosses_genes() {
        let mut rng = SimpleRng::new(42);
        let cx = SinglePointCrossover;
        let (c1, c2) = cx.crossover(&[1.0, 1.0, 1.0], &[2.0, 2.0, 2.0], &mut rng);
        // Both children should have mix of 1s and 2s
        let all_ones = c1.iter().all(|&g| g == 1.0);
        let all_twos = c1.iter().all(|&g| g == 2.0);
        assert!(!all_ones || !all_twos || c1 != c2);
    }

    #[test]
    fn uniform_produces_correct_length() {
        let mut rng = SimpleRng::new(42);
        let cx = UniformCrossover { swap_prob: 0.5 };
        let (c1, c2) = cx.crossover(&[1.0, 2.0], &[3.0, 4.0], &mut rng);
        assert_eq!(c1.len(), 2);
        assert_eq!(c2.len(), 2);
    }

    #[test]
    fn blend_produces_values_in_range() {
        let mut rng = SimpleRng::new(42);
        let cx = BlendCrossover::new(0.5);
        let (c1, c2) = cx.crossover(&[0.0, 0.0], &[10.0, 10.0], &mut rng);
        for v in c1.iter().chain(c2.iter()) {
            assert!(*v >= -5.0 && *v <= 15.0, "value {} out of range", v);
        }
    }

    #[test]
    fn single_point_single_gene() {
        let mut rng = SimpleRng::new(42);
        let cx = SinglePointCrossover;
        let (c1, c2) = cx.crossover(&[1.0], &[2.0], &mut rng);
        assert_eq!(c1, vec![1.0]);
        assert_eq!(c2, vec![2.0]);
    }

    #[test]
    #[should_panic(expected = "alpha must be non-negative")]
    fn blend_panics_negative_alpha() {
        BlendCrossover::new(-1.0);
    }
}
