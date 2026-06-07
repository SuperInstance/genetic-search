//! Mutation operators: Gaussian, uniform, bit-flip style.

use crate::rng::SimpleRng;

/// Trait for mutation operators.
pub trait Mutation {
    fn mutate(&self, genes: &mut [f64], rng: &mut SimpleRng);
}

/// Gaussian mutation: add N(0, sigma) to each gene with probability `rate`.
#[derive(Clone)]
pub struct GaussianMutation {
    pub rate: f64,
    pub sigma: f64,
}

impl Mutation for GaussianMutation {
    fn mutate(&self, genes: &mut [f64], rng: &mut SimpleRng) {
        for gene in genes.iter_mut() {
            if rng.gen_bool(self.rate) {
                // Box-Muller for approximate normal distribution
                let u1 = rng.gen_f64().max(1e-10);
                let u2 = rng.gen_f64();
                let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                *gene += self.sigma * normal;
            }
        }
    }
}

/// Uniform mutation: replace each gene with a random value in range with probability `rate`.
#[derive(Clone)]
pub struct UniformMutation {
    pub rate: f64,
    pub min: f64,
    pub max: f64,
}

impl Mutation for UniformMutation {
    fn mutate(&self, genes: &mut [f64], rng: &mut SimpleRng) {
        for gene in genes.iter_mut() {
            if rng.gen_bool(self.rate) {
                *gene = rng.gen_range(self.min..self.max);
            }
        }
    }
}

/// Perturbation mutation: add small random delta to each gene with probability `rate`.
#[derive(Clone)]
pub struct PerturbationMutation {
    pub rate: f64,
    pub delta: f64,
}

impl Mutation for PerturbationMutation {
    fn mutate(&self, genes: &mut [f64], rng: &mut SimpleRng) {
        for gene in genes.iter_mut() {
            if rng.gen_bool(self.rate) {
                *gene += rng.gen_range(-self.delta..self.delta);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_mutation_can_change_genes() {
        let mut rng = SimpleRng::new(42);
        let mut genes = vec![0.0; 100];
        GaussianMutation { rate: 1.0, sigma: 1.0 }.mutate(&mut genes, &mut rng);
        let changed = genes.iter().filter(|&&g| g != 0.0).count();
        assert!(changed > 50, "many genes should change, got {}", changed);
    }

    #[test]
    fn gaussian_zero_rate_no_change() {
        let mut rng = SimpleRng::new(42);
        let mut genes = vec![1.0, 2.0, 3.0];
        GaussianMutation { rate: 0.0, sigma: 1.0 }.mutate(&mut genes, &mut rng);
        assert_eq!(genes, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn uniform_mutation_respects_range() {
        let mut rng = SimpleRng::new(42);
        let mut genes = vec![0.0; 100];
        UniformMutation { rate: 1.0, min: -5.0, max: 5.0 }.mutate(&mut genes, &mut rng);
        for g in &genes {
            assert!(*g >= -5.0 && *g < 5.0, "gene {} out of range", g);
        }
    }

    #[test]
    fn perturbation_mutation_small_changes() {
        let mut rng = SimpleRng::new(42);
        let original = vec![10.0; 100];
        let mut genes = original.clone();
        PerturbationMutation { rate: 1.0, delta: 0.1 }.mutate(&mut genes, &mut rng);
        for (o, g) in original.iter().zip(&genes) {
            assert!((g - o).abs() <= 0.1 + 1e-10, "change too large: {}", (g - o).abs());
        }
    }

    #[test]
    fn perturbation_zero_rate() {
        let mut rng = SimpleRng::new(42);
        let mut genes = vec![5.0, 5.0];
        PerturbationMutation { rate: 0.0, delta: 1.0 }.mutate(&mut genes, &mut rng);
        assert_eq!(genes, vec![5.0, 5.0]);
    }
}
