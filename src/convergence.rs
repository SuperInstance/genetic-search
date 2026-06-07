//! Convergence detection: stall detection, diversity metrics.

use crate::population::Population;

/// Trait for convergence detection.
pub trait ConvergenceDetector {
    fn is_converged(&self, population: &Population, generation: usize) -> bool;
}

/// Detects convergence when best fitness hasn't improved for `stall_generations`.
#[derive(Clone)]
pub struct StallDetector {
    pub stall_generations: usize,
    pub tolerance: f64,
    pub last_best: f64,
    pub stall_count: usize,
}

impl StallDetector {
    pub fn new(stall_generations: usize, tolerance: f64) -> Self {
        Self { stall_generations, tolerance, last_best: f64::NEG_INFINITY, stall_count: 0 }
    }
}

impl ConvergenceDetector for StallDetector {
    fn is_converged(&self, _population: &Population, _generation: usize) -> bool {
        // Use the checked version instead
        false
    }
}

impl StallDetector {
    /// Check and update convergence state. Returns true if converged.
    pub fn check(&mut self, best_fitness: f64) -> bool {
        if (best_fitness - self.last_best).abs() < self.tolerance {
            self.stall_count += 1;
        } else {
            self.stall_count = 0;
        }
        self.last_best = best_fitness;
        self.stall_count >= self.stall_generations
    }
}

/// Detects convergence based on population diversity (standard deviation of genes).
#[derive(Clone)]
pub struct DiversityDetector {
    pub min_diversity: f64,
}

impl DiversityDetector {
    pub fn new(min_diversity: f64) -> Self {
        Self { min_diversity }
    }

    /// Compute gene-level standard deviation across the population.
    pub fn diversity(&self, population: &Population) -> f64 {
        if population.individuals.is_empty() {
            return 0.0;
        }
        let n = population.individuals.len();
        let gene_len = population.individuals[0].genes.len();
        let mut total_std = 0.0;
        for g in 0..gene_len {
            let mean: f64 = population.individuals.iter().map(|i| i.genes[g]).sum::<f64>() / n as f64;
            let var: f64 = population.individuals.iter()
                .map(|i| (i.genes[g] - mean).powi(2))
                .sum::<f64>() / n as f64;
            total_std += var.sqrt();
        }
        total_std / gene_len as f64
    }
}

impl ConvergenceDetector for DiversityDetector {
    fn is_converged(&self, population: &Population, _generation: usize) -> bool {
        self.diversity(population) < self.min_diversity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selection::Individual;

    fn make_uniform_pop() -> Vec<Individual> {
        (0..10).map(|_| Individual { genes: vec![5.0, 5.0], fitness: 1.0 }).collect()
    }

    fn make_diverse_pop() -> Vec<Individual> {
        (0..10).map(|i| Individual { genes: vec![i as f64], fitness: i as f64 }).collect()
    }

    #[test]
    fn stall_not_converged_initially() {
        let mut det = StallDetector::new(5, 0.001);
        assert!(!det.check(10.0));
    }

    #[test]
    fn stall_converges_after_stall() {
        let mut det = StallDetector::new(3, 0.001);
        det.check(10.0);
        det.check(10.0);
        det.check(10.0);
        assert!(det.check(10.0));
    }

    #[test]
    fn stall_resets_on_improvement() {
        let mut det = StallDetector::new(3, 0.001);
        det.check(10.0);
        det.check(10.0);
        det.check(20.0); // improvement
        assert_eq!(det.stall_count, 0);
    }

    #[test]
    fn diversity_uniform_is_low() {
        let det = DiversityDetector::new(0.1);
        let pop = Population { individuals: make_uniform_pop(), elitism_count: 0 };
        assert!(det.diversity(&pop) < 0.1);
    }

    #[test]
    fn diversity_diverse_is_high() {
        let det = DiversityDetector::new(0.1);
        let pop = Population { individuals: make_diverse_pop(), elitism_count: 0 };
        assert!(det.diversity(&pop) > 1.0);
    }

    #[test]
    fn diversity_detects_convergence() {
        let det = DiversityDetector::new(0.1);
        let pop = Population { individuals: make_uniform_pop(), elitism_count: 0 };
        assert!(det.is_converged(&pop, 0));
    }

    #[test]
    fn diversity_not_converged_when_diverse() {
        let det = DiversityDetector::new(0.1);
        let pop = Population { individuals: make_diverse_pop(), elitism_count: 0 };
        assert!(!det.is_converged(&pop, 0));
    }
}
