//! # Genetic Search
//!
//! A genetic algorithm engine with selection, crossover, mutation, elitism,
//! fitness tracking, and convergence detection. Zero external dependencies.
//!
//! # Example
//! ```
//! use genetic_search::{
//!     SimpleRng, GeneticAlgorithm, TournamentSelection,
//!     SinglePointCrossover, GaussianMutation, StallDetector,
//! };
//!
//! let mut rng = SimpleRng::new(42);
//! let mut ga = GeneticAlgorithm::new(50, 5, -10.0, 10.0);
//! let result = ga.run(
//!     &mut rng,
//!     |genes| -genes.iter().map(|g| g * g).sum::<f64>(),
//!     &TournamentSelection::new(3),
//!     &SinglePointCrossover,
//!     &GaussianMutation { rate: 0.3, sigma: 0.5 },
//!     0.8,
//!     100,
//! );
//! println!("Best fitness: {}", result.fitness);
//! ```

pub mod selection;
pub mod crossover;
pub mod mutation;
pub mod population;
pub mod convergence;
mod rng;

pub use rng::SimpleRng;
pub use selection::{Selection, TournamentSelection, RouletteSelection, RankSelection, Individual};
pub use crossover::{Crossover, SinglePointCrossover, UniformCrossover, BlendCrossover};
pub use mutation::{Mutation, GaussianMutation, UniformMutation, PerturbationMutation};
pub use population::Population;
pub use convergence::{ConvergenceDetector, StallDetector, DiversityDetector};

/// Main genetic algorithm runner.
pub struct GeneticAlgorithm {
    pop_size: usize,
    gene_len: usize,
    gene_min: f64,
    gene_max: f64,
    elitism_count: usize,
}

impl GeneticAlgorithm {
    pub fn new(pop_size: usize, gene_len: usize, gene_min: f64, gene_max: f64) -> Self {
        Self { pop_size, gene_len, gene_min, gene_max, elitism_count: 2 }
    }

    pub fn with_elitism(mut self, count: usize) -> Self {
        self.elitism_count = count;
        self
    }

    /// Run the genetic algorithm.
    #[allow(clippy::too_many_arguments)]
    pub fn run<S: Selection, C: Crossover, M: Mutation>(
        &mut self,
        rng: &mut SimpleRng,
        fitness_fn: impl Fn(&[f64]) -> f64,
        selection: &S,
        crossover: &C,
        mutation: &M,
        crossover_rate: f64,
        max_generations: usize,
    ) -> Individual {
        let mut pop = Population::random(self.pop_size, self.gene_len, self.gene_min, self.gene_max, rng);
        pop.elitism_count = self.elitism_count;
        pop.evaluate(&fitness_fn);

        for _gen in 0..max_generations {
            pop = pop.evolve(selection, crossover, mutation, crossover_rate, rng);
            pop.evaluate(&fitness_fn);
        }

        pop.best().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ga_finds_reasonable_solution() {
        let mut rng = SimpleRng::new(42);
        let mut ga = GeneticAlgorithm::new(50, 3, -5.0, 5.0);
        let result = ga.run(
            &mut rng,
            |genes| -genes.iter().map(|g| g * g).sum::<f64>(),
            &TournamentSelection::new(3),
            &SinglePointCrossover,
            &GaussianMutation { rate: 0.3, sigma: 0.5 },
            0.8,
            200,
        );
        assert!(result.fitness > -5.0, "should find near-zero solution, got {}", result.fitness);
    }

    #[test]
    fn ga_with_uniform_crossover() {
        let mut rng = SimpleRng::new(42);
        let mut ga = GeneticAlgorithm::new(30, 2, -5.0, 5.0);
        let result = ga.run(
            &mut rng,
            |genes| -(genes[0] - 3.0).powi(2),
            &TournamentSelection::new(2),
            &UniformCrossover { swap_prob: 0.5 },
            &PerturbationMutation { rate: 0.2, delta: 0.5 },
            0.9,
            100,
        );
        assert!(result.fitness > -5.0);
    }

    #[test]
    fn ga_with_blend_crossover() {
        let mut rng = SimpleRng::new(42);
        let mut ga = GeneticAlgorithm::new(30, 2, -5.0, 5.0);
        let result = ga.run(
            &mut rng,
            |genes| -(genes[0] + genes[1]),
            &TournamentSelection::new(3),
            &BlendCrossover::new(0.5),
            &UniformMutation { rate: 0.1, min: -5.0, max: 0.0 },
            0.8,
            100,
        );
        assert!(result.fitness > -10.0);
    }
}
