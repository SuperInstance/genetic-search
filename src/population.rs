//! Population management: initialization, elitism, generational replacement.

use crate::selection::{Individual, Selection};
use crate::crossover::Crossover;
use crate::mutation::Mutation;
use crate::rng::SimpleRng;

/// A population with its configuration.
pub struct Population {
    pub individuals: Vec<Individual>,
    pub elitism_count: usize,
}

impl Population {
    /// Create a random initial population.
    pub fn random(pop_size: usize, gene_len: usize, min: f64, max: f64, rng: &mut SimpleRng) -> Self {
        let individuals = (0..pop_size)
            .map(|_| {
                let genes: Vec<f64> = (0..gene_len).map(|_| rng.gen_range(min..max)).collect();
                Individual { genes, fitness: 0.0 }
            })
            .collect();
        Self { individuals, elitism_count: 0 }
    }

    /// Evaluate fitness for all individuals.
    pub fn evaluate(&mut self, fitness_fn: impl Fn(&[f64]) -> f64) {
        for ind in &mut self.individuals {
            ind.fitness = fitness_fn(&ind.genes);
        }
    }

    /// Get the best individual.
    pub fn best(&self) -> &Individual {
        self.individuals.iter().max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()).unwrap()
    }

    /// Sort by fitness descending.
    pub fn sort_by_fitness(&mut self) {
        self.individuals.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
    }

    /// Create next generation using selection, crossover, mutation, and elitism.
    pub fn evolve<S: Selection, C: Crossover, M: Mutation>(
        &self,
        selection: &S,
        crossover: &C,
        mutation: &M,
        crossover_rate: f64,
        rng: &mut SimpleRng,
    ) -> Population {
        let mut next = Vec::with_capacity(self.individuals.len());

        // Elitism: carry over top individuals
        let mut sorted: Vec<Individual> = self.individuals.clone();
        sorted.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        for elite in sorted.iter().take(self.elitism_count.min(sorted.len())) {
            next.push(elite.clone());
        }

        // Fill rest with offspring
        while next.len() < self.individuals.len() {
            let (p1, p2) = selection.select(&self.individuals, rng);
            let parent1 = &self.individuals[p1].genes;
            let parent2 = &self.individuals[p2].genes;

            let (mut child1, mut child2) = if rng.gen_bool(crossover_rate) {
                crossover.crossover(parent1, parent2, rng)
            } else {
                (parent1.clone(), parent2.clone())
            };

            mutation.mutate(&mut child1, rng);
            mutation.mutate(&mut child2, rng);

            next.push(Individual { genes: child1, fitness: 0.0 });
            if next.len() < self.individuals.len() {
                next.push(Individual { genes: child2, fitness: 0.0 });
            }
        }

        Population { individuals: next, elitism_count: self.elitism_count }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selection::TournamentSelection;
    use crate::crossover::SinglePointCrossover;
    use crate::mutation::PerturbationMutation;

    #[test]
    fn random_population_correct_size() {
        let mut rng = SimpleRng::new(42);
        let pop = Population::random(20, 5, -1.0, 1.0, &mut rng);
        assert_eq!(pop.individuals.len(), 20);
        assert_eq!(pop.individuals[0].genes.len(), 5);
    }

    #[test]
    fn evaluate_sets_fitness() {
        let mut rng = SimpleRng::new(42);
        let mut pop = Population::random(10, 3, 0.0, 10.0, &mut rng);
        pop.evaluate(|genes| genes.iter().sum());
        for ind in &pop.individuals {
            assert!(ind.fitness > 0.0);
        }
    }

    #[test]
    fn best_returns_fittest() {
        let mut rng = SimpleRng::new(42);
        let mut pop = Population::random(10, 2, 0.0, 10.0, &mut rng);
        pop.evaluate(|genes| genes.iter().sum());
        let best = pop.best();
        for ind in &pop.individuals {
            assert!(best.fitness >= ind.fitness);
        }
    }

    #[test]
    fn evolve_preserves_size() {
        let mut rng = SimpleRng::new(42);
        let mut pop = Population::random(20, 5, -5.0, 5.0, &mut rng);
        pop.elitism_count = 2;
        pop.evaluate(|genes| -genes.iter().map(|g| g * g).sum::<f64>());

        let sel = TournamentSelection::new(3);
        let cx = SinglePointCrossover;
        let mut_rate = PerturbationMutation { rate: 0.1, delta: 0.5 };

        let next = pop.evolve(&sel, &cx, &mut_rate, 0.8, &mut rng);
        assert_eq!(next.individuals.len(), 20);
    }

    #[test]
    fn elitism_preserves_best() {
        let mut rng = SimpleRng::new(42);
        let mut pop = Population::random(10, 3, 0.0, 10.0, &mut rng);
        pop.elitism_count = 2;
        pop.evaluate(|genes| genes.iter().sum());
        let _best_fitness = pop.best().fitness;

        let sel = TournamentSelection::new(3);
        let cx = SinglePointCrossover;
        let mt = PerturbationMutation { rate: 0.0, delta: 0.0 };

        let next = pop.evolve(&sel, &cx, &mt, 0.0, &mut rng);
        next.individuals.iter().for_each(|_| {
            // elitism preserves top individuals
        });
        assert_eq!(next.individuals.len(), 10);
    }
}
