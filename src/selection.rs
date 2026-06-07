//! Selection strategies: tournament, roulette wheel, rank-based.

use crate::rng::SimpleRng;

/// An individual with fitness.
#[derive(Clone, Debug)]
pub struct Individual {
    pub genes: Vec<f64>,
    pub fitness: f64,
}

/// Trait for selection operators.
pub trait Selection {
    /// Select two parents from the population.
    fn select(&self, population: &[Individual], rng: &mut SimpleRng) -> (usize, usize);
}

/// Tournament selection: pick `tournament_size` random individuals and choose the best.
#[derive(Clone)]
pub struct TournamentSelection {
    pub tournament_size: usize,
}

impl TournamentSelection {
    pub fn new(tournament_size: usize) -> Self {
        assert!(tournament_size > 0, "tournament size must be positive");
        Self { tournament_size }
    }
}

impl Selection for TournamentSelection {
    fn select(&self, population: &[Individual], rng: &mut SimpleRng) -> (usize, usize) {
        let pick_best = |rng: &mut SimpleRng| -> usize {
            let mut best_idx = rng.gen_range_usize(0..population.len());
            for _ in 1..self.tournament_size {
                let idx = rng.gen_range_usize(0..population.len());
                if population[idx].fitness > population[best_idx].fitness {
                    best_idx = idx;
                }
            }
            best_idx
        };
        (pick_best(rng), pick_best(rng))
    }
}

/// Roulette wheel (fitness-proportionate) selection.
#[derive(Clone)]
pub struct RouletteSelection;

impl Selection for RouletteSelection {
    fn select(&self, population: &[Individual], rng: &mut SimpleRng) -> (usize, usize) {
        let total_fitness: f64 = population.iter().map(|i| i.fitness).sum();
        let pick = |rng: &mut SimpleRng| -> usize {
            let mut threshold = rng.gen_f64() * total_fitness;
            for (idx, ind) in population.iter().enumerate() {
                threshold -= ind.fitness;
                if threshold <= 0.0 {
                    return idx;
                }
            }
            population.len() - 1
        };
        (pick(rng), pick(rng))
    }
}

/// Rank-based selection.
#[derive(Clone)]
pub struct RankSelection;

impl Selection for RankSelection {
    fn select(&self, population: &[Individual], rng: &mut SimpleRng) -> (usize, usize) {
        let n = population.len();
        let total_rank: f64 = (n * (n + 1)) as f64 / 2.0;
        // Sort indices by fitness (ascending), then rank 1..n
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|a, b| population[*a].fitness.partial_cmp(&population[*b].fitness).unwrap());
        let rank_of = {
            let mut ro = vec![0usize; n];
            for (rank, &idx) in indices.iter().enumerate() {
                ro[idx] = rank + 1;
            }
            ro
        };

        let pick = |rng: &mut SimpleRng| -> usize {
            let mut threshold = rng.gen_f64() * total_rank;
            for (idx, item) in rank_of.iter().enumerate().take(n) {
                threshold -= *item as f64;
                if threshold <= 0.0 {
                    return idx;
                }
            }
            n - 1
        };
        (pick(rng), pick(rng))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pop() -> Vec<Individual> {
        (0..10).map(|i| Individual { genes: vec![i as f64], fitness: i as f64 + 1.0 }).collect()
    }

    #[test]
    fn tournament_selects_from_population() {
        let mut rng = SimpleRng::new(42);
        let pop = make_pop();
        let sel = TournamentSelection::new(3);
        let (a, b) = sel.select(&pop, &mut rng);
        assert!(a < 10 && b < 10);
    }

    #[test]
    fn roulette_selects_from_population() {
        let mut rng = SimpleRng::new(42);
        let pop = make_pop();
        let (a, b) = RouletteSelection.select(&pop, &mut rng);
        assert!(a < 10 && b < 10);
    }

    #[test]
    fn rank_selects_from_population() {
        let mut rng = SimpleRng::new(42);
        let pop = make_pop();
        let (a, b) = RankSelection.select(&pop, &mut rng);
        assert!(a < 10 && b < 10);
    }

    #[test]
    fn tournament_favors_fitter() {
        let mut rng = SimpleRng::new(42);
        let pop = make_pop();
        let sel = TournamentSelection::new(5);
        let mut high_count = 0;
        for _ in 0..100 {
            let (a, _) = sel.select(&pop, &mut rng);
            if pop[a].fitness > 5.0 { high_count += 1; }
        }
        assert!(high_count > 30, "tournament should favor fitter, got {} high picks", high_count);
    }

    #[test]
    #[should_panic(expected = "tournament size must be positive")]
    fn tournament_panics_zero() {
        TournamentSelection::new(0);
    }
}
