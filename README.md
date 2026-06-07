# genetic-search

A genetic algorithm engine in Rust with zero external dependencies.

## Features

- **Selection**: Tournament, roulette wheel, and rank-based selection
- **Crossover**: Single-point, uniform, and blend (BLX-alpha) crossover
- **Mutation**: Gaussian, uniform, and perturbation mutation operators
- **Elitism**: Configurable number of elite individuals preserved per generation
- **Convergence**: Stall detection and diversity-based convergence detection
- **Fitness Tracking**: Best individual tracking across generations

## Usage

```rust
use genetic_search::{
    SimpleRng, GeneticAlgorithm, TournamentSelection,
    SinglePointCrossover, GaussianMutation,
};

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

println!("Best fitness: {}", result.fitness);
```

## License

MIT
