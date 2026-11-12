"""A small evolution strategy with a protected champion.

Search distribution: independent Gaussian per gene (mean, sigma), clipped to
[0, 1]. Each generation:

  ask()   population = [champion, current mean, mean + sigma * noise ...]
  tell()  rank by fitness; the mean moves to the rank-weighted average of the
          best quarter; sigma moves towards their spread (never below a floor).

Elitism ("never lose the best genome"):
  * The champion is re-evaluated every generation on that generation's fresh
    seeds, next to all candidates, so the comparison is paired.
  * A candidate replaces the champion only if it beats the champion's score
    in that same generation.
  * The champion's score on each new generation's seeds is therefore an
    honest (not cherry-picked) measure of progress: those seeds played no part
    in choosing it.

Alternatives considered: CMA-ES (full covariance; better on correlated genes,
but more to get wrong and 54-131 genes with population 64 is already
workable for a diagonal method) and a genetic algorithm with crossover (no
clear benefit for a continuous, low-dimensional genome).
"""
from __future__ import annotations

import numpy as np


class ES:
    def __init__(self, init_mean: np.ndarray, sigma0: float = 0.2, pop: int = 64,
                 elite_frac: float = 0.25, sigma_min: float = 0.02, seed: int = 0):
        self.mean = np.asarray(init_mean, np.float64).copy()
        self.sigma = np.full_like(self.mean, sigma0)
        self.pop = pop
        self.mu = max(2, int(round(pop * elite_frac)))
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.sigma_min = sigma_min
        self.rng = np.random.default_rng(seed)
        self.champion = self.mean.copy()
        self.champion_gen = -1
        self.generation = 0
        self._asked = None

    def ask(self) -> np.ndarray:
        x = self.mean + self.sigma * self.rng.standard_normal((self.pop, len(self.mean)))
        x = np.clip(x, 0.0, 1.0)
        x[0] = self.champion
        x[1] = self.mean
        self._asked = x
        return x

    def tell(self, fit: np.ndarray) -> dict:
        x = self._asked
        order = np.argsort(-fit, kind="stable")
        elite = x[order[: self.mu]]
        new_mean = self.w @ elite
        spread = np.sqrt(self.w @ (elite - new_mean) ** 2)
        self.mean = new_mean
        self.sigma = np.maximum(0.7 * self.sigma + 0.3 * spread, self.sigma_min)
        best = int(order[0])
        replaced = best != 0 and fit[best] > fit[0]
        info = {"champion_fitness": float(fit[0]), "best_fitness": float(fit[best]),
                "mean_fitness": float(fit.mean()), "median_fitness": float(np.median(fit)),
                "sigma_mean": float(self.sigma.mean()), "champion_replaced": bool(replaced)}
        if replaced:
            self.champion = x[best].copy()
            self.champion_gen = self.generation
        self.generation += 1
        return info

    # -- checkpointing ---------------------------------------------------
    def state(self) -> dict:
        return {"mean": self.mean, "sigma": self.sigma, "champion": self.champion,
                "champion_gen": self.champion_gen, "generation": self.generation,
                "rng": np.array([str(self.rng.bit_generator.state)])}

    def load(self, st) -> None:
        import ast
        self.mean, self.sigma, self.champion = st["mean"].copy(), st["sigma"].copy(), st["champion"].copy()
        self.champion_gen, self.generation = int(st["champion_gen"]), int(st["generation"])
        self.rng.bit_generator.state = ast.literal_eval(str(st["rng"][0]))
