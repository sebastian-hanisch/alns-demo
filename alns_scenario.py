"""Lieferinstanz: ein Depot in der Mitte und n Kundenstopps in einem 100 x 100-km-Gebiet (euklidische Entfernungen).
Wortgleiche Kopie von `vrpn_scenario.py` (VRP-Nachbarschaften-Demo, die CVRP-Infrastruktur dieser Linie) - die
geometrische Basis ist wiederum wortgleich zur Hill-Climbing-Demo (`hc_scenario.generate`). Kapazität >= Summe
aller Bedarfe reduziert die Instanz auf GENAU einen Fahrzeugweg - exakt die TSP-Instanz der Wurzel dieser Linie."""

from dataclasses import dataclass

import numpy as np

import alns_constants as C


@dataclass(frozen=True)
class Instance:
    xy: np.ndarray            # (n + 1, 2); Zeile 0 = Depot
    demands: np.ndarray       # (n + 1,); demands[0] = 0 (Depot)
    capacity: float
    n: int
    cluster_share: int
    seed: int

    @property
    def n_nodes(self):
        return self.n + 1

    @property
    def total_demand(self):
        return float(self.demands.sum())


def generate(n, cluster_share=0, seed=0, capacity=C.DEFAULT_CAPACITY):
    rng = np.random.default_rng(seed)
    n_grouped = int(round(n * cluster_share / 100))
    uniform = rng.random((n - n_grouped, 2)) * C.AREA
    centres = C.CLUSTER_MARGIN + rng.random((C.N_CLUSTERS, 2)) * (C.AREA - 2 * C.CLUSTER_MARGIN)
    which = rng.integers(0, C.N_CLUSTERS, size=n_grouped)
    grouped = np.clip(centres[which] + rng.normal(0.0, C.CLUSTER_SIGMA, size=(n_grouped, 2)), 0.0, C.AREA)
    depot = np.array([[C.AREA / 2, C.AREA / 2]])
    xy = np.vstack([depot, uniform, grouped])
    demands = np.concatenate([[0], rng.integers(C.DEMAND_MIN, C.DEMAND_MAX + 1, size=n)]).astype(np.int64)
    return Instance(xy, demands, float(capacity), n, int(cluster_share), int(seed))
