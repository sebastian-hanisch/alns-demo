"""Auswertung der ALNS-Demo: die zentrale Frage ist der Vergleich ALNS (destroy/repair, `alns_algorithm.alns`)
gegen die kleine Nachbarschaft der Schwester-Demo (Relocate/Swap/2-opt*/CROSS-exchange, `alns_interroute.descend`)
bei GLEICHEM (deklariertem) Bewertungsbudget, von derselben Savings-Konstruktion aus - siehe Modul-Docstring von
`alns_algorithm.py` für die bewusst grobkörnigere Budget-Konvention von ALNS.

ALNS hat (anders als die deterministische Schwester-Demo) wieder Zufall im Kern - wie jedes SA/PT/Tabu/GRASP/ILS-
Stück dieser Linie wird über `SWEEP_CHAINS` Ketten-Seeds je Instanz gemittelt (nur für `method="alns"`; die kleine
Nachbarschaft bleibt deterministisch, ein Ketten-Seed ändert dort nichts)."""

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import alns_algorithm as A
import alns_constants as C
import alns_construction as CO
import alns_interroute as IR
import alns_scenario as S
import alns_tour as T


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    cluster_share: int = C.DEFAULT_BALLUNG
    seed: int = C.DEFAULT_SEED
    capacity: float = C.DEFAULT_CAPACITY
    budget: int = C.DEFAULT_BUDGET
    chain_seed: int = C.DEFAULT_CHAIN_SEED
    method: str = "alns"                              # "alns" oder "small_neighborhood"
    adaptive: bool = C.DEFAULT_ADAPTIVE
    destroy_ops: tuple = C.DESTROY_OPS
    repair_ops: tuple = C.REPAIR_OPS
    k_percent: float = C.DEFAULT_K_PERCENT
    active_moves: tuple = IR.MOVE_TYPES                # nur für method="small_neighborhood"


@lru_cache(maxsize=256)
def instance(n, cluster_share, seed, capacity):
    inst = S.generate(n, cluster_share, seed, capacity=capacity)
    return inst, T.dist_matrix(inst.xy)


@lru_cache(maxsize=256)
def construction(n, cluster_share, seed, capacity):
    inst, D = instance(n, cluster_share, seed, capacity)
    routes = CO.savings_construction(n, D, inst.demands, capacity)
    return tuple(tuple(r.tolist()) for r in routes)


def _routes_array(routes_tuple):
    return [np.array(r, dtype=np.int64) for r in routes_tuple]


@dataclass
class Analysis:
    settings: Settings
    inst: object
    D: np.ndarray
    construction_routes: list
    construction_cost: float
    run: object                     # A.ALNSRun oder IR.Descent
    cost: float
    evaluations: int
    n_routes: int

    @property
    def improvement(self):
        """Prozentuale Verbesserung gegenüber der Savings-Konstruktion (positiv = besser)."""
        return 100.0 * (self.construction_cost - self.cost) / self.construction_cost


def analyse(settings):
    inst, D = instance(settings.n, settings.cluster_share, settings.seed, settings.capacity)
    routes = _routes_array(construction(settings.n, settings.cluster_share, settings.seed, settings.capacity))
    construction_cost = T.solution_cost(routes, D)
    if settings.method == "alns":
        run = A.alns(D, routes, inst.demands, settings.capacity, settings.budget, settings.chain_seed,
                      destroy_ops=settings.destroy_ops, repair_ops=settings.repair_ops, adaptive=settings.adaptive,
                      k_percent=settings.k_percent, keep_snapshots=False)
        cost, evaluations, n_routes = run.best_cost, run.evaluations, len(run.best_routes)
    else:
        run = IR.descend(D, routes, inst.demands, settings.capacity, active_moves=settings.active_moves,
                          keep_steps=False, max_evaluations=settings.budget)
        cost, evaluations, n_routes = run.cost, run.evaluations, len(run.routes)
    return Analysis(settings, inst, D, routes, construction_cost, run, cost, evaluations, n_routes)


# --- Sweeps und Tabellen -----------------------------------------------------------------------------------------------------------------------


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def run_config(base, seeds=C.SWEEP_SEEDS, chains=C.SWEEP_CHAINS, **changes):
    """Mittel über die festen Instanz-Seeds; bei `method='alns'` zusätzlich über `chains` Ketten-Seeds je Instanz
    (die kleine Nachbarschaft ist deterministisch, dort zählt jede Instanz nur einmal)."""
    s0 = replace(base, **changes)
    chain_range = range(chains) if s0.method == "alns" else (0,)
    rows = []
    for seed in seeds:
        for ci in chain_range:
            a = analyse(replace(s0, seed=seed, chain_seed=ci))
            rows.append({"improvement": a.improvement, "cost": a.cost, "construction_cost": a.construction_cost,
                         "n_routes": a.n_routes, "evaluations": a.evaluations})
    out = {k: _mean(rows, k) for k in rows[0]}
    out.update({"improvement_sd": float(np.std([r["improvement"] for r in rows])), "n_runs": len(rows)})
    return out


SWEEP_VALUES = {"budget": C.BUDGETS, "capacity": (15, 30, 60, 120, 250, 600), "n": (10, 20, 40, 60, 100, 150, 200),
                "k_percent": (5, 10, 20, 30, 50)}
SWEEP_LABELS = {"budget": "Budget (bewertete Kandidaten)", "capacity": "Kapazität", "n": "Stopps",
                "k_percent": "Zerstörungsgröße k (% der Kunden)"}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


def compare_methods(base=Settings()):
    """ALNS gegen die kleine Nachbarschaft der Schwester-Demo, gleiches (deklariertes) Budget - die zentrale
    Frage dieses Stücks."""
    return {"ALNS": run_config(base, method="alns"), "Kleine Nachbarschaft": run_config(base, method="small_neighborhood")}


def compare_adaptive(base=Settings()):
    return {"Adaptiv": run_config(base, method="alns", adaptive=True), "Nicht-adaptiv (gleichverteilt)": run_config(base, method="alns", adaptive=False)}


def compare_vs_fixed_sisr(base=Settings()):
    return {
        "ALNS (voller Pool, adaptiv)": run_config(base, method="alns"),
        "Festes SISR-Ruin (nur SISR+Greedy)": run_config(base, method="alns", destroy_ops=("sisr",), repair_ops=("greedy",), adaptive=False),
    }


def compare_destroy_ops(base=Settings()):
    out = {C.DESTROY_LABELS[op]: run_config(base, method="alns", destroy_ops=(op,)) for op in C.DESTROY_OPS}
    out["Alle zusammen (adaptiv)"] = run_config(base, method="alns", destroy_ops=C.DESTROY_OPS)
    return out


def compare_repair_ops(base=Settings()):
    return {C.REPAIR_LABELS[op]: run_config(base, method="alns", repair_ops=(op,)) for op in C.REPAIR_OPS}


SCALING_POLICIES = (("Budget 50 Tausend (fest)", lambda n: 50000), ("Budget 1 000 · Stopps", lambda n: 1000 * n))


def scaling_table(base=Settings()):
    return [{"label": label, "rows": [{"value": n, **run_config(base, n=n, budget=fn(n))} for n in C.SCALING_N]} for label, fn in SCALING_POLICIES]
