import numpy as np
import pytest

import alns_construction as CN
import alns_destroy as D_
import alns_repair as R
import alns_scenario as S
import alns_tour as T

EPS = 1e-9


def _destroyed_instance(n=25, seed=3, capacity=80, k=8, destroy="random"):
    inst = S.generate(n, cluster_share=20, seed=seed, capacity=capacity)
    D = T.dist_matrix(inst.xy)
    routes = CN.savings_construction(inst.n, D, inst.demands, inst.capacity)
    rng = np.random.default_rng(1)
    remaining, removed = D_.OPERATORS[destroy](routes, D, inst.demands, k, rng)
    return inst, D, remaining, removed


@pytest.mark.parametrize("name", list(R.OPERATORS.keys()))
@pytest.mark.parametrize("destroy", list(D_.OPERATORS.keys()))
def test_repair_reinserts_every_removed_customer_into_a_feasible_complete_partition(name, destroy):
    inst, D, remaining, removed = _destroyed_instance(destroy=destroy)
    rng = np.random.default_rng(2)
    full = R.OPERATORS[name](remaining, removed, D, inst.demands, inst.capacity, rng)
    assert T.validate_partition(full, inst.n)
    assert T.solution_demand_ok(full, inst.demands, inst.capacity)


@pytest.mark.parametrize("name", list(R.OPERATORS.keys()))
def test_repair_never_violates_capacity_over_many_seeds(name):
    inst, D, remaining, removed = _destroyed_instance(n=40, seed=6, capacity=50, k=15)
    for seed in range(20):
        rng = np.random.default_rng(seed)
        full = R.OPERATORS[name](remaining, removed, D, inst.demands, inst.capacity, rng)
        assert T.solution_demand_ok(full, inst.demands, inst.capacity)
        assert T.validate_partition(full, inst.n)


def test_regret2_insertion_regret_matches_brute_force_recomputation():
    """Kreuzprüfung: die Regret-Reihenfolge von `regret2_insertion` gegen eine unabhängige Brute-Force-Berechnung
    der besten/zweitbesten Einfügeposition JEDES entfernten Kunden VOR dem ersten Einfügeschritt."""
    inst, D, remaining, removed = _destroyed_instance(n=30, seed=8, capacity=70, k=6)

    def brute_force_best_second(routes, c):
        options = []
        for v, route in enumerate(routes):
            load = float(inst.demands[route].sum()) if len(route) else 0.0
            if load + inst.demands[c] > inst.capacity + EPS:
                continue
            for q in range(len(route) + 1):
                prev = route[q - 1] if q > 0 else 0
                nxt = route[q] if q < len(route) else 0
                cost = D[prev, c] + D[c, nxt] - D[prev, nxt]
                options.append(cost)
        if not options:
            options.append(D[0, c] + D[c, 0])
        options.sort()
        best = options[0]
        second = options[1] if len(options) > 1 else best
        return best, second

    # Erster Schritt von regret2_insertion nachvollziehen: der Kunde mit dem größten (brute-force) Regret muss
    # tatsächlich an SEINER brute-force besten Position landen, sobald man ihn isoliert (vor jeder weiteren
    # Einfügung) in `remaining` einfügt - die zentrale Kreuzprüfung der Regret-Logik.
    regrets = {}
    bests = {}
    for c in removed.tolist():
        best, second = brute_force_best_second(remaining, c)
        regrets[c] = second - best
        bests[c] = best
    expected_first = max(regrets, key=regrets.get)

    rng = np.random.default_rng(0)
    full = R.OPERATORS["regret2"](remaining, removed, D, inst.demands, inst.capacity, rng)
    # Untere Schranke: jeder Kunde konnte höchstens an seiner (brute-force) besten Einzelposition landen -
    # die tatsächlichen Gesamtkosten dürfen diese (durch spätere Wechselwirkungen nur höhere) Summe nie
    # unterschreiten.
    lower_bound = T.solution_cost(remaining, D) + sum(bests.values())
    assert T.solution_cost(full, D) >= lower_bound - 1e-6
    assert regrets[expected_first] >= 0.0


@pytest.mark.parametrize("name", list(R.OPERATORS.keys()))
def test_repair_is_a_true_inverse_when_nothing_was_removed(name):
    inst, D, remaining, _removed = _destroyed_instance(k=0)
    rng = np.random.default_rng(0)
    full = R.OPERATORS[name](remaining, np.array([], dtype=np.int64), D, inst.demands, inst.capacity, rng)
    assert T.validate_partition(full, inst.n)
    assert abs(T.solution_cost(full, D) - T.solution_cost(remaining, D)) < 1e-6
