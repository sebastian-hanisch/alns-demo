import numpy as np
import pytest

import alns_algorithm as A
import alns_construction as CN
import alns_scenario as S
import alns_tour as T

EPS = 1e-9


# --- accept() (Metropolis) - Grenzfälle -----------------------------------------------------------------------------------------------------


def test_accept_always_true_for_non_worsening_delta():
    rng = np.random.default_rng(0)
    for T_val in (0.0, 1e-6, 1.0, 1000.0):
        assert A.accept(0.0, T_val, rng) is True
        assert A.accept(-5.0, T_val, rng) is True


def test_accept_never_true_for_worsening_delta_when_frozen():
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert A.accept(5.0, 0.0, rng) is False


def test_accept_almost_always_true_for_worsening_delta_when_very_hot():
    rng = np.random.default_rng(0)
    results = [A.accept(1.0, 1e6, rng) for _ in range(500)]
    assert np.mean(results) > 0.99


def test_accept_matches_the_metropolis_formula_directly():
    """Empirische Annahmequote von `accept()` gegen die geschlossene Formel exp(-delta/T), unabhängig gezogen mit
    einem eigenen RNG-Strom (kein gemeinsamer Seed-Trick)."""
    delta, temperature = 3.0, 2.0
    expected_prob = np.exp(-delta / temperature)
    accept_rng = np.random.default_rng(123)
    n = 20000
    hits = sum(1 for _ in range(n) if A.accept(delta, temperature, accept_rng))
    empirical = hits / n
    assert abs(empirical - expected_prob) < 0.01                         # ~140 Standardfehler Toleranz bei n=20000


# --- AdaptiveWeights -------------------------------------------------------------------------------------------------------------------------


def test_adaptive_weights_selection_probability_matches_weight_proportion():
    w = A.AdaptiveWeights(["a", "b", "c"])
    w.weights = {"a": 1.0, "b": 2.0, "c": 3.0}
    rng = np.random.default_rng(0)
    draws = [w.select(rng, adaptive=True) for _ in range(30000)]
    counts = {n: draws.count(n) for n in w.names}
    total = sum(counts.values())
    for n in w.names:
        expected = w.weights[n] / sum(w.weights.values())
        assert abs(counts[n] / total - expected) < 0.02


def test_adaptive_weights_disabled_selects_uniformly():
    w = A.AdaptiveWeights(["a", "b", "c"])
    w.weights = {"a": 1.0, "b": 100.0, "c": 0.01}                        # extrem schief - darf bei adaptive=False keine Rolle spielen
    rng = np.random.default_rng(0)
    draws = [w.select(rng, adaptive=False) for _ in range(30000)]
    counts = {n: draws.count(n) for n in w.names}
    for n in w.names:
        assert abs(counts[n] / len(draws) - 1 / 3) < 0.02


def test_adaptive_weights_stay_positive_after_many_segment_updates():
    w = A.AdaptiveWeights(["a", "b"], reaction_factor=0.3)
    rng = np.random.default_rng(0)
    for segment in range(50):
        for _ in range(20):
            name = w.select(rng, adaptive=True)
            w.reward(name, 0.0)                                          # nur Ablehnungen -> Gewichte duerfen NIE negativ werden
        w.update_segment()
        assert all(v > 0 for v in w.weights.values())


def test_adaptive_weights_rewards_the_operator_that_actually_finds_the_best_solutions():
    """Ein Operator, der IMMER die volle Belohnung bekommt, muss nach genug Segmenten ein klar höheres Gewicht
    haben als einer, der nie belohnt wird - sonst würde das Update-Gesetz nichts bewirken."""
    w = A.AdaptiveWeights(["good", "bad"], reaction_factor=0.3)
    rng = np.random.default_rng(0)
    for segment in range(30):
        for _ in range(10):
            w.reward("good", 33.0)
            w.reward("bad", 0.0)
        w.update_segment()
    assert w.weights["good"] > 5 * w.weights["bad"]


# --- Hauptschleife ---------------------------------------------------------------------------------------------------------------------------


def _run(n=25, seed=3, capacity=70, budget=15000, alns_seed=0, **kwargs):
    inst = S.generate(n, cluster_share=20, seed=seed, capacity=capacity)
    D = T.dist_matrix(inst.xy)
    routes = CN.savings_construction(inst.n, D, inst.demands, inst.capacity)
    run = A.alns(D, routes, inst.demands, inst.capacity, budget, alns_seed, **kwargs)
    return inst, D, run


def test_best_so_far_is_monotone_non_worsening():
    _inst, _D, run = _run()
    bh = run.best_history
    assert all(bh[i + 1] <= bh[i] + EPS for i in range(len(bh) - 1))


def test_final_and_best_solutions_are_valid_and_feasible():
    inst, D, run = _run()
    assert T.validate_partition(run.best_routes, inst.n)
    assert T.validate_partition(run.final_routes, inst.n)
    assert T.solution_demand_ok(run.best_routes, inst.demands, inst.capacity)
    assert T.solution_demand_ok(run.final_routes, inst.demands, inst.capacity)
    assert abs(T.solution_cost(run.best_routes, D) - run.best_cost) < 1e-6
    assert abs(T.solution_cost(run.final_routes, D) - run.final_cost) < 1e-6


def test_best_cost_never_exceeds_the_construction_cost():
    _inst, _D, run = _run()
    assert run.best_cost <= run.construction_cost + EPS


def test_budget_bookkeeping_reaches_but_does_not_wildly_overshoot_the_budget():
    _inst, _D, run = _run(budget=10000)
    assert run.evaluations >= 10000
    assert run.evaluations < 10000 + 200                                 # ein Durchlauf ueberschreitet nur um max. seine eigene Groesse


def test_alns_is_seed_deterministic():
    _inst1, _D1, run1 = _run(alns_seed=42)
    _inst2, _D2, run2 = _run(alns_seed=42)
    assert run1.best_cost == run2.best_cost
    assert run1.evaluations == run2.evaluations


def test_non_adaptive_run_still_produces_a_valid_and_improving_result():
    inst, D, run = _run(adaptive=False)
    assert T.validate_partition(run.best_routes, inst.n)
    assert run.best_cost <= run.construction_cost + EPS


def test_fixed_sisr_only_configuration_runs_without_needing_adaptive_weights():
    inst, D, run = _run(destroy_ops=("sisr",), repair_ops=("greedy",), adaptive=False)
    assert T.validate_partition(run.best_routes, inst.n)
    assert run.best_cost <= run.construction_cost + EPS
