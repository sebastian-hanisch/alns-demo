"""Smoke-Tests für die aus `vrp-nachbarschaften-demo` kopierten Bausteine (`alns_scenario.py`, `alns_tour.py`,
`alns_construction.py`, `alns_interroute.py`) - die vollständige Korrektheitskette (Brute-Force-Kreuzprüfung aller
vier Inter-Route-Züge, TSP-Sonderfall-Regressionstest) lebt bereits in `vrp-nachbarschaften-demo/tests/` und wird
hier NICHT dupliziert; nur wortgleiche Kopie-Treue wird stichprobenartig geprüft. Dazu `alns_evaluation.py`
(Settings/analyse/run_config für beide Verfahren)."""

from pathlib import Path

import numpy as np
import pytest

import alns_construction as CN
import alns_evaluation as EV
import alns_interroute as IR
import alns_scenario as S
import alns_tour as T

EPS = 1e-9


def test_capacity_ge_total_demand_reduces_to_a_single_route():
    inst = S.generate(30, cluster_share=0, seed=7, capacity=100000)
    D = T.dist_matrix(inst.xy)
    routes = CN.savings_construction(inst.n, D, inst.demands, inst.capacity)
    assert len(routes) == 1
    assert T.validate_partition(routes, inst.n)


@pytest.mark.parametrize("seed", range(10))
def test_construction_never_violates_capacity(seed):
    inst = S.generate(40, cluster_share=20, seed=seed, capacity=50)
    D = T.dist_matrix(inst.xy)
    routes = CN.savings_construction(inst.n, D, inst.demands, inst.capacity)
    assert T.solution_demand_ok(routes, inst.demands, inst.capacity)
    assert T.validate_partition(routes, inst.n)


def test_small_neighborhood_descend_never_worsens_the_construction():
    inst = S.generate(35, cluster_share=10, seed=3, capacity=70)
    D = T.dist_matrix(inst.xy)
    routes = CN.savings_construction(inst.n, D, inst.demands, inst.capacity)
    start_cost = T.solution_cost(routes, D)
    result = IR.descend(D, routes, inst.demands, inst.capacity, max_evaluations=50000)
    assert result.cost <= start_cost + EPS
    assert T.validate_partition(result.routes, inst.n)


_SIBLING_TOUR = Path(__file__).resolve().parents[2] / "vrp-nachbarschaften-demo" / "vrpn_tour.py"


@pytest.mark.skipif(not _SIBLING_TOUR.exists(), reason="vrp-nachbarschaften-demo nicht lokal vorhanden (z.B. CI-Checkout)")
def test_copied_modules_match_the_sibling_repo_byte_for_byte_on_a_fixed_instance():
    """Stellt sicher, dass die Kopie (alns_*) auf einer festen Instanz exakt dieselbe Konstruktion + Suche liefert
    wie das Original in `vrp-nachbarschaften-demo` - die eigentliche Korrektheitskette lebt dort."""
    import sys
    sib_dir = str(_SIBLING_TOUR.parent)
    sys.path.insert(0, sib_dir)
    try:
        import vrpn_construction as sib_CN
        import vrpn_interroute as sib_IR
        import vrpn_scenario as sib_S
        import vrpn_tour as sib_T

        inst_a = S.generate(25, cluster_share=15, seed=11, capacity=80)
        inst_b = sib_S.generate(25, cluster_share=15, seed=11, capacity=80)
        assert np.array_equal(inst_a.xy, inst_b.xy)
        assert np.array_equal(inst_a.demands, inst_b.demands)

        D = T.dist_matrix(inst_a.xy)
        routes_a = CN.savings_construction(inst_a.n, D, inst_a.demands, inst_a.capacity)
        routes_b = sib_CN.savings_construction(inst_b.n, D, inst_b.demands, inst_b.capacity)
        assert T.solution_cost(routes_a, D) == pytest.approx(sib_T.solution_cost(routes_b, D))

        result_a = IR.descend(D, routes_a, inst_a.demands, inst_a.capacity, max_evaluations=20000)
        result_b = sib_IR.descend(D, routes_b, inst_b.demands, inst_b.capacity, max_evaluations=20000)
        assert result_a.cost == pytest.approx(result_b.cost)
    finally:
        sys.path.remove(sib_dir)
        for mod in ("vrpn_construction", "vrpn_interroute", "vrpn_scenario", "vrpn_tour"):
            sys.modules.pop(mod, None)


# --- alns_evaluation.py -----------------------------------------------------------------------------------------------------------------------


def test_analyse_alns_and_small_neighborhood_both_return_valid_feasible_solutions():
    for method in ("alns", "small_neighborhood"):
        s = EV.Settings(n=20, capacity=60, budget=5000, method=method)
        a = EV.analyse(s)
        assert T.validate_partition(a.construction_routes, s.n)
        assert a.construction_cost > 0
        assert a.evaluations > 0


def test_analyse_improvement_is_between_0_and_100_percent_typically():
    a = EV.analyse(EV.Settings(n=25, capacity=70, budget=8000, method="alns"))
    assert -1.0 <= a.improvement <= 100.0                                 # kleine negative Werte sind in der Theorie moeglich (Rundung), sollten aber winzig sein


def test_run_config_averages_over_instance_seeds_and_alns_chains():
    out_alns = EV.run_config(EV.Settings(n=15, capacity=50, budget=3000, method="alns"), seeds=(1, 2), chains=2)
    assert out_alns["n_runs"] == 4                                        # 2 Instanzen x 2 Ketten
    out_small = EV.run_config(EV.Settings(n=15, capacity=50, budget=3000, method="small_neighborhood"), seeds=(1, 2), chains=2)
    assert out_small["n_runs"] == 2                                       # deterministisch: 1 Kette je Instanz


def test_analyse_is_seed_deterministic_for_alns():
    s = EV.Settings(n=20, capacity=60, budget=5000, method="alns", chain_seed=7)
    a1 = EV.analyse(s)
    a2 = EV.analyse(s)
    assert a1.cost == a2.cost
