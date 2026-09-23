import numpy as np
import pytest

import alns_constants as C
import alns_construction as CN
import alns_destroy as D_
import alns_scenario as S
import alns_tour as T


def _instance(n=25, seed=3, capacity=80):
    inst = S.generate(n, cluster_share=20, seed=seed, capacity=capacity)
    D = T.dist_matrix(inst.xy)
    routes = CN.savings_construction(inst.n, D, inst.demands, inst.capacity)
    return inst, D, routes


@pytest.mark.parametrize("name", list(D_.OPERATORS.keys()))
@pytest.mark.parametrize("k", [1, 5, 12, 999])
def test_destroy_removes_exactly_k_customers_or_all(name, k):
    inst, D, routes = _instance()
    op = D_.OPERATORS[name]
    rng = np.random.default_rng(7)
    remaining, removed = op(routes, D, inst.demands, k, rng)
    expected = min(k, inst.n)
    assert len(removed) == expected
    assert len(set(removed.tolist())) == expected                       # keine Duplikate


@pytest.mark.parametrize("name", list(D_.OPERATORS.keys()))
def test_destroy_leaves_a_valid_partial_partition(name):
    inst, D, routes = _instance()
    op = D_.OPERATORS[name]
    rng = np.random.default_rng(11)
    remaining, removed = op(routes, D, inst.demands, 8, rng)
    remaining_customers = np.concatenate(remaining) if remaining else np.array([], dtype=np.int64)
    all_customers = sorted(remaining_customers.tolist() + removed.tolist())
    assert all_customers == list(range(1, inst.n + 1))                  # jeder Kunde genau einmal


@pytest.mark.parametrize("name", list(D_.OPERATORS.keys()))
def test_destroy_over_many_seeds_never_duplicates_or_loses_customers(name):
    inst, D, routes = _instance(n=40, seed=9, capacity=60)
    op = D_.OPERATORS[name]
    for seed in range(20):
        rng = np.random.default_rng(seed)
        remaining, removed = op(routes, D, inst.demands, 10, rng)
        remaining_customers = np.concatenate(remaining) if remaining else np.array([], dtype=np.int64)
        all_customers = sorted(remaining_customers.tolist() + removed.tolist())
        assert all_customers == list(range(1, inst.n + 1))


def test_worst_removal_prefers_high_gain_customers_on_average():
    """Nicht deterministisch (Zufallsrauschen), aber der Mittelwert des entfernten Gewinns muss über viele Seeds
    klar über dem einer zufälligen Auswahl liegen - sonst würde 'worst' nichts anderes tun als 'random'."""
    inst, D, routes = _instance(n=30, seed=5, capacity=70)

    def removal_gains(routes):
        gains = {}
        for route in routes:
            for p, c in enumerate(route):
                prev = route[p - 1] if p > 0 else 0
                nxt = route[p + 1] if p + 1 < len(route) else 0
                gains[int(c)] = D[prev, c] + D[c, nxt] - D[prev, nxt]
        return gains

    gains = removal_gains(routes)
    worst_means, random_means = [], []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        _remaining, removed_w = D_.OPERATORS["worst"](routes, D, inst.demands, 8, rng)
        _remaining, removed_r = D_.OPERATORS["random"](routes, D, inst.demands, 8, rng)
        worst_means.append(np.mean([gains[c] for c in removed_w.tolist()]))
        random_means.append(np.mean([gains[c] for c in removed_r.tolist()]))
    assert np.mean(worst_means) > np.mean(random_means)


def test_shaw_removal_group_is_more_compact_than_random_on_average():
    """Die von 'shaw' entfernte Gruppe muss im Mittel eine kleinere Streuung (Summe paarweiser Distanzen) haben
    als eine zufällige Gruppe gleicher Größe - sonst wäre 'shaw' nur eine umständliche Variante von 'random'."""
    inst, D, routes = _instance(n=40, seed=13, capacity=90)

    def spread(customers):
        customers = list(customers)
        if len(customers) < 2:
            return 0.0
        return float(np.mean([D[a, b] for i, a in enumerate(customers) for b in customers[i + 1:]]))

    shaw_spreads, random_spreads = [], []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        _remaining, removed_s = D_.OPERATORS["shaw"](routes, D, inst.demands, 6, rng)
        _remaining, removed_r = D_.OPERATORS["random"](routes, D, inst.demands, 6, rng)
        shaw_spreads.append(spread(removed_s.tolist()))
        random_spreads.append(spread(removed_r.tolist()))
    assert np.mean(shaw_spreads) < np.mean(random_spreads)


def test_sisr_removal_takes_contiguous_strings_not_scattered_customers():
    """Jede von 'sisr' entfernte zusammenhängende Gruppe muss (bis auf die Fugen zwischen mehreren entfernten
    Strings) aus tatsächlich benachbarten Positionen der Ausgangsroute stammen - der strukturelle Unterschied zu
    'random'."""
    inst, D, routes = _instance(n=35, seed=17, capacity=70)
    route_of = {int(c): v for v, r in enumerate(routes) for c in r}
    position_of = {int(c): p for r in routes for p, c in enumerate(r)}
    rng = np.random.default_rng(4)
    _remaining, removed = D_.OPERATORS["sisr"](routes, D, inst.demands, 10, rng)
    by_route = {}
    for c in removed.tolist():
        by_route.setdefault(route_of[c], []).append(position_of[c])
    for positions in by_route.values():
        positions.sort()
        gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
        assert all(g <= C.SISR_MAX_STRING_LENGTH for g in gaps)


def test_destroy_operators_are_seed_deterministic():
    inst, D, routes = _instance()
    for name, op in D_.OPERATORS.items():
        r1 = op(routes, D, inst.demands, 6, np.random.default_rng(42))
        r2 = op(routes, D, inst.demands, 6, np.random.default_rng(42))
        assert np.array_equal(r1[1], r2[1]), name
