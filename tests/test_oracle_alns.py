"""Orakel-Tests mit anderem Rechenweg: jeder Zug/Einfügeschritt per Anwenden + voller Neubewertung (statt Delta-Formel), Greedy/Regret-2 und
Worst-Removal als Brute-Force-Nachbau, ALNS-Kosten und Zulässigkeit per Neuberechnung aus den Routen, und die Routenzahl der kleinen Nachbarschaft
(Züge können eine Route leeren - leere Routen dürfen weder gezählt noch angezeigt werden)."""
import numpy as np
import pytest

import alns_algorithm as A
import alns_construction as CO
import alns_destroy as DE
import alns_evaluation as EV
import alns_interroute as IR
import alns_repair as RE
import alns_scenario as S
import alns_tour as T


def _inst(n, share, seed, cap):
    i = S.generate(n, share, seed, capacity=cap)
    return i, T.dist_matrix(i.xy)


def _cost(routes, D):
    tot = 0.0
    for r in routes:
        if len(r):
            p = [0] + [int(c) for c in r] + [0]
            tot += sum(D[p[k], p[k + 1]] for k in range(len(p) - 1))
    return tot


def _partition_ok(routes, n):
    return sorted(int(c) for r in routes for c in r) == list(range(1, n + 1))


def _cap_ok(routes, demands, cap):
    return all(sum(demands[int(c)] for c in r) <= cap + 1e-9 for r in routes)


def _scrambled(n, share, seed, cap):
    i, D = _inst(n, share, seed, cap)
    routes = CO.savings_construction(n, D, i.demands, cap)
    g = np.random.default_rng(seed)
    rem, removed = DE.random_removal(routes, D, i.demands, n, g)
    return i, D, RE.greedy_insertion(rem, removed, D, i.demands, cap, g)


def _best_deltas(routes, D, demands, cap, max_seg=3):
    base = _cost(routes, D)
    R = [list(map(int, r)) for r in routes]
    best = {k: 0.0 for k in ("relocate", "swap", "2opt_star", "cross")}

    def consider(kind, new):
        if _cap_ok(new, demands, cap):
            best[kind] = min(best[kind], _cost(new, D) - base)

    m = len(R)
    for a in range(m):
        for p, c in enumerate(R[a]):
            for b in range(m):
                for q in range(len(R[b]) + 1 if b != a else 0):
                    new = [r[:] for r in R]
                    new[a].pop(p)
                    new[b].insert(q, c)
                    consider("relocate", new)
        for b in range(a + 1, m):
            for p in range(len(R[a])):
                for q in range(len(R[b])):
                    new = [r[:] for r in R]
                    new[a][p], new[b][q] = new[b][q], new[a][p]
                    consider("swap", new)
            for i in range(len(R[a]) + 1):
                for j in range(len(R[b]) + 1):
                    if (i == 0 and j == 0) or (i == len(R[a]) and j == len(R[b])):
                        continue
                    new = [r[:] for r in R]
                    new[a], new[b] = R[a][:i] + R[b][j:], R[b][:j] + R[a][i:]
                    consider("2opt_star", new)
            for l1 in range(1, max_seg + 1):
                for l2 in range(1, max_seg + 1):
                    for p1 in range(len(R[a]) - l1 + 1):
                        for p2 in range(len(R[b]) - l2 + 1):
                            new = [r[:] for r in R]
                            new[a] = R[a][:p1] + R[b][p2:p2 + l2] + R[a][p1 + l1:]
                            new[b] = R[b][:p2] + R[a][p1:p1 + l1] + R[b][p2 + l2:]
                            consider("cross", new)
    return best


@pytest.mark.parametrize("n,share,seed,cap", [(8, 0, 1, 15), (10, 100, 2, 20), (12, 0, 3, 15), (9, 50, 4, 12), (11, 0, 5, 30), (10, 0, 6, 15)])
def test_best_inter_route_move_and_its_delta_match_full_re_evaluation(n, share, seed, cap):
    i, D, routes = _scrambled(n, share, seed, cap)
    base = _cost(routes, D)
    bf = _best_deltas(routes, D, i.demands, cap)
    for kind, find, apply_, idx in (("relocate", lambda: IR.find_relocate_move(routes, D, i.demands, cap), IR.apply_relocate_move, 4),
                                    ("swap", lambda: IR.find_swap_move(routes, D, i.demands, cap), IR.apply_swap_move, 4),
                                    ("2opt_star", lambda: IR.find_two_opt_star_move(routes, D, i.demands, cap), IR.apply_two_opt_star_move, 4),
                                    ("cross", lambda: IR.find_cross_exchange_move(routes, D, i.demands, cap, 3), IR.apply_cross_exchange_move, 6)):
        move, _ = find()
        assert (move[idx] if move else 0.0) == pytest.approx(bf[kind], abs=1e-9), kind     # bester Zug = beste volle Neubewertung
        if move:
            new = apply_(routes, move)
            assert _partition_ok(new, n) and _cap_ok(new, i.demands, cap)
            assert _cost(new, D) - base == pytest.approx(move[idx], abs=1e-9)


def _best_insert(route, c, D):
    base = _cost([route], D)
    best = None
    for q in range(len(route) + 1):
        d = _cost([route[:q] + [c] + route[q:]], D) - base
        if best is None or d < best[0] - 1e-9:
            best = (d, q)
    return best


@pytest.mark.parametrize("seed", range(8))
def test_repair_operators_match_a_brute_force_reimplementation(seed):
    n, cap = 14, 20
    i, D = _inst(n, 0, seed, cap)
    routes = CO.savings_construction(n, D, i.demands, cap)
    rem, removed = DE.random_removal(routes, D, i.demands, 6, np.random.default_rng(seed))
    out_g = RE.greedy_insertion([r.copy() for r in rem], removed, D, i.demands, cap, np.random.default_rng(seed))
    out_r = RE.regret2_insertion([r.copy() for r in rem], removed, D, i.demands, cap, np.random.default_rng(seed))
    for out in (out_g, out_r):
        assert _partition_ok(out, n) and _cap_ok(out, i.demands, cap) and all(len(r) for r in out)
    cur = [list(map(int, r)) for r in rem]
    for c in np.random.default_rng(seed).permutation(np.asarray(removed)):
        c, best = int(c), None
        for v, r in enumerate(cur):
            if sum(i.demands[x] for x in r) + i.demands[c] <= cap + 1e-9:
                d, q = _best_insert(r, c, D)
                if best is None or d < best[0] - 1e-9:
                    best = (d, v, q)
        cur.append([c]) if best is None else cur[best[1]].insert(best[2], c)
    assert _cost(out_g, D) == pytest.approx(_cost([r for r in cur if r], D), abs=1e-9)
    cur, todo = [list(map(int, r)) for r in rem], [int(c) for c in removed]
    while todo:
        choice = None
        for c in todo:
            opts = sorted((_best_insert(r, c, D) + (v,) for v, r in enumerate(cur) if sum(i.demands[x] for x in r) + i.demands[c] <= cap + 1e-9), key=lambda t: t[0])
            if not opts:
                opts = [(D[0, c] + D[c, 0], None, None)]
            regret = opts[1][0] - opts[0][0] if len(opts) > 1 else 0.0
            if choice is None or regret > choice[0] + 1e-9:
                choice = (regret, c, opts[0])
        _, c, (d, q, v) = choice
        cur.append([c]) if v is None else cur[v].insert(q, c)
        todo.remove(c)
    assert _cost(out_r, D) == pytest.approx(_cost([r for r in cur if r], D), abs=1e-9)


class _Rank0Rng:
    def random(self):
        return 0.0


def test_worst_removal_removes_the_customers_with_the_largest_recomputed_saving():
    i, D = _inst(15, 0, 3, 30)
    routes = [list(map(int, r)) for r in CO.savings_construction(15, D, i.demands, 30)]
    _, removed = DE.worst_removal([np.array(r) for r in routes], D, i.demands, 4, _Rank0Rng())
    cur, seq = [r[:] for r in routes], []
    for _ in range(4):
        base = _cost(cur, D)
        gains = {}
        for r in cur:
            for c in r:
                gains[c] = base - _cost([[x for x in q if x != c] for q in cur], D)       # volle Neubewertung statt Delta-Formel
        pick = max(gains, key=lambda c: (gains[c], -c))
        seq.append(pick)
        cur = [[x for x in q if x != pick] for q in cur]
    assert sorted(seq) == removed.tolist()


@pytest.mark.parametrize("seed,dops,rops,adaptive", [(0, ("random", "worst", "shaw", "sisr"), ("greedy", "regret2"), True), (1, ("sisr",), ("greedy",), False),
                                                      (2, ("shaw", "worst"), ("regret2",), True)])
def test_alns_costs_feasibility_and_budget_hold_up_under_recomputation(seed, dops, rops, adaptive):
    n, cap = 30, 40
    i, D = _inst(n, 100, seed, cap)
    routes = CO.savings_construction(n, D, i.demands, cap)
    run = A.alns(D, routes, i.demands, cap, 1500, seed, destroy_ops=dops, repair_ops=rops, adaptive=adaptive, keep_snapshots=True)
    for r_, c_ in ((run.best_routes, run.best_cost), (run.final_routes, run.final_cost)):
        assert _partition_ok(r_, n) and _cap_ok(r_, i.demands, cap)
        assert c_ == pytest.approx(_cost(r_, D), abs=1e-9)
    assert run.best_cost <= run.construction_cost + 1e-9 and run.best_cost <= min(run.cost_history) + 1e-9
    assert len(run.best_history) == run.iterations + 1 and 1500 <= run.evaluations < 1500 + n + 1


def test_route_count_of_the_small_neighborhood_ignores_emptied_routes():
    a = EV.analyse(EV.Settings(n=10, cluster_share=0, seed=53, capacity=20, budget=20000, method="small_neighborhood"))
    assert all(len(r) > 0 for r in a.run.routes)                                            # früher blieb eine geleerte Route als "Route mit 0 Stopps" stehen
    assert a.n_routes == len(a.run.routes)
    assert _partition_ok(a.run.routes, 10) and _cost(a.run.routes, a.D) == pytest.approx(a.cost, abs=1e-9)
