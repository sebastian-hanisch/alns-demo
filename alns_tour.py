"""Routendarstellung, Kosten/Machbarkeit und Intra-Route-2-opt - wortgleiche Kopie von `vrpn_tour.py`
(VRP-Nachbarschaften-Demo, die CVRP-Infrastruktur dieser Linie). Dient hier als Baustein DES Vergleichsmaßstabs
(`alns_interroute.descend`, die "kleine Nachbarschaft"), gegen den ALNS (destroy/repair, `alns_algorithm.py`)
gemessen wird - siehe README.

Eine Route ist ein 1D-Array von Kundenindizes (OHNE Depot); eine Lösung ist eine Liste von Routen. Depot (Index 0)
ist an beiden Enden JEDER Route implizit (offener Pfad, Depot an beiden Enden fest verankert - siehe
`vrp-nachbarschaften-demo/vrpn_tour.py` für die vollständige Herleitung und den empirisch verifizierten Beweis der
Äquivalenz zur zyklischen 2-opt-Nachbarschaft der TSP-Wurzel im Sonderfall einer Route)."""

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-9


def dist_matrix(xy):
    xy = np.asarray(xy, dtype=float)
    d = xy[:, None, :] - xy[None, :, :]
    return np.sqrt((d * d).sum(axis=2))


def route_cost(route, D):
    if len(route) == 0:
        return 0.0
    t = np.concatenate([[0], route, [0]])
    return float(D[t[:-1], t[1:]].sum())


def route_demand(route, demands):
    return float(demands[route].sum()) if len(route) else 0.0


def solution_cost(routes, D):
    return sum(route_cost(r, D) for r in routes)


def solution_demand_ok(routes, demands, capacity):
    return all(route_demand(r, demands) <= capacity + EPS for r in routes)


def validate_partition(routes, n):
    """Jeder Kunde 1..n muss in GENAU einer Route vorkommen."""
    seen = np.concatenate(routes) if routes else np.array([], dtype=np.int64)
    return sorted(seen.tolist()) == list(range(1, n + 1))


# --- Intra-Route 2-opt (Pfad, Depot an beiden Enden fest) --------------------------------------------------------------------------------------


def _valid_pairs_path(n):
    i, j = np.indices((n, n))
    return (j >= i + 2) & (j <= n - 2)


def _delta_2opt_path(t, D):
    """`t` ist die Route MIT Depot an beiden Enden (`[0, c1, ..., ck, 0]`, Länge n)."""
    n = len(t)
    nxt = np.roll(t, -1)
    e = D[t, nxt]
    delta = D[np.ix_(t, t)] + D[np.ix_(nxt, nxt)] - e[:, None] - e[None, :]
    return delta, _valid_pairs_path(n)


def find_two_opt_move(route, D, rule="best"):
    """Bestes/erstes verbesserndes 2-opt innerhalb EINER Route (Depot an beiden Enden fest). Gibt ((i, j, delta)
    oder None, Zahl bewerteter Nachbarn) zurück - Positionen beziehen sich auf `t = [0, *route, 0]`."""
    if len(route) < 3:
        return None, 0
    t = np.concatenate([[0], route, [0]])
    delta, ok = _delta_2opt_path(t, D)
    d = np.where(ok, delta, np.inf)
    flat = d.ravel()
    n = len(t)
    if rule == "best":
        k = int(np.argmin(flat))
        if flat[k] >= -EPS:
            return None, int(ok.sum())
        i, j = divmod(k, n)
        return (i, j, float(flat[k])), int(ok.sum())
    improving = np.where(flat < -EPS)[0]
    if len(improving) == 0:
        return None, int(ok.sum())
    k = int(improving[0])
    i, j = divmod(k, n)
    return (i, j, float(flat[k])), int(ok.sum())


def apply_two_opt_move(route, move):
    i, j, _delta = move
    t = np.concatenate([[0], route, [0]])
    t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
    return t[1:-1]


@dataclass
class Step:
    move: object
    routes: list
    cost: float


@dataclass
class Descent:
    routes: list
    cost: float
    steps: list = field(default_factory=list)
    evaluations: int = 0
    n_moves: int = 0


def descend_intra_route(D, routes, rule="best", max_moves=100000, keep_steps=True, max_evaluations=None):
    """2-opt je Route für sich, bis keine Route mehr eine Verbesserung findet."""
    routes = [r.copy() for r in routes]
    cost = solution_cost(routes, D)
    steps = [Step(None, [r.copy() for r in routes], cost)] if keep_steps else []
    evaluations = n_moves = 0
    for _ in range(max_moves):
        if max_evaluations is not None and evaluations >= max_evaluations:
            break
        best = None
        for v, route in enumerate(routes):
            move, count = find_two_opt_move(route, D, rule)
            evaluations += count
            if move is not None and (best is None or move[2] < best[1][2] - EPS):
                best = (v, move)
        if best is None:
            break
        v, move = best
        routes[v] = apply_two_opt_move(routes[v], move)
        cost += move[2]
        n_moves += 1
        if keep_steps:
            steps.append(Step(("2opt", v, move), [r.copy() for r in routes], cost))
    cost = solution_cost(routes, D)
    return Descent(routes, cost, steps, evaluations, n_moves)
