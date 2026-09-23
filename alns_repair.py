"""Repair-Operatoren: fügen ALLE entfernten Kunden wieder ein, Ergebnis ist immer eine vollständige, gültige,
kapazitäts-machbare Partition (leere Routen werden am Ende verworfen - eine Route ohne Kunden trägt nichts bei).
Signatur `(routes, removed, D, demands, capacity, rng) -> vollständige Routen` für beide Operatoren (der `rng`
wird nur von `greedy_insertion` für die Einfüge-REIHENFOLGE gebraucht, `regret2_insertion` ist deterministisch -
gemeinsame Signatur, damit `alns_algorithm.alns` sie austauschbar aufrufen kann). Kann kein Kunde in eine
bestehende Route eingefügt werden (Kapazität), wird eine neue Ein-Kunden-Route eröffnet - bei `DEMAND_MAX=9 <
CAPACITY_MIN=15` (siehe `alns_constants.py`) tritt das nur bei extrem kleiner, vom Nutzer selbst gewählter
Kapazität auf.

- **`greedy_insertion`** (Ropke & Pisinger 2006, reine Greedy-Variante): die entfernten Kunden werden in
  ZUFÄLLIGER Reihenfolge nacheinander an ihrer jeweils günstigsten machbaren Position eingefügt - keine
  Neusortierung nach jedem Einfügen (das unterscheidet sie von Regret-2 unten).
- **`regret2_insertion`**: in jeder Runde wird über ALLE noch nicht eingefügten Kunden neu berechnet, wer den
  größten Regret hat (Differenz zwischen bester und zweitbester Einfügeposition über ALLE Routen) - dieser Kunde
  wird zuerst eingefügt. Ein Kunde, dessen zweitbeste Position deutlich teurer ist als seine beste, wird so nicht
  bis zuletzt aufgeschoben (wo er dann nur noch schlechte Optionen hätte)."""

import numpy as np

EPS = 1e-9


def _load(route, demands):
    return float(demands[route].sum()) if len(route) else 0.0


def _best_insertion(route, c, D):
    """(cost, position) der günstigsten Einfügeposition von Kunde `c` in `route` (Depot an beiden Enden)."""
    best_cost, best_q = None, None
    for q in range(len(route) + 1):
        prev = route[q - 1] if q > 0 else 0
        nxt = route[q] if q < len(route) else 0
        cost = D[prev, c] + D[c, nxt] - D[prev, nxt]
        if best_cost is None or cost < best_cost - EPS:
            best_cost, best_q = cost, q
    return best_cost, best_q


def _drop_empty(routes):
    return [r for r in routes if len(r) > 0]


def greedy_insertion(routes, removed, D, demands, capacity, rng):
    routes = [r.copy() for r in routes]
    order = rng.permutation(np.asarray(removed))
    for c in order:
        c = int(c)
        best = None                                                   # (cost, v, q)
        for v, route in enumerate(routes):
            if _load(route, demands) + demands[c] > capacity + EPS:
                continue
            cost, q = _best_insertion(route, c, D)
            if best is None or cost < best[0] - EPS:
                best = (cost, v, q)
        if best is None:
            routes.append(np.array([c], dtype=np.int64))
        else:
            _cost, v, q = best
            routes[v] = np.insert(routes[v], q, c)
    return _drop_empty(routes)


def regret2_insertion(routes, removed, D, demands, capacity, rng):
    routes = [r.copy() for r in routes]
    remaining = [int(c) for c in removed]
    while remaining:
        best_choice = None                                            # (regret, c, v, q, cost)
        for c in remaining:
            options = []                                               # (cost, v, q)
            for v, route in enumerate(routes):
                if _load(route, demands) + demands[c] > capacity + EPS:
                    continue
                cost, q = _best_insertion(route, c, D)
                options.append((cost, v, q))
            if not options:
                options.append((D[0, c] + D[c, 0], None, None))        # neue Route als einzige Option
            options.sort(key=lambda t: t[0])
            best_cost, best_v, best_q = options[0]
            second_cost = options[1][0] if len(options) > 1 else best_cost
            regret = second_cost - best_cost
            if best_choice is None or regret > best_choice[0] + EPS:
                best_choice = (regret, c, best_v, best_q, best_cost)
        _regret, c, v, q, _cost = best_choice
        if v is None:
            routes.append(np.array([c], dtype=np.int64))
        else:
            routes[v] = np.insert(routes[v], q, c)
        remaining.remove(c)
    return _drop_empty(routes)


OPERATORS = {"greedy": greedy_insertion, "regret2": regret2_insertion}
