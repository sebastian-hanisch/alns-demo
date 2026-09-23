"""Destroy-Operatoren: jeder entfernt GENAU `k` Kunden (oder alle, falls `k` > Kundenzahl) aus den Routen und
gibt (verbleibende Routen, entfernte Kundenliste) zurück - Signatur `(routes, D, demands, k, rng) ->
(routes, removed)` für alle vier Operatoren, damit `alns_algorithm.alns` sie austauschbar aufrufen kann.

- **`random_removal`**: k gleichverteilt zufällige Kunden (die Kontrollgröße).
- **`worst_removal`** (Ropke & Pisinger 2006): iterativ der Kunde mit dem größten "Entfernungsgewinn" (Strecke,
  die seine Entfernung spart) - mit `p`-ter-Potenz-Zufallsrauschen in der Rangauswahl (`WORST_REMOVAL_P`), damit
  nicht immer exakt dieselbe Reihenfolge gewählt wird. Gewinne werden nach JEDER Entfernung neu berechnet (die
  Nachbarn eines Kunden ändern sich, sobald ein anderer Kunde aus derselben Route verschwindet).
- **`shaw_removal`** (Shaw 1998, relatedness): ein zufälliger Startkunde, dann iterativ - gewichtet über dieselbe
  `p`-te-Potenz-Zufallsauswahl - der zu der bereits entfernten Menge "ähnlichste" Kunde (gewichtete Summe aus
  normierter Distanz UND normierter Bedarfsdifferenz zum zuletzt entfernten Kunden, `SHAW_DISTANCE_WEIGHT`/
  `SHAW_DEMAND_WEIGHT`) - entfernt eine räumlich/nach Bedarf zusammenhängende Gruppe.
- **`sisr_string_removal`** (vereinfacht nach Christiaens & Vanden Berghe 2020): entfernt zusammenhängende
  STRINGS (Teilsequenzen fester Reihenfolge) aus zufälligen Routen statt einzelner Kunden, bis `k` Kunden entfernt
  sind (letzter String wird bei Bedarf gekürzt). Vereinfachung gegenüber dem Papier: keine Bevorzugung räumlich
  benachbarter Routen bei der String-Auswahl, keine adaptive Stringlängen-Verteilung - dokumentiert in den
  Grenzen des README, der für die Operator-Ablation entscheidende Unterschied (zusammenhängende Strings statt
  Einzelkunden) bleibt erhalten."""

import numpy as np

import alns_constants as C

EPS = 1e-9


def _all_customers(routes):
    return np.concatenate(routes) if routes else np.array([], dtype=np.int64)


def _remove_from_routes(routes, to_remove_set):
    return [np.array([c for c in r if c not in to_remove_set], dtype=np.int64) for r in routes]


def _power_pick(rng, n, p):
    """Index 0..n-1 über `floor(rand()^p * n)` (Ropke & Pisinger 2006) - p=1 ist gleichverteilt, wachsendes p
    bevorzugt zunehmend den ersten (besten) Rang."""
    idx = int(rng.random() ** p * n)
    return min(idx, n - 1)


def random_removal(routes, D, demands, k, rng):
    customers = _all_customers(routes)
    k = min(k, len(customers))
    removed = rng.choice(customers, size=k, replace=False)
    removed_set = set(removed.tolist())
    return _remove_from_routes(routes, removed_set), np.array(sorted(removed.tolist()), dtype=np.int64)


def worst_removal(routes, D, demands, k, rng):
    routes = [r.copy() for r in routes]
    n_total = len(_all_customers(routes))
    k = min(k, n_total)
    removed = []
    for _ in range(k):
        gains = []
        for v, route in enumerate(routes):
            for p, c in enumerate(route):
                prev = route[p - 1] if p > 0 else 0
                nxt = route[p + 1] if p + 1 < len(route) else 0
                gain = D[prev, c] + D[c, nxt] - D[prev, nxt]
                gains.append((gain, v, p, c))
        if not gains:
            break
        gains.sort(key=lambda t: -t[0])
        idx = _power_pick(rng, len(gains), C.WORST_REMOVAL_P)
        _gain, v, p, c = gains[idx]
        routes[v] = np.delete(routes[v], p)
        removed.append(int(c))
    return routes, np.array(sorted(removed), dtype=np.int64)


def shaw_removal(routes, D, demands, k, rng):
    customers = _all_customers(routes)
    n_total = len(customers)
    k = min(k, n_total)
    if k == 0:
        return [r.copy() for r in routes], np.array([], dtype=np.int64)
    max_dist = float(D.max()) if D.max() > 0 else 1.0
    max_demand = float(demands.max()) if demands.max() > 0 else 1.0
    start = int(rng.choice(customers))
    removed = [start]
    remaining = set(customers.tolist()) - {start}
    while len(removed) < k and remaining:
        last = removed[-1]
        rem_list = sorted(remaining)
        rel = [
            C.SHAW_DISTANCE_WEIGHT * (D[last, j] / max_dist) + C.SHAW_DEMAND_WEIGHT * (abs(demands[last] - demands[j]) / max_demand)
            for j in rem_list
        ]
        order = np.argsort(rel)
        idx = _power_pick(rng, len(order), C.WORST_REMOVAL_P)
        pick = rem_list[order[idx]]
        removed.append(pick)
        remaining.discard(pick)
    removed_set = set(removed)
    return _remove_from_routes(routes, removed_set), np.array(sorted(removed), dtype=np.int64)


def sisr_string_removal(routes, D, demands, k, rng):
    routes = [r.copy() for r in routes]
    k = min(k, len(_all_customers(routes)))
    removed = []
    guard = 0
    while len(removed) < k and guard < 10 * (k + 1):
        guard += 1
        non_empty = [v for v, r in enumerate(routes) if len(r) > 0]
        if not non_empty:
            break
        v = int(rng.choice(non_empty))
        route = routes[v]
        max_len = min(C.SISR_MAX_STRING_LENGTH, len(route), k - len(removed))
        if max_len < 1:
            continue
        length = int(rng.integers(1, max_len + 1))
        start = int(rng.integers(0, len(route) - length + 1))
        string = route[start:start + length]
        removed.extend(int(c) for c in string)
        routes[v] = np.delete(route, np.arange(start, start + length))
    return routes, np.array(sorted(removed), dtype=np.int64)


OPERATORS = {"random": random_removal, "worst": worst_removal, "shaw": shaw_removal, "sisr": sisr_string_removal}
