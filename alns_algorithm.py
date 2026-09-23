"""ALNS-Hauptschleife: destroy (`alns_destroy.py`) + repair (`alns_repair.py`) + Metropolis-Akzeptanz (portiert aus
demselben Baustein wie `simulated-annealing-demo`/`parallel-tempering-demo` - kein eigenständig hergeleitetes
Risiko) + adaptive Operator-Gewichte (Ropke & Pisinger 2006, roulette-wheel-Auswahl, segmentweises Update).

**Budget-Konvention** (angepasst an die groß-schrittige Natur von ALNS, siehe README/Grenzen): EIN destroy+repair-
Durchlauf, der `m` Kunden neu einfügt, zählt als `m` bewertete Kandidaten - NICHT als die tatsächliche Zahl der
Einfüge-Kostenvergleiche (die bei Greedy/Regret-2 deutlich höher liegt). Bewusst grobkörniger als die
Bewertungs-Konvention von `alns_interroute.descend` (dort zählt JEDE geprüfte Position einzeln) - ein direkter
Vergleich "gleiches Budget" ist deshalb eine bewusste Vereinfachung, kein exakter Rechenzeit-Vergleich, siehe
README.

**Warum nur Metropolis** (wie bei `parallel-tempering-demo`): Ropke & Pisinger 2006 selbst nutzen die
Simulated-Annealing-Akzeptanzregel mit geometrischem Abkühlplan - hier wortgleich übernommen, nicht neu erfunden."""

from dataclasses import dataclass, field

import numpy as np

import alns_constants as C
import alns_construction as CN
import alns_destroy as DE
import alns_repair as RE
import alns_tour as T

EPS = 1e-9


def accept(delta, temperature, rng):
    """Metropolis: Verbesserung (delta<=0) immer an; sonst an mit Wahrscheinlichkeit exp(-delta/T). T<=0: nur
    verbessernd wird angenommen (Grenzfall "eingefroren")."""
    if delta <= EPS:
        return True
    if temperature <= 0:
        return False
    return bool(rng.random() < np.exp(-delta / temperature))


class AdaptiveWeights:
    """Ropke & Pisinger 2006: jeder Operator startet mit Gewicht 1, Segment-Belohnungen (REWARD_*) werden
    gesammelt und alle `segment_length` Iterationen per `w <- w*(1-r) + r*(Segment-Mittel)` eingearbeitet
    (`reaction_factor` r). `select` wählt roulette-wheel proportional zum aktuellen Gewicht - oder (adaptive=False)
    gleichverteilt, ohne die Gewichte je zu benutzen."""

    def __init__(self, names, reaction_factor=C.REACTION_FACTOR):
        self.names = list(names)
        self.reaction_factor = reaction_factor
        self.weights = {n: 1.0 for n in self.names}
        self._scores = {n: 0.0 for n in self.names}
        self._counts = {n: 0 for n in self.names}

    def select(self, rng, adaptive=True):
        if not adaptive:
            return self.names[int(rng.integers(0, len(self.names)))]
        w = np.array([self.weights[n] for n in self.names])
        p = w / w.sum()
        return self.names[int(rng.choice(len(self.names), p=p))]

    def reward(self, name, points):
        self._scores[name] += points
        self._counts[name] += 1

    def update_segment(self):
        for n in self.names:
            if self._counts[n] > 0:
                segment_mean = self._scores[n] / self._counts[n]
                self.weights[n] = self.weights[n] * (1 - self.reaction_factor) + self.reaction_factor * segment_mean
            self._scores[n] = 0.0
            self._counts[n] = 0


@dataclass
class Step:
    destroy_op: str
    repair_op: str
    accepted: bool
    cost: float
    best_cost: float


@dataclass
class ALNSRun:
    best_routes: list
    best_cost: float
    final_routes: list
    final_cost: float
    construction_cost: float
    evaluations: int
    iterations: int
    destroy_weights: dict
    repair_weights: dict
    cost_history: list = field(default_factory=list)
    best_history: list = field(default_factory=list)
    steps: list = field(default_factory=list)


def alns(D, start_routes, demands, capacity, budget, seed, destroy_ops=C.DESTROY_OPS, repair_ops=C.REPAIR_OPS,
         adaptive=True, k_percent=C.DEFAULT_K_PERCENT, t0_factor=C.DEFAULT_T0_FACTOR, cooling=C.DEFAULT_COOLING,
         segment_length=C.SEGMENT_LENGTH, keep_snapshots=True):
    rng = np.random.default_rng(seed)
    n = int(sum(len(r) for r in start_routes))
    k = max(1, round(k_percent / 100 * n))

    current = [r.copy() for r in start_routes]
    current_cost = T.solution_cost(current, D)
    construction_cost = current_cost
    best = [r.copy() for r in current]
    best_cost = current_cost
    temperature = t0_factor * current_cost

    dw = AdaptiveWeights(destroy_ops)
    rw = AdaptiveWeights(repair_ops)

    evaluations = 0
    iterations = 0
    cost_history = [current_cost]
    best_history = [best_cost]
    steps = [] if not keep_snapshots else [Step(None, None, True, current_cost, best_cost)]

    while evaluations < budget:
        d_name = dw.select(rng, adaptive)
        r_name = rw.select(rng, adaptive)
        remaining_routes, removed = DE.OPERATORS[d_name](current, D, demands, k, rng)
        new_routes = RE.OPERATORS[r_name](remaining_routes, removed, D, demands, capacity, rng)
        new_cost = T.solution_cost(new_routes, D)
        delta = new_cost - current_cost
        evaluations += max(1, len(removed))
        iterations += 1

        if new_cost < best_cost - EPS:
            reward = C.REWARD_BEST
            best = [r.copy() for r in new_routes]
            best_cost = new_cost
        elif new_cost < current_cost - EPS:
            reward = C.REWARD_BETTER
        else:
            reward = None

        accepted = accept(delta, temperature, rng)
        if accepted:
            if reward is None:
                reward = C.REWARD_ACCEPTED
            current = new_routes
            current_cost = new_cost
        else:
            reward = C.REWARD_REJECTED

        dw.reward(d_name, reward)
        rw.reward(r_name, reward)
        if iterations % segment_length == 0:
            dw.update_segment()
            rw.update_segment()
        temperature *= cooling

        cost_history.append(current_cost)
        best_history.append(best_cost)
        if keep_snapshots:
            steps.append(Step(d_name, r_name, accepted, current_cost, best_cost))

    return ALNSRun(best, best_cost, current, current_cost, construction_cost, evaluations, iterations,
                    dict(dw.weights), dict(rw.weights), cost_history, best_history, steps)
