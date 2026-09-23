"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier über die fünf festen Sweep-
Instanzen belegt, mit denselben Auswertungsfunktionen wie die App selbst (`ev.run_config`/`ev.sweep`/
`ev.compare_*`) - NIE über ein Ad-hoc-Skript mit abweichender Zufalls-Bindung (die Lehre aus der
lin-kernighan-demo dieser Linie). Positive UND negative Aussagen: ALNS schlägt die kleine Nachbarschaft klar
(positiv) - aber die adaptive Gewichtsanpassung bringt praktisch nichts (negativ, ehrlicher Kernbefund).

Eine ALNS-Iteration kostet ungleich mehr Rechenzeit als ein einzelner Nachbarschafts-Kandidat (siehe
`alns_algorithm.py`) - dieses Modul pinnt deshalb bewusst NUR die Eckwerte jedes Sweeps (nicht jeden einzelnen
Punkt) mit `lru_cache`, damit dieselbe Konfiguration nie zweimal gerechnet wird; die Laufzeit dieser Datei liegt
trotzdem bei mehreren Minuten (mehr als bei jedem anderen Stück dieser Linie), das ist eine bewusste, dokumentierte
Grenze (siehe README)."""

from functools import lru_cache

import pytest

import alns_constants as C
import alns_evaluation as ev


@lru_cache(maxsize=None)
def _cfg(items):
    return ev.run_config(ev.Settings(), **dict(items))


def cfg(**kw):
    return _cfg(tuple(sorted(kw.items())))


def near(value, expected, tol):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


# --- Zentrale Frage: ALNS gegen die kleine Nachbarschaft der Schwester-Demo ----------------------------------------------------------------------


def test_alns_default_improvement():
    near(cfg()["improvement"], 8.886, 1.0)


def test_small_neighborhood_default_improvement():
    near(cfg(method="small_neighborhood")["improvement"], 2.278, 0.5)


def test_alns_clearly_beats_the_small_neighborhood_at_equal_declared_budget():
    alns = cfg()["improvement"]
    small = cfg(method="small_neighborhood")["improvement"]
    assert alns > small + 3.0


@pytest.mark.parametrize("capacity,expected,tol", [(15, 0.363, 0.3), (250, 2.535, 0.9), (600, 1.164, 0.5)])
def test_small_neighborhood_stays_far_behind_alns_across_capacities(capacity, expected, tol):
    """Die Schwester-Demo-Zahlen selbst (nicht nur beim Standardfall) - bestätigt, dass der ALNS-Vorteil nicht nur
    im Standardfall gilt (siehe README)."""
    small = cfg(method="small_neighborhood", capacity=capacity)["improvement"]
    near(small, expected, tol)


# --- Adaptivität: NICHT bestätigte Vorab-Hypothese (zentraler ehrlicher Befund) ------------------------------------------------------------------


def test_non_adaptive_improvement():
    near(cfg(adaptive=False)["improvement"], 8.869, 1.0)


def test_adaptive_weights_bring_practically_no_benefit_here():
    adaptive = cfg()["improvement"]
    non_adaptive = cfg(adaptive=False)["improvement"]
    assert abs(adaptive - non_adaptive) < 0.5                                # praktisch eine Krawatte


# --- ALNS gegen festes SISR-Ruin ------------------------------------------------------------------------------------------------------------------


def test_fixed_sisr_ruin_improvement():
    row = cfg(destroy_ops=("sisr",), repair_ops=("greedy",), adaptive=False)
    near(row["improvement"], 8.788, 1.0)


def test_full_adaptive_pool_edges_out_fixed_sisr_ruin_but_not_by_much():
    full = cfg()["improvement"]
    fixed = cfg(destroy_ops=("sisr",), repair_ops=("greedy",), adaptive=False)["improvement"]
    assert full >= fixed - 0.2                                               # kein klarer Verlust
    assert full < fixed + 2.0                                                # aber auch kein riesiger Vorteil


# --- Destroy-Operator-Ablation: Worst überraschend am schwächsten -----------------------------------------------------------------------------


@pytest.mark.parametrize("op,expected,tol", [("random", 8.602, 1.0), ("worst", 8.520, 1.0), ("shaw", 8.837, 1.0), ("sisr", 8.875, 1.0)])
def test_single_destroy_operator_numbers(op, expected, tol):
    row = cfg(destroy_ops=(op,))
    near(row["improvement"], expected, tol)


def test_worst_removal_is_the_weakest_single_destroy_operator():
    """Gegen die naive Erwartung ('teuerste Kunden zuerst entfernen sollte helfen') - hier gemessen, nicht
    angenommen: Worst bleibt sogar hinter Zufällig zurück."""
    vals = {op: cfg(destroy_ops=(op,))["improvement"] for op in C.DESTROY_OPS}
    assert min(vals, key=vals.get) == "worst"


def test_sisr_alone_captures_almost_all_of_the_full_pool_value():
    full = cfg()["improvement"]
    sisr_only = cfg(destroy_ops=("sisr",))["improvement"]
    assert sisr_only > full - 0.5


# --- Repair-Operator-Vergleich: Greedy schlägt Regret-2 --------------------------------------------------------------------------------------


def test_repair_operator_numbers():
    near(cfg(repair_ops=("greedy",))["improvement"], 8.952, 1.0)
    near(cfg(repair_ops=("regret2",))["improvement"], 8.509, 1.0)


def test_greedy_beats_regret2_against_the_common_literature_intuition():
    greedy = cfg(repair_ops=("greedy",))["improvement"]
    regret2 = cfg(repair_ops=("regret2",))["improvement"]
    assert greedy > regret2 + 0.2


# --- Zerstörungsgröße-Sweep: Plateau ab ~20 % ------------------------------------------------------------------------------------------------


def test_small_destruction_size_is_clearly_worse():
    near(cfg(k_percent=5)["improvement"], 7.831, 1.0)


def test_destruction_size_plateaus_from_twenty_percent_on():
    small_k = cfg(k_percent=5)["improvement"]
    default_k = cfg()["improvement"]                                        # k_percent=20 (Voreinstellung)
    large_k = cfg(k_percent=50)["improvement"]
    assert default_k > small_k + 0.5
    assert abs(default_k - large_k) < 0.5


# --- Budget-Sweep: konvergiert bei ~100 Tausend ---------------------------------------------------------------------------------------------


def test_budget_sweep_numbers():
    near(cfg(budget=5000)["improvement"], 7.741, 1.0)
    near(cfg(budget=200000)["improvement"], 8.934, 1.0)


def test_budget_plateaus_by_one_hundred_thousand():
    at_100k = cfg(budget=100000)["improvement"]
    at_200k = cfg(budget=200000)["improvement"]
    assert at_100k == pytest.approx(at_200k, abs=0.05)


def test_improvement_grows_with_budget_at_the_small_end():
    small = cfg(budget=5000)["improvement"]
    default = cfg()["improvement"]                                          # budget=50000 (Voreinstellung)
    assert default > small + 0.5


# --- Kapazitäts-Sweep: NICHT monoton, Optimum bei mittlerer Kapazität (zentraler ehrlicher Befund, wie die Schwester-Demo) -----------------------


def test_capacity_sweep_numbers():
    near(cfg(capacity=15)["improvement"], 1.041, 0.6)
    near(cfg(capacity=600)["improvement"], 6.567, 1.0)


def test_capacity_value_is_not_monotone_the_optimum_is_in_the_middle():
    small = cfg(capacity=15)["improvement"]
    medium = cfg()["improvement"]                                           # capacity=120 (Voreinstellung)
    large = cfg(capacity=600)["improvement"]
    assert medium > small + 3.0 and medium > large + 1.0


# --- Skalierung: NICHT monoton, Optimum bei n=100 (derselbe Routenzahl-Treiber wie die Kapazität) -------------------------------------------------


def test_scaling_numbers_at_fixed_fifty_thousand_budget():
    near(cfg(n=20, budget=50000)["improvement"], 2.045, 1.0)
    near(cfg(n=100, budget=50000)["improvement"], 9.525, 1.5)


def test_scaling_is_not_monotone_the_optimum_is_at_n_equals_one_hundred():
    small_n = cfg(n=20, budget=50000)["improvement"]
    peak_n = cfg(n=100, budget=50000)["improvement"]
    default_n = cfg()["improvement"]                                        # n=60, budget=50000 (Voreinstellung)
    assert peak_n > default_n > small_n


# --- Sonstiges --------------------------------------------------------------------------------------------------------------------------------


def test_preset_count_matches_the_readme():
    assert len(C.PRESETS) == 6


def test_savings_construction_gives_a_positive_finite_cost():
    inst, D = ev.instance(60, 0, C.DEFAULT_SEED, C.DEFAULT_CAPACITY)
    a = ev.analyse(ev.Settings())
    assert 0 < a.construction_cost < D.sum()
