"""Presets: Vollständigkeit, gültige Werte, Verbesserung bleibt in der gemessenen Spannweite über die 5 festen
Sweep-Instanzen (bei method='alns' zusätzlich über 2 Ketten-Seeds je Instanz), Permalink-Konstanten."""

import pytest

import alns_constants as C
import alns_evaluation as ev
import alns_presets as P


def _settings(p, seed=None):
    return ev.Settings(n=p["n"], cluster_share=p["ballung"], seed=p["seed"] if seed is None else seed,
                        capacity=p["capacity"], budget=p["budget"], method=p["method"], adaptive=p["adaptive"],
                        destroy_ops=tuple(p["destroy_ops"]), repair_ops=tuple(p["repair_ops"]), k_percent=p["k_percent"])


def test_every_preset_has_help_bands_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS)
    assert 5 <= len(C.PRESETS) <= 7
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and (p["n"] - C.N_MIN) % C.N_STEP == 0
        assert C.BALLUNG_MIN <= p["ballung"] <= C.BALLUNG_MAX and p["ballung"] % C.BALLUNG_STEP == 0
        assert p["budget"] in C.BUDGETS
        assert p["method"] in P.METHODS
        assert C.K_MIN <= p["k_percent"] <= C.K_MAX
        assert all(o in C.DESTROY_OPS for o in p["destroy_ops"]) and len(p["destroy_ops"]) > 0
        assert all(o in C.REPAIR_OPS for o in p["repair_ops"]) and len(p["repair_ops"]) > 0


def test_default_preset_equals_the_default_settings():
    assert _settings(C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_improvement_stays_in_its_measured_band_over_instances(name):
    p = C.PRESETS[name]
    lo, hi = C.PRESET_EXPECTED_BANDS[name]
    for seed in C.SWEEP_SEEDS:
        a = ev.analyse(_settings(p, seed=seed))
        assert lo <= a.improvement <= hi, (seed, a.improvement)
    a_default = ev.analyse(_settings(p))
    assert lo <= a_default.improvement <= hi


def test_small_neighborhood_preset_uses_that_method():
    matches = [p for p in C.PRESETS.values() if p["method"] == "small_neighborhood"]
    assert len(matches) >= 1


def test_non_adaptive_preset_exists():
    matches = [p for p in C.PRESETS.values() if p["method"] == "alns" and not p["adaptive"]]
    assert len(matches) >= 1


def test_bounds_and_snapping_constants():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)
    assert C.DEFAULT_BUDGET in C.BUDGETS


def test_ops_permalink_roundtrip():
    cast = P._ops_from_str(C.DESTROY_OPS)
    encoded = P._ops_to_str(("worst", "shaw"))
    decoded = cast(encoded)
    assert decoded == ("worst", "shaw")
    with pytest.raises(ValueError):
        cast("")


def test_bool_permalink_roundtrip():
    assert P._bool_from_str(P._bool_to_str(True)) is True
    assert P._bool_from_str(P._bool_to_str(False)) is False
