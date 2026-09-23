"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt, beide Verfahren, Randwerte, Würfel-Knopf,
Permalink-Grenzen, Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import alns_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(alns_step=1, **state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if alns_step != 1:
        at.select_slider(key="alns_step").set_value(alns_step).run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_default_run_has_no_exception():
    at = _run()
    _ok(at)
    assert at.metric


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert at.session_state["n_slider"] == p["n"] and at.session_state["capacity_slider"] == p["capacity"]
    assert at.session_state["budget_select"] == p["budget"]
    assert at.metric


@pytest.mark.parametrize("step", [1, 2, 3])
def test_every_step_runs_for_both_methods(step):
    for method in ("alns", "small_neighborhood"):
        at = _run(n_slider=20, budget_select=5000, method_select=method, alns_step=step)
        _ok(at)
        assert at.get("plotly_chart") and at.session_state["alns_step"] == step


def test_deselecting_all_destroy_ops_falls_back_to_random_with_a_warning():
    at = _run(destroy_select=[])
    _ok(at)
    assert any("Zufällig wird automatisch" in w.value for w in at.warning)


def test_deselecting_all_repair_ops_falls_back_to_greedy_with_a_warning():
    at = _run(repair_select=[])
    _ok(at)
    assert any("Greedy wird automatisch" in w.value for w in at.warning)


def test_dice_button_changes_the_seed():
    at = _run(budget_select=5000)
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_toggling_adaptive_off_runs_without_exception():
    _ok(_run(budget_select=5000, adaptive_toggle=False))


@pytest.mark.parametrize("kw", [
    dict(n_slider=C.N_MAX, budget_select=10000), dict(n_slider=C.N_MIN, budget_select=5000),
    dict(capacity_slider=C.CAPACITY_MIN, budget_select=5000), dict(capacity_slider=C.CAPACITY_MAX, budget_select=5000),
    dict(k_percent_slider=C.K_MIN, budget_select=5000), dict(k_percent_slider=C.K_MAX, budget_select=5000),
    dict(destroy_select=["sisr"], budget_select=5000), dict(repair_select=["regret2"], budget_select=5000),
])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["n"] = "9999"
    at.query_params["capacity"] = "9999999"
    at.query_params["budget"] = "12345"
    at.query_params["k"] = "9999"
    at.query_params["destroy"] = "sisr,worst"
    at.query_params["adaptive"] = "0"
    at.query_params["method"] = "small_neighborhood"
    at.run()
    _ok(at)
    assert at.session_state["n_slider"] == C.N_MAX
    assert at.session_state["capacity_slider"] == C.CAPACITY_MAX and at.session_state["budget_select"] == C.DEFAULT_BUDGET
    assert at.session_state["k_percent_slider"] == C.K_MAX
    assert at.session_state["destroy_select"] == ("worst", "sisr") or at.session_state["destroy_select"] == ("sisr", "worst")
    assert at.session_state["adaptive_toggle"] is False
    assert at.session_state["method_select"] == "small_neighborhood"


def test_sweeps_run_on_demand():
    at = _run(n_slider=15, budget_select=5000)
    at.selectbox(key="sweep_select").set_value("capacity").run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)
    assert at.get("plotly_chart")


def test_experiments_run_on_demand(monkeypatch):
    import alns_constants as C_
    monkeypatch.setattr(C_, "SCALING_N", (10, 20))
    at = _run(n_slider=15, budget_select=5000)
    for key, flag in (("methods_start", "methods_on"), ("adaptive_start", "adaptive_on"), ("sisr_start", "sisr_on"),
                       ("destroy_start", "destroy_on"), ("repair_start", "repair_on")):
        next(b for b in at.button if b.key == key).click().run()
        _ok(at)
        assert at.session_state[flag]


def test_footer_and_grenzen_are_present():
    at = _run(budget_select=5000)
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
