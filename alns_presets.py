"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Buttons (Standardmuster aus dem Demo-Portfolio, siehe
vrpn_presets.py in vrp-nachbarschaften-demo)."""

import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import alns_constants as C

METHODS = ("alns", "small_neighborhood")


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _int_choice(options):
    def cast(value):
        value = int(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


def _bool_from_str(value):
    return str(value).strip().lower() in ("1", "true", "yes", "ja")


def _bool_to_str(value):
    return "1" if value else "0"


def _ops_from_str(all_ops):
    def cast(value):
        parts = [p for p in str(value).split(",") if p]
        ops = tuple(o for o in all_ops if o in parts)
        if not ops:
            raise ValueError(value)
        return ops
    return cast


def _ops_to_str(ops):
    return ",".join(ops)


def _method_from_str(value):
    value = str(value)
    if value not in METHODS:
        raise ValueError(value)
    return value


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "ballung_slider": SettingSpec("ballung", int, C.DEFAULT_BALLUNG, C.BALLUNG_MIN, C.BALLUNG_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "capacity_slider": SettingSpec("capacity", int, C.DEFAULT_CAPACITY, C.CAPACITY_MIN, C.CAPACITY_MAX),
    "budget_select": SettingSpec("budget", _int_choice(C.BUDGETS), C.DEFAULT_BUDGET),
    "chain_seed_input": SettingSpec("chain", int, C.DEFAULT_CHAIN_SEED, 0, C.SEED_MAX),
    "k_percent_slider": SettingSpec("k", int, C.DEFAULT_K_PERCENT, C.K_MIN, C.K_MAX),
    "adaptive_toggle": SettingSpec("adaptive", _bool_from_str, C.DEFAULT_ADAPTIVE),
    "destroy_select": SettingSpec("destroy", _ops_from_str(C.DESTROY_OPS), list(C.DESTROY_OPS)),
    "repair_select": SettingSpec("repair", _ops_from_str(C.REPAIR_OPS), list(C.REPAIR_OPS)),
    "method_select": SettingSpec("method", _method_from_str, "alns"),
}
PRESET_KEYS = {
    "n": "n_slider", "ballung": "ballung_slider", "seed": "seed_input", "capacity": "capacity_slider",
    "budget": "budget_select", "k_percent": "k_percent_slider", "adaptive": "adaptive_toggle",
    "destroy_ops": "destroy_select", "repair_ops": "repair_select", "method": "method_select",
}
STEPS = {"n_slider": C.N_STEP, "ballung_slider": C.BALLUNG_STEP, "capacity_slider": C.CAPACITY_STEP, "k_percent_slider": C.K_STEP}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            lo = SETTING_SPECS[key].lo
            st.session_state[key] = int(lo + round((st.session_state[key] - lo) / step) * step)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            if state_key in ("destroy_select", "repair_select"):
                value = _ops_to_str(value)
            elif state_key == "adaptive_toggle":
                value = _bool_to_str(value)
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
