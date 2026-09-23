"""ALNS - Adaptive Large Neighborhood Search: destroy + repair statt kleiner Züge - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Elftes und letztes Stück der Trajektorien-Metaheuristiken-Linie der "Konzepte"-Reihe - der Konvergenzpunkt des
Nachbarschafts-Zweigs UND der ILS/VNS-Kette. Statt eines kleinen Zugs (Relocate/Swap/2-opt*/CROSS-exchange, siehe
vrp-nachbarschaften-demo) reißt ALNS in jeder Iteration einen GROSSEN Teil der Lösung ein (destroy, k Kunden
entfernt) und baut ihn adaptiv wieder auf (repair) - schlägt das die kleine Nachbarschaft bei gleichem Budget?
Muss gemessen werden - order-batching-demo hat ALNS auf einem anderen Vehikel bereits einmal verworfen.

Lauffähig mit: streamlit run app.py
"""

from dataclasses import replace

import streamlit as st

import alns_constants as C
import alns_tour as T
from alns_evaluation import (
    SWEEP_LABELS,
    Settings,
    analyse,
    compare_adaptive,
    compare_destroy_ops,
    compare_methods,
    compare_repair_ops,
    compare_vs_fixed_sisr,
    scaling_table,
    sweep,
)
from alns_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from alns_visualization import (
    build_comparison_bar,
    build_instance,
    build_progress,
    build_routes,
    build_scaling,
    build_sweep,
    build_weights_bar,
)

st.set_page_config(page_title="ALNS – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _scaling(base):
    return scaling_table(base)


@st.cache_data(show_spinner=False)
def _compare_methods(base):
    return compare_methods(base)


@st.cache_data(show_spinner=False)
def _compare_adaptive(base):
    return compare_adaptive(base)


@st.cache_data(show_spinner=False)
def _compare_sisr(base):
    return compare_vs_fixed_sisr(base)


@st.cache_data(show_spinner=False)
def _compare_destroy(base):
    return compare_destroy_ops(base)


@st.cache_data(show_spinner=False)
def _compare_repair(base):
    return compare_repair_ops(base)


def _fmt_int(x):
    return f"{int(round(x)):,}".replace(",", ".")


st.title("💥 ALNS – destroy + repair statt kleiner Züge")
st.markdown(
    """
Statt eines kleinen Zugs (ein Kunde wandert, zwei Kunden tauschen) reißt **ALNS** (Adaptive Large Neighborhood
Search) in jeder Iteration einen GROSSEN Teil der Lösung ein - ein **Destroy**-Operator entfernt `k` Kunden aus
allen Routen - und baut ihn mit einem **Repair**-Operator wieder auf. Vier Destroy-Operatoren (Zufällig, Worst,
Shaw/Ähnlichkeit, SISR/Strings) und zwei Repair-Operatoren (Greedy, Regret-2) konkurrieren über **adaptive
Gewichte**, die lernen, welche Kombination gerade am erfolgreichsten ist. Angenommen wird nach der
Simulated-Annealing-Metropolis-Regel. Schlägt das die kleine Nachbarschaft der Schwester-Demo bei gleichem Budget?
Und bringt die Adaptivität wirklich etwas?
"""
)
st.caption(
    "Elftes und letztes Stück der Trajektorien-Metaheuristiken-Linie der \"Konzepte\"-Reihe - der Konvergenzpunkt "
    "des Nachbarschafts-Zweigs (neben [lin-kernighan-demo](https://github.com/sebastian-hanisch/lin-kernighan-demo), "
    "[dynasearch-demo](https://github.com/sebastian-hanisch/dynasearch-demo) und "
    "[vrp-nachbarschaften-demo](https://github.com/sebastian-hanisch/vrp-nachbarschaften-demo)) UND der ILS/VNS-Kette "
    "([iterated-local-search-demo](https://github.com/sebastian-hanisch/iterated-local-search-demo), "
    "[variable-neighborhood-search-demo](https://github.com/sebastian-hanisch/variable-neighborhood-search-demo)). "
    "Baut direkt auf der CVRP-Infrastruktur der Schwester-Demo auf (dieselbe Savings-Konstruktion, dieselbe kleine "
    "Nachbarschaft als Vergleichsmaßstab)."
)

with st.expander("So funktioniert ALNS", expanded=True):
    st.markdown(
        """
1. **Savings-Konstruktion** (Clarke & Wright) - derselbe Startpunkt wie die kleine Nachbarschaft, damit der
   Vergleich fair ist.
2. **Destroy**: ein Operator (roulette-wheel gewählt, falls adaptiv) entfernt `k` Kunden - Zufällig, Worst
   (teuerste Kunden zuerst), Shaw (räumlich/nach Bedarf ähnliche Gruppe) oder SISR (zusammenhängende Strings).
3. **Repair**: ein Operator fügt alle entfernten Kunden wieder ein - Greedy (zufällige Reihenfolge, jeweils
   günstigste Position) oder Regret-2 (der Kunde mit dem größten Regret zuerst).
4. **Akzeptanz**: Simulated-Annealing-Metropolis mit geometrischem Abkühlplan (wie
   [parallel-tempering-demo](https://github.com/sebastian-hanisch/parallel-tempering-demo)/
   [simulated-annealing-demo](https://github.com/sebastian-hanisch/simulated-annealing-demo)) - Verbesserungen
   immer an, Verschlechterungen mit sinkender Wahrscheinlichkeit.
5. **Gewichte lernen** (Ropke & Pisinger 2006): jeder Operator bekommt eine Belohnung (neues Gesamtbestes am
   meisten, Ablehnung nichts) - alle 100 Iterationen fließt der Segment-Durchschnitt in sein Gewicht ein.
        """
    )

if C.PRESETS:
    st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
    preset_names = list(C.PRESETS.keys())
    for row in (preset_names[:3], preset_names[3:]):
        if not row:
            continue
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            with col:
                st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    method = st.radio("Verfahren", options=["alns", "small_neighborhood"],
                       format_func=lambda m: "ALNS" if m == "alns" else "Kleine Nachbarschaft (Vergleich)",
                       key="method_select", horizontal=True,
                       help="Zum direkten Vergleich: dieselbe Instanz/Konstruktion, aber die kleine Nachbarschaft der Schwester-Demo statt destroy/repair.")
    n_stops = st.slider("Stopps", *bounds("n_slider"), key="n_slider", step=C.N_STEP, help="Anzahl der Kundenstopps (das Depot kommt dazu).")
    cluster_share = st.slider("Anteil der Stopps in Gruppen [%]", *bounds("ballung_slider"), key="ballung_slider", step=C.BALLUNG_STEP)
    capacity = st.slider("Fahrzeug-Kapazität", *bounds("capacity_slider"), key="capacity_slider", step=C.CAPACITY_STEP)
    budget = st.select_slider("Budget (bewertete Kandidaten)", options=list(C.BUDGETS), key="budget_select", format_func=_fmt_int,
                               help="Eine ALNS-Iteration kostet ungleich mehr Rechenzeit als ein einzelner Nachbarschafts-Kandidat - deshalb kleinere Zahlen als bei der Schwester-Demo, siehe Grenzen.")
    if method == "alns":
        k_percent = st.slider("Zerstörungsgröße k [% der Kunden]", *bounds("k_percent_slider"), key="k_percent_slider", step=C.K_STEP)
        adaptive = st.toggle("Adaptive Gewichte", key="adaptive_toggle", help="Aus: alle aktiven Operatoren werden gleichverteilt gewählt, ohne je aus dem Verlauf zu lernen.")
        destroy_ops = st.multiselect("Destroy-Operatoren", options=list(C.DESTROY_OPS), key="destroy_select", format_func=lambda o: C.DESTROY_LABELS[o])
        repair_ops = st.multiselect("Repair-Operatoren", options=list(C.REPAIR_OPS), key="repair_select", format_func=lambda o: C.REPAIR_LABELS[o])
        chain_seed = st.number_input("Zufalls-Seed der Kette", *bounds("chain_seed_input"), key="chain_seed_input", step=1)
    else:
        k_percent, adaptive, destroy_ops, repair_ops, chain_seed = C.DEFAULT_K_PERCENT, True, C.DESTROY_OPS, C.REPAIR_OPS, 0
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)

if method == "alns" and not destroy_ops:
    st.warning("Mindestens ein Destroy-Operator muss aktiv sein - Zufällig wird automatisch wieder aktiviert.")
    destroy_ops = ("random",)
if method == "alns" and not repair_ops:
    st.warning("Mindestens ein Repair-Operator muss aktiv sein - Greedy wird automatisch wieder aktiviert.")
    repair_ops = ("greedy",)

sync_query_params({
    "method_select": method, "n_slider": int(n_stops), "ballung_slider": int(cluster_share), "seed_input": int(seed),
    "capacity_slider": int(capacity), "budget_select": int(budget), "k_percent_slider": int(k_percent),
    "adaptive_toggle": bool(adaptive), "destroy_select": tuple(destroy_ops), "repair_select": tuple(repair_ops),
    "chain_seed_input": int(chain_seed),
})

settings = Settings(int(n_stops), int(cluster_share), int(seed), float(capacity), int(budget), int(chain_seed),
                     method, bool(adaptive), tuple(destroy_ops), tuple(repair_ops), int(k_percent))
with st.spinner("Rechne..."):
    a = _analysis(settings)
xy = a.inst.xy

# --- ALNS in Aktion ------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 ALNS in Aktion" if method == "alns" else "## 🎯 Kleine Nachbarschaft in Aktion (Vergleich)")
STEP_LABELS = {1: "1 · Instanz", 2: "2 · Konstruktion", 3: "3 · Ergebnis"}
step = st.select_slider("Schritt", options=list(STEP_LABELS), key="alns_step", format_func=lambda s: STEP_LABELS[s])

if step == 1:
    st.markdown(f"**{a.inst.n} Kundenstopps und das Depot (Stern)** – {a.inst.cluster_share} % der Stopps in Gruppen, Gesamtbedarf {a.inst.total_demand:.0f}, Kapazität {settings.capacity:.0f}")
    st.plotly_chart(build_instance(xy), width="stretch", key="s1_map")
elif step == 2:
    st.markdown(f"**Savings-Konstruktion**: {len(a.construction_routes)} Routen, {a.construction_cost:.1f} km")
    st.plotly_chart(build_routes(xy, a.construction_routes), width="stretch", key="s2_map")
else:
    routes = a.run.best_routes if method == "alns" else a.run.routes
    st.markdown(f"**Nach der Suche**: {a.n_routes} Routen, {a.cost:.1f} km – {a.improvement:.2f} % besser als die Konstruktion")
    st.plotly_chart(build_routes(xy, routes), width="stretch", key="s3_map")
    if method == "alns":
        st.plotly_chart(build_progress(a.run.cost_history, a.run.best_history), width="stretch", key="s3_progress")
        st.caption("Aktuell (angenommene Lösung, blau) gegen bisher bestes (rot) über die Iterationen - Metropolis lässt zwischendurch Verschlechterungen zu, `bestes` fällt monoton.")
        st.plotly_chart(build_weights_bar(a.run.destroy_weights, C.DESTROY_LABELS), width="stretch", key="s3_dw")
        st.plotly_chart(build_weights_bar(a.run.repair_weights, C.REPAIR_LABELS), width="stretch", key="s3_rw")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die Suche gefunden hat")
st.caption("**Verbesserung:** Prozent gegenüber der Savings-Konstruktion (keine untere Schranke wie bei der TSP-Wurzel, siehe README).")
m1, m2, m3 = st.columns(3)
m1.metric("Verbesserung ggü. Konstruktion", f"{a.improvement:.2f} %")
m2.metric("Routen", f"{a.n_routes}", delta=f"aus {len(a.construction_routes)} nach Konstruktion", delta_color="off")
m3.metric("Länge (km)", f"{a.cost:.1f}", delta=f"Konstruktion {a.construction_cost:.1f}", delta_color="off")

d1, d2 = st.columns(2)
with d1:
    st.markdown("**Kennzahlen im Detail**")
    st.table({"": ["Länge (km)", "Routen", "Bewertete Kandidaten"],
              "Konstruktion": [f"{a.construction_cost:.1f}", f"{len(a.construction_routes)}", "-"],
              "Nach der Suche": [f"{a.cost:.1f}", f"{a.n_routes}", _fmt_int(a.evaluations)]})
with d2:
    routes = a.run.best_routes if method == "alns" else a.run.routes
    st.markdown("**Routendemand**")
    loads = [T.route_demand(r, a.inst.demands) for r in routes]
    st.table({"Route": [f"{i + 1}" for i in range(len(loads))], "Bedarf": [f"{l:.0f}" for l in loads],
              "Kapazität": [f"{settings.capacity:.0f}" for _ in loads]})

st.markdown("---")

# --- Sweeps ------------------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von den Reglern ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
base_sweep = replace(settings, seed=0)
if st.button("Sweep über 5 feste Instanzen berechnen (kann je nach Budget einige Minuten dauern)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base_sweep)}
if (sweep_param, base_sweep) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Instanzen..."):
        rows_sweep = _sweep(sweep_param, base_sweep)
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_chart")
    st.caption("Mittel über 5 feste Instanzen (Seeds 100000–100004); bei ALNS zusätzlich über 3 Ketten-Seeds je Instanz.")

st.markdown("---")

# --- Experimente --------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Schlägt ALNS die kleine Nachbarschaft?")
if st.button("ALNS gegen kleine Nachbarschaft vergleichen (kann einige Minuten dauern)", key="methods_start"):
    st.session_state["methods_on"] = True
if st.session_state.get("methods_on"):
    with st.spinner("Rechne beide Verfahren über 5 Instanzen..."):
        rows_m = _compare_methods(base_sweep)
    labels = list(rows_m.keys())
    st.plotly_chart(build_comparison_bar(labels, [rows_m[l]["improvement"] for l in labels]), width="stretch", key="methods_chart")
    st.caption("Mittel über 5 feste Instanzen, gleiches (deklariertes) Budget, dieselbe Savings-Startlösung.")

st.markdown("---")

st.subheader("🔬 Bringt die Adaptivität etwas?")
if st.button("Adaptiv gegen gleichverteilt vergleichen (kann einige Minuten dauern)", key="adaptive_start"):
    st.session_state["adaptive_on"] = True
if st.session_state.get("adaptive_on"):
    with st.spinner("Rechne beide Varianten über 5 Instanzen × 3 Ketten..."):
        rows_a = _compare_adaptive(base_sweep)
    labels = list(rows_a.keys())
    st.plotly_chart(build_comparison_bar(labels, [rows_a[l]["improvement"] for l in labels]), width="stretch", key="adaptive_chart")

st.markdown("---")

st.subheader("🔬 ALNS gegen festes SISR-Ruin")
if st.button("Vollen Operator-Pool gegen festes SISR vergleichen (kann einige Minuten dauern)", key="sisr_start"):
    st.session_state["sisr_on"] = True
if st.session_state.get("sisr_on"):
    with st.spinner("Rechne beide Varianten über 5 Instanzen × 3 Ketten..."):
        rows_s = _compare_sisr(base_sweep)
    labels = list(rows_s.keys())
    st.plotly_chart(build_comparison_bar(labels, [rows_s[l]["improvement"] for l in labels]), width="stretch", key="sisr_chart")

st.markdown("---")

st.subheader("🔬 Welcher Destroy-/Repair-Operator trägt am meisten bei?")
c1, c2 = st.columns(2)
with c1:
    if st.button("Destroy-Operatoren einzeln vergleichen (kann einige Minuten dauern)", key="destroy_start"):
        st.session_state["destroy_on"] = True
    if st.session_state.get("destroy_on"):
        with st.spinner("Rechne jeden Destroy-Operator einzeln + alle zusammen..."):
            rows_d = _compare_destroy(base_sweep)
        labels = list(rows_d.keys())
        st.plotly_chart(build_comparison_bar(labels, [rows_d[l]["improvement"] for l in labels]), width="stretch", key="destroy_chart")
with c2:
    if st.button("Repair-Operatoren vergleichen (kann einige Minuten dauern)", key="repair_start"):
        st.session_state["repair_on"] = True
    if st.session_state.get("repair_on"):
        with st.spinner("Rechne Greedy gegen Regret-2..."):
            rows_r = _compare_repair(base_sweep)
        labels = list(rows_r.keys())
        st.plotly_chart(build_comparison_bar(labels, [rows_r[l]["improvement"] for l in labels]), width="stretch", key="repair_chart")

st.markdown("---")

st.subheader("🔬 Skalierung: wie verändert sich die Verbesserung mit der Instanzgröße?")
if st.button(f"Stopps von {C.SCALING_N[0]} bis {C.SCALING_N[-1]} durchfahren (kann mehrere Minuten dauern)", key="scaling_start"):
    st.session_state["scaling_on"] = True
if st.session_state.get("scaling_on"):
    with st.spinner("Rechne mehrere Größen × 2 Budgetregeln × 5 Instanzen..."):
        sc = _scaling(replace(base_sweep, n=C.DEFAULT_N))
    st.plotly_chart(build_scaling(sc), width="stretch", key="scaling_chart")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Budget zählt ALNS-Iterationen grobkörniger als die kleine Nachbarschaft** | Ein ALNS-Durchlauf zählt als `k` (Zerstörungsgröße) bewertete Kandidaten - nicht als die tatsächliche Zahl der Einfüge-Kostenvergleiche. Ein direkter "gleiches Budget"-Vergleich ist deshalb eine bewusste Vereinfachung, kein exakter Rechenzeit-Vergleich. | (kein Nachfolger nötig - dokumentierte, bewusste Konvention) |
| **SISR ist vereinfacht** | Keine Bevorzugung räumlich benachbarter Routen bei der String-Auswahl, keine adaptive Stringlängen-Verteilung wie im Original-Papier - der für die Operator-Ablation entscheidende Unterschied (zusammenhängende Strings statt Einzelkunden) bleibt erhalten. | (kein Nachfolger nötig - dokumentierte Vereinfachung) |
| **Nur Metropolis-Akzeptanz** | Wie bei parallel-tempering-demo literaturtreu (Ropke & Pisinger 2006 selbst nutzen dieselbe Regel), nicht willkürlich. | (kein Nachfolger nötig) |
| **Kleinere Budget-/Stopp-Bereiche als die Schwester-Demo** | Eine ALNS-Iteration kostet ungleich mehr Rechenzeit als ein einzelner Nachbarschafts-Kandidat - realistische Bereiche für dieses Stück, nicht 2 Millionen wie bei der TSP-Wurzel. | (kein Nachfolger nötig) |
| **Keine untere Schranke** | Wie bei der Schwester-Demo relativ zur Savings-Konstruktion, nicht zum Optimum. | (kein Nachfolger nötig) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Akzeptanz (Metropolis).** $\Delta \le 0$: immer an. $\Delta > 0$: an mit $p = \exp(-\Delta/T)$, $T$ fällt
geometrisch je Iteration.

**Adaptive Gewichte (Ropke & Pisinger 2006).** Operator $i$ wird mit $p_i = w_i / \sum_j w_j$ gewählt (roulette-
wheel). Alle `segment_length` Iterationen: $w_i \leftarrow w_i(1-\rho) + \rho \cdot \bar{\pi}_i$ (Segment-Mittel
der Belohnungen $\pi_i$, $\rho$ = Reaktionsfaktor).

**Regret-2.** Für jeden noch nicht eingefügten Kunden $c$: $\text{regret}(c) = \text{Kosten}_2(c) -
\text{Kosten}_1(c)$ (zweitbeste minus beste Einfügeposition über alle Routen). Der Kunde mit dem größten Regret
wird zuerst eingefügt.

**Literatur.** Ropke, S., & Pisinger, D. (2006). *An Adaptive Large Neighborhood Search Heuristic for the Pickup
and Delivery Problem with Time Windows.* Transportation Science, 40(4), 455-472. Shaw, P. (1998). *Using
Constraint Programming and Local Search Methods to Solve Vehicle Routing Problems.* CP98, LNCS 1520, 417-431.
Christiaens, J., & Vanden Berghe, G. (2020). *Slack Induction by String Removals for Vehicle Routing Problems.*
Transportation Science, 54(2), 417-433.

Implementiert in `alns_destroy.py` (vier Destroy-Operatoren), `alns_repair.py` (zwei Repair-Operatoren),
`alns_algorithm.py` (Metropolis, adaptive Gewichte, Hauptschleife), `alns_evaluation.py` (Kennzahlen, Sweeps,
Vergleiche).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
