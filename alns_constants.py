"""Konstanten der ALNS-Demo: Szenario (geometrische Basis + Bedarfe/Kapazität, wortgleich zur
VRP-Nachbarschaften-Demo), Regler, Beschriftungen (ALNS-eigene Messwerte + Presets folgen nach den Messungen)."""

AREA = 100.0                     # Kantenlänge des Gebiets in km
N_CLUSTERS = 5
CLUSTER_SIGMA = 6.0              # Streuung einer Gruppe in km
CLUSTER_MARGIN = 12.0            # Gruppenmittelpunkte liegen mindestens so weit vom Rand entfernt
SWEEP_SEEDS = tuple(range(100000, 100005))
# 2 statt 3 Ketten-Seeds je Instanz (kleiner als sonst in dieser Linie üblich): eine ALNS-Iteration kostet
# ungleich mehr Rechenzeit als ein einzelner Nachbarschafts-Kandidat (siehe alns_algorithm.py) - bei 5 festen
# Instanzen x 2 Ketten bleiben die Sweeps/Vergleiche noch interaktiv nutzbar.
SWEEP_CHAINS = 2
BOUND_ITERATIONS = 300

N_MIN, N_MAX, DEFAULT_N, N_STEP = 10, 150, 60, 5
BALLUNG_MIN, BALLUNG_MAX, DEFAULT_BALLUNG, BALLUNG_STEP = 0, 100, 0, 25
SEED_MAX = 999999
DEFAULT_SEED = 35
DEFAULT_CHAIN_SEED = 0
DEFAULT_BUDGET = 50000
# Kleinere Budget-/Stopp-Bereiche als die Schwester-Demo: EIN ALNS-Vorschlag (destroy+repair) kostet ungleich
# mehr Rechenzeit als EIN einzelner 2-opt-/Relocate-Kandidat (siehe alns_algorithm.py) - bei gleicher
# Wall-Clock-Zeit sind hier deshalb kleinere Budget-Zahlen realistisch, nicht 2 Millionen wie bei der Wurzel.
# DEFAULT_BUDGET MUSS Mitglied von BUDGETS sein (st.select_slider snappt sonst stillschweigend auf den ersten
# Wert - genau dieser Fehler wurde beim Bau hier von test_app.py::test_permalink_values_are_clamped_and_snapped
# gefangen).
BUDGETS = (5000, 10000, 25000, 50000, 100000, 200000)
SCALING_N = (20, 40, 60, 100, 150)

# --- VRP-eigen: Bedarfe und Kapazität (wortgleich zur VRP-Nachbarschaften-Demo) -------------------------------------------------------------
DEMAND_MIN, DEMAND_MAX = 1, 9      # rng.integers(DEMAND_MIN, DEMAND_MAX + 1) je Kunde
CAPACITY_MIN, CAPACITY_MAX, DEFAULT_CAPACITY, CAPACITY_STEP = 15, 600, 120, 5
# Kapazität 120 als Voreinstellung (nicht 60 wie die Schwester-Demo): dort war 120 das gemessene Optimum des
# Inter-Route-Nachbarschaftswerts - hier der Kapazitätswert, bei dem sich ALNS am ehesten von der kleinen
# Nachbarschaft absetzen können sollte (mehr Routen zum Umschichten als bei sehr großer Kapazität).
MAX_SEGMENT = 3                    # Segmentlänge für CROSS-exchange in der kleinen Nachbarschaft, wie die Wurzel

# --- ALNS-eigen: Regler ---------------------------------------------------------------------------------------------------------------------
K_MIN, K_MAX, DEFAULT_K_PERCENT, K_STEP = 5, 50, 20, 5     # Zerstörungsgröße als Prozent der Kundenzahl
DEFAULT_ADAPTIVE = True
REACTION_FACTOR = 0.2              # Ropke & Pisinger 2006: Gewichts-Update-Trägheit
SEGMENT_LENGTH = 100                # Iterationen je Gewichts-Update-Segment
REWARD_BEST, REWARD_BETTER, REWARD_ACCEPTED, REWARD_REJECTED = 33.0, 9.0, 3.0, 0.0   # Ropke & Pisinger 2006, Tabelle
SISR_MAX_STRING_LENGTH = 10        # Christiaens & Vanden Berghe 2020: max. Stringlänge je entfernter Route
SHAW_DISTANCE_WEIGHT, SHAW_DEMAND_WEIGHT = 9.0, 1.0        # Shaw 1998: relative Gewichtung im Ähnlichkeitsmaß
WORST_REMOVAL_P = 3                 # Ropke & Pisinger 2006: p-te Potenz beim "worst removal"-Zufallsrauschen
DEFAULT_T0_FACTOR = 0.10            # Start-Temperatur als Vielfaches der Savings-Startkosten (siehe Kalibrierung)
DEFAULT_COOLING = 0.9997            # geometrischer Abkühlfaktor je Iteration (siehe Kalibrierung)

DESTROY_OPS = ("random", "worst", "shaw", "sisr")
REPAIR_OPS = ("greedy", "regret2")
DESTROY_LABELS = {"random": "Zufällig", "worst": "Worst (teuerste Kunden)", "shaw": "Shaw (Ähnlichkeit)", "sisr": "SISR (Strings)"}
REPAIR_LABELS = {"greedy": "Greedy", "regret2": "Regret-2"}

# --- Gemessene Werte (Mittel über 5 feste Sweep-Instanzen x 2 Ketten-Seeds, Seeds 100000-100004; n=60,
# --- Kapazität 120, Budget 50 Tausend, sofern nicht anders angegeben; 2026-09-23, alle Werte über
# --- ev.run_config/ev.sweep/ev.compare_*/ev.scaling_table nachgerechnet, s. tests/test_claims.py) -----------------
# ZENTRALE FRAGE - ALNS gegen die kleine Nachbarschaft der Schwester-Demo, gleiches Budget, dieselbe
#   Savings-Startlösung: ALNS 8.89 %, kleine Nachbarschaft 2.28 % - ALNS klar besser (über jeden getesteten
#   Kapazitäts-Punkt hinweg, nicht nur im Standardfall, siehe README/tests/test_claims.py).
# ADAPTIVITÄT - NICHT bestätigte Vorab-Hypothese: adaptive Gewichte 8.89 % gegen gleichverteilte Zufallswahl
#   8.87 % - praktisch eine Krawatte. Wenn alle aktiven Operatoren brauchbar sind, bringt das Lernen, WELCHER
#   gerade am besten passt, hier kaum etwas - ein Echo der order-batching-demo-Vorgeschichte (ALNS dort insgesamt
#   verworfen), hier trägt zumindest destroy+repair selbst klar, nur die Adaptivität on top kaum.
# ALNS (voller, adaptiver Pool) gegen FESTES SISR-Ruin (nur SISR+Greedy, keine Gewichtsanpassung nötig):
#   8.89 % gegen 8.79 % - nah beieinander, der volle Pool hat einen kleinen, aber messbaren Vorteil.
# DESTROY-OPERATOR-ABLATION (je einzeln + alle zusammen, adaptiv): Zufällig 8.60 %, Worst 8.52 %, Shaw 8.84 %,
#   SISR 8.88 %, alle zusammen 8.89 % - SISR ALLEIN erreicht fast den vollen Wert; **Worst ist überraschend der
#   SCHWÄCHSTE Einzeloperator** (sogar schwächer als Zufällig) - die "teuerste Kunden zuerst"-Heuristik hilft
#   hier nicht, vermutlich weil sie wiederholt dieselben Kunden entfernt (Zufalls-Rauschen mildert das nur
#   teilweise).
# REPAIR-OPERATOR-VERGLEICH: Greedy 8.95 % gegen Regret-2 8.51 % - **Greedy schlägt Regret-2**, gegen die
#   verbreitete Literatur-Intuition ("Regret-Sortierung ist meist besser"). Nicht angenommen, sondern gemessen.
# ZERSTÖRUNGSGRÖSSE-SWEEP (k als % der Kunden): 5/10/20/30/50: 7.83/8.48/8.89/8.87/8.88 % - steigt bis ~20 %,
#   danach ein flaches Plateau (kein scharfes Optimum wie sonst in dieser Linie üblich, aber ein klarer
#   Schwellenwert: zu kleines k lässt zu wenig Spielraum für sinnvollen Wiederaufbau).
# BUDGET-SWEEP: 5T/10T/25T/50T/100T/200T: 7.74/8.66/8.83/8.89/8.93/8.93 % - konvergiert bei ~100 Tausend.
# KAPAZITÄTS-SWEEP: 15/30/60/120/250/600 (Routenzahl 22.8/11.4/5.8/3.0/2.0/1.0): 1.04/2.85/3.94/8.89/7.19/6.57 % -
#   NICHT monoton, Optimum bei Kapazität 120 (3 Routen) - derselbe Kapazitäts-Sweet-Spot wie die Schwester-Demo,
#   hier aber mit einem viel größeren Ausschlag (8.89 % gegen deren 2.47 %).
# SKALIERUNG (Kapazität 120 fest), 20/40/60/100/150 Stopps: bei festem 50-Tausend-Budget 2.04/6.17/8.89/9.52/
#   6.59 %; bei wachsendem Budget (1000 x Stopps) 1.71/6.17/8.89/9.70/7.66 % - AUCH nicht monoton, Optimum bei
#   n=100 (4.8 Routen im Mittel). Derselbe zugrundeliegende Treiber wie der Kapazitäts-Sweep: die Routenzahl, nicht
#   die Stoppzahl selbst - zu wenige Routen (n=20, 1 Route) lassen nichts zum Umverteilen, zu viele (n=150, 7
#   Routen) verdünnen das feste k-Kunden-Budget jeder Destroy+Repair-Runde über immer mehr Routen.

PRESETS = {
    "Standardfall (Voreinstellung)": {
        "n": 60, "ballung": 0, "seed": 35, "capacity": 120, "budget": 50000, "k_percent": 20,
        "adaptive": True, "destroy_ops": list(DESTROY_OPS), "repair_ops": list(REPAIR_OPS), "method": "alns",
    },
    "Kleine Nachbarschaft (Vergleich)": {
        "n": 60, "ballung": 0, "seed": 35, "capacity": 120, "budget": 50000, "k_percent": 20,
        "adaptive": True, "destroy_ops": list(DESTROY_OPS), "repair_ops": list(REPAIR_OPS), "method": "small_neighborhood",
    },
    "Nicht-adaptiv (Kontrolle)": {
        "n": 60, "ballung": 0, "seed": 35, "capacity": 120, "budget": 50000, "k_percent": 20,
        "adaptive": False, "destroy_ops": list(DESTROY_OPS), "repair_ops": list(REPAIR_OPS), "method": "alns",
    },
    "Festes SISR-Ruin": {
        "n": 60, "ballung": 0, "seed": 35, "capacity": 120, "budget": 50000, "k_percent": 20,
        "adaptive": False, "destroy_ops": ["sisr"], "repair_ops": ["greedy"], "method": "alns",
    },
    "Kleine Kapazität (viele Routen)": {
        "n": 60, "ballung": 0, "seed": 35, "capacity": 15, "budget": 50000, "k_percent": 20,
        "adaptive": True, "destroy_ops": list(DESTROY_OPS), "repair_ops": list(REPAIR_OPS), "method": "alns",
    },
    "Mittlere Stoppzahl (Skalierungs-Optimum)": {
        "n": 100, "ballung": 0, "seed": 35, "capacity": 120, "budget": 50000, "k_percent": 20,
        "adaptive": True, "destroy_ops": list(DESTROY_OPS), "repair_ops": list(REPAIR_OPS), "method": "alns",
    },
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "60 Stopps, Kapazität 120 (3 Routen), 50 Tausend Vorschläge: ALNS verbessert die Savings-Konstruktion im Mittel um 8.89 % - die kleine Nachbarschaft der Schwester-Demo bei gleichem Budget nur um 2.28 %.",
    "Kleine Nachbarschaft (Vergleich)": "Dieselbe Instanz und dasselbe Budget, aber die kleine Nachbarschaft (Relocate/Swap/2-opt*/CROSS-exchange) statt destroy/repair: nur 2.28 % statt 8.89 % - der zentrale Befund dieses Stücks.",
    "Nicht-adaptiv (Kontrolle)": "Dieselbe Instanz, aber Destroy-/Repair-Operatoren gleichverteilt statt gelernt gewählt: 8.87 % statt 8.89 % - praktisch kein Unterschied. Die Adaptivität trägt hier kaum etwas.",
    "Festes SISR-Ruin": "Nur der SISR-String-Operator + Greedy-Repair, ohne jede Gewichtsanpassung: 8.79 % - nah am vollen, adaptiven Operator-Pool (8.89 %).",
    "Kleine Kapazität (viele Routen)": "Kapazität 15 (im Mittel knapp 23 winzige Routen): nur 1.04 % - wie bei der Schwester-Demo sind sehr kleine Routen auch für ALNS zu klein für nennenswerte Umbauten.",
    "Mittlere Stoppzahl (Skalierungs-Optimum)": "100 Stopps statt 60 (im Mittel 4.8 Routen): 9.52 % - das gemessene Optimum der Skalierung, mehr als bei kleineren UND bei größeren Instanzen (nicht-monoton, wie der Kapazitäts-Sweep).",
}
# Beobachtete Spannweite der Verbesserung ggü. Konstruktion über die 5 festen Sweep-Instanzen (je EIN
# Ketten-Seed=0-Lauf, mit Sicherheitsabstand) - ALNS hat wieder Zufall im Kern (anders als die deterministische
# Schwester-Demo), die Spannweite kommt deshalb sowohl aus der Instanz-Geometrie als auch aus der Suche selbst.
PRESET_EXPECTED_BANDS = {
    "Standardfall (Voreinstellung)": (2.0, 16.0),
    "Kleine Nachbarschaft (Vergleich)": (0.0, 7.0),
    "Nicht-adaptiv (Kontrolle)": (2.0, 16.0),
    "Festes SISR-Ruin": (2.0, 16.0),
    "Kleine Kapazität (viele Routen)": (0.0, 5.0),
    "Mittlere Stoppzahl (Skalierungs-Optimum)": (3.0, 14.0),
}
