# ALNS – Adaptive Large Neighborhood Search – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-alns-demo.streamlit.app/)**

Elftes und letztes Stück der **Trajektorien-Metaheuristiken-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning" - der **Konvergenzpunkt** des Nachbarschafts-Zweigs ([lin-kernighan-demo](https://sebastianhanisch-lin-kernighan-demo.streamlit.app/), [dynasearch-demo](https://sebastianhanisch-dynasearch-demo.streamlit.app/), [vrp-nachbarschaften-demo](https://sebastianhanisch-vrp-nachbarschaften-demo.streamlit.app/)) UND der ILS/VNS-Kette ([iterated-local-search-demo](https://sebastianhanisch-iterated-local-search-demo.streamlit.app/), [variable-neighborhood-search-demo](https://sebastianhanisch-variable-neighborhood-search-demo.streamlit.app/)).

**Einordnung in die Reihe:** statt eines kleinen Zugs (ein Kunde wandert, zwei Kunden tauschen - [vrp-nachbarschaften-demo](https://sebastianhanisch-vrp-nachbarschaften-demo.streamlit.app/)) reißt **ALNS** in jeder Iteration einen GROSSEN Teil der Lösung ein (**Destroy**: `k` Kunden entfernt) und baut ihn **adaptiv** wieder auf (**Repair**). Vier Destroy-Operatoren (Zufällig, Worst, Shaw/Ähnlichkeit, SISR/Strings), zwei Repair-Operatoren (Greedy, Regret-2), Operator-Gewichte lernen aus dem Verlauf (Ropke & Pisinger 2006), Akzeptanz nach der Simulated-Annealing-Metropolis-Regel. Baut auf derselben CVRP-Infrastruktur wie die Schwester-Demo auf (dieselbe Savings-Konstruktion, dieselbe kleine Nachbarschaft als Vergleichsmaßstab).

```
Hill Climbing (Wurzel: nur bergab, bleibt im ersten Optimum stecken)                     [gebaut]
  ├─ Simulated Annealing ── Parallel Tempering                                           [gebaut]
  ├─ Iterated Local Search ── Variable Neighborhood Search ──┐                           [gebaut]
  ├─ Tabu Search                                             │                           [gebaut]
  ├─ GRASP                                                   │                           [gebaut]
  └─ Nachbarschafts-Zweig                                    │
        ├─ Lin-Kernighan (variable Tiefe)                    │                           [gebaut]
        ├─ Dynasearch (viele unabhängige Züge auf einmal)    │                           [gebaut]
        └─ VRP-Nachbarschaften (Züge zwischen Routen) ───────┤                           [gebaut]
                                                               ▼
                                ALNS (Destroy + Repair, adaptive Gewichte)          [dieses Stück]
```
(Tabu Search und GRASP hängen nicht selbst am Konvergenzpunkt - die durchlaufende Linie ist nur die Fortsetzung des VNS-Zweigs nach unten zu ALNS.)

Ergebnis in Kürze: **ALNS schlägt die kleine Nachbarschaft der Schwester-Demo klar** - 8.89 % gegenüber 2.28 % Verbesserung bei gleichem (deklariertem) Budget. Aber die zweite Vorab-Hypothese trägt **nicht**: **adaptive Operator-Gewichte bringen praktisch nichts** gegenüber gleichverteilter Zufallswahl der Operatoren (8.89 % gegen 8.87 % - eine Krawatte, kein Unterschied) - wenn alle aktiven Operatoren brauchbar sind, lohnt sich das Lernen, WELCHER gerade am besten passt, hier kaum. Ein Echo der `order-batching-demo`-Vorgeschichte, die ALNS insgesamt bereits einmal verworfen hat - hier trägt zumindest der Destroy+Repair-Mechanismus selbst klar, nur die Adaptivität on top nicht.

| Frage | Ergebnis (60 gleichverteilte Stopps, Kapazität 120, 50 Tausend Vorschläge, sofern nicht anders angegeben; Mittel über 5 feste Instanzen, Seeds 100000–100004, bei ALNS zusätzlich über 2 Ketten-Seeds; Verbesserung = Prozent ggü. der Savings-Konstruktion) |
|---|---|
| **ALNS gegen kleine Nachbarschaft** | ✅ ALNS **8.89 %** gegen **2.28 %** - klar besser bei gleichem Budget |
| **Adaptiv gegen gleichverteilt** | ⚠️ **8.89 %** gegen **8.87 %** - praktisch kein Unterschied (Krawatte) |
| **ALNS gegen festes SISR-Ruin** | ✅ voller Pool **8.89 %** gegen **8.79 %** - nah beieinander, kleiner Vorteil für den vollen Pool |
| **Destroy-Operator-Ablation** | ⚠️ Zufällig 8.60 %, **Worst 8.52 % (schwächster!)**, Shaw 8.84 %, **SISR 8.88 %** (fast der volle Wert), alle 8.89 % |
| **Repair-Operator-Vergleich** | ⚠️ **Greedy 8.95 %** schlägt **Regret-2 8.51 %** - gegen die übliche Literatur-Intuition |
| **Zerstörungsgröße-Sweep (k, % der Kunden)** | ✅ 5/10/20/30/50: 7.83/8.48/**8.89**/8.87/8.88 % - Plateau ab ~20 % |
| **Budget-Sweep** | ✅ 5T/10T/25T/50T/100T/200T: 7.74/8.66/8.83/8.89/**8.93**/8.93 % - konvergiert bei ~100 Tausend |
| **Kapazitäts-Sweep** | ⚠️ 15/30/60/120/250/600: 1.04/2.85/3.94/**8.89**/7.19/6.57 % - NICHT monoton, Optimum bei Kapazität 120 (3 Routen) |
| **Skalierung (50-Tausend-Budget fest)** | ⚠️ 20/40/60/100/150 Stopps: 2.04/6.17/8.89/**9.52**/6.59 % - AUCH nicht monoton, Optimum bei n=100 (4.8 Routen) - derselbe Routenzahl-Treiber wie der Kapazitäts-Sweep |

## Was die Demo zeigt

1. **ALNS in Aktion** (Schritt-Slider): **Instanz** → **Konstruktion** (Savings-Routen) → **Ergebnis** (Routen nach der Suche, Verlaufskurve aktuell/bestes, Operator-Gewichte am Ende).
2. **Was die Suche gefunden hat:** Verbesserung ggü. Konstruktion, Routenzahl, Länge, Routendemand je Route.
3. **📐 Sweeps** über Kapazität, Budget, Stopps und Zerstörungsgröße (5 feste Instanzen ab Seed 100000).
4. **🔬 Experimente auf Abruf:** ALNS gegen kleine Nachbarschaft, adaptiv gegen gleichverteilt, ALNS gegen festes SISR-Ruin, Destroy-/Repair-Operator-Vergleiche, Skalierung.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (Budget-Konvention, vereinfachtes SISR, nur Metropolis, kleinere Regler-Bereiche, keine untere Schranke).

Regler: Verfahren (ALNS / kleine Nachbarschaft zum direkten Vergleich), Stopps (10–150), Anteil der Stopps in Gruppen, Fahrzeug-Kapazität (15–600), Budget (5 Tausend bis 200 Tausend bewertete Kandidaten - kleinere Zahlen als die Schwester-Demo, siehe Grenzen), **Zerstörungsgröße k** (5–50 % der Kunden), **adaptive Gewichte** (an/aus), **Destroy-/Repair-Operatoren** (Mehrfachauswahl), Seed der Instanz (+ 🎲), Seed der Kette (ALNS hat wieder Zufall im Kern - anders als die deterministische Schwester-Demo).

## Messwerte der Presets

| Preset | Verbesserung (Instanz-Seed 35, Ketten-Seed 0) |
|---|---|
| Standardfall (Voreinstellung) | 7.28 % (Sweep-Mittel 8.89 %) |
| Kleine Nachbarschaft (Vergleich) | 4.22 % (Sweep-Mittel 2.28 %) |
| Nicht-adaptiv (Kontrolle) | 7.28 % - praktisch identisch zum Standardfall |
| Festes SISR-Ruin | 7.28 % - bei diesem Seed sogar bytegleich zum Standardfall |
| Kleine Kapazität (viele Routen) | 2.17 % (Sweep-Mittel 1.04 %) |
| Mittlere Stoppzahl (Skalierungs-Optimum) | 4.42 % (Sweep-Mittel 9.52 %) |

ALNS hat (anders als die deterministische Schwester-Demo) wieder Zufall im Kern - die einzelne Standardinstanz weicht deshalb spürbar von den Sweep-Mittelwerten ab (z. B. "Mittlere Stoppzahl" 4.42 % statt 9.52 %, "Kleine Nachbarschaft" sogar in die andere Richtung: 4.22 % statt 2.28 %) - die Mittelwerte oben in der Ergebnis-Tabelle sind die belastbaren Zahlen; jedes Preset prüft sich zusätzlich über die 5 festen Sweep-Instanzen gegen eine gemessene Spannweite (siehe `tests/test_presets.py`).

## Modell und Verfahren

- **CVRP-Infrastruktur** (`alns_scenario.py`, `alns_tour.py`, `alns_construction.py`, `alns_interroute.py`): wortgleiche Kopien aus [vrp-nachbarschaften-demo](https://sebastianhanisch-vrp-nachbarschaften-demo.streamlit.app/) - Instanz, Routendarstellung, Clarke-&-Wright-Savings-Konstruktion UND die kleine Nachbarschaft (Relocate/Swap/2-opt\*/CROSS-exchange) als Vergleichsmaßstab dieses Stücks.
- **Destroy-Operatoren** (`alns_destroy.py`): Zufällig, Worst (Ropke & Pisinger 2006, `p`-te-Potenz-Zufallsrauschen), Shaw/Relatedness (Shaw 1998, Distanz + Bedarfsdifferenz), SISR-String (Christiaens & Vanden Berghe 2020, vereinfacht - siehe Grenzen).
- **Repair-Operatoren** (`alns_repair.py`): Greedy (zufällige Reihenfolge, jeweils günstigste Position), Regret-2 (größter Regret zuerst) - beide gegen Brute-Force-Neuberechnung der Einfügekosten kreuzgeprüft.
- **ALNS-Hauptschleife** (`alns_algorithm.py`): Metropolis-Akzeptanz (portiert, nicht neu hergeleitet), adaptive Gewichte mit roulette-wheel-Auswahl und segmentweisem Update (Ropke & Pisinger 2006).
- **Auswertung** (`alns_evaluation.py`): Kennzahlen, Sweeps, Methoden-/Adaptivitäts-/Operator-Vergleiche.

## Was nicht funktioniert hat / Grenzen

- **Adaptive Gewichte bringen praktisch nichts** (8.89 % gegen 8.87 % nicht-adaptiv) - die zweite Vorab-Hypothese dieses Stücks trägt NICHT, obwohl der Destroy+Repair-Mechanismus selbst klar gewinnt. Möglicher Grund: alle vier Destroy- und beide Repair-Operatoren sind hier für sich brauchbar (keiner katastrophal schlecht) - wenn nichts wirklich aussortiert werden muss, bringt das Lernen kaum etwas. Ein Echo der `order-batching-demo`-Vorgeschichte (ALNS dort insgesamt verworfen).
- **Worst-Removal ist überraschend der schwächste Einzel-Destroy-Operator** (8.52 %, schwächer sogar als Zufällig mit 8.60 %) - die "teuerste Kunden zuerst"-Heuristik entfernt vermutlich wiederholt ähnliche, schon auffällige Kunden statt eine vielfältige Störung zu erzeugen; das `p`-te-Potenz-Zufallsrauschen mildert das nur teilweise.
- **Greedy schlägt Regret-2** (8.95 % gegen 8.51 %) - gegen die verbreitete Literatur-Intuition, dass Regret-Sortierung meist hilft. Nicht angenommen, sondern gemessen - ein weiterer Beleg für die Linien-Disziplin "immer messen, nie annehmen".
- **Budget-Konvention grobkörniger als die Schwester-Demo**: ein ALNS-Durchlauf zählt als `k` bewertete Kandidaten, nicht als die tatsächliche Zahl der Einfüge-Kostenvergleiche - eine bewusste Vereinfachung (siehe `alns_algorithm.py`-Docstring), kein exakter Rechenzeit-Vergleich. Der ALNS-Vorteil ist trotzdem robust: die kleine Nachbarschaft bleibt über jeden getesteten Kapazitäts-Punkt weit zurück (siehe `tests/test_claims.py::test_small_neighborhood_stays_far_behind_alns_across_capacities`), nicht nur im Standardfall.
- **SISR ist vereinfacht**: keine Bevorzugung räumlich benachbarter Routen bei der String-Auswahl, keine adaptive Stringlängen-Verteilung wie im Original-Papier - der für die Operator-Ablation entscheidende Unterschied (zusammenhängende Strings statt Einzelkunden) bleibt erhalten.
- **Nur Metropolis-Akzeptanz**: wie bei [parallel-tempering-demo](https://sebastianhanisch-parallel-tempering-demo.streamlit.app/) literaturtreu (Ropke & Pisinger 2006 selbst nutzen dieselbe Regel), nicht willkürlich.
- **Kleinere Budget-/Stopp-Bereiche als die Schwester-Demo**: eine ALNS-Iteration kostet ungleich mehr Rechenzeit als ein einzelner Nachbarschafts-Kandidat.
- **Keine untere Schranke**: wie die Schwester-Demo relativ zur Savings-Konstruktion, nicht zum Optimum.
- **Synthetische Instanzen:** euklidisch, gleichverteilt oder in fünf Gruppen, Bedarfe unabhängig gleichverteilt (1-9), keine Zeitfenster.

## Verifikation

- **Destroy-Operatoren**: entfernen jeweils exakt `k` Kunden (oder alle, falls `k` > Kundenzahl), verbleibende Routen bleiben eine gültige Teilpartition - über viele Zufallsinstanzen und Seeds geprüft.
- **Repair-Operatoren**: fügen ALLE entfernten Kunden wieder ein, Ergebnis ist immer eine vollständige, kapazitäts-machbare Partition; Regret-2 gegen Brute-Force-Neuberechnung der besten/zweitbesten Einfügeposition kreuzgeprüft.
- **`accept()` (Metropolis)** gegen die geschlossene Formel kreuzgeprüft (Grenzfälle: Verbesserung immer an, eingefroren nur verbessernd, sehr heiß fast immer an).
- **Adaptive Gewichte** bleiben normiert/positiv, Roulette-Wheel-Auswahlwahrscheinlichkeit nachweislich proportional zum Gewicht (viele Ziehungen geprüft), `adaptive=False` wählt nachweislich gleichverteilt.
- **Bestes-je-gefundenes-Ergebnis** über die ganze Suche monoton nicht schlechter werdend, Budget-Buchführung geprüft.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt**, über dieselben Auswertungsfunktionen wie die App selbst (`ev.run_config`/`ev.sweep`/`ev.compare_*`), NIE über ein Ad-hoc-Skript mit abweichender Zufalls-Bindung; AppTest-Rauchtests (Voreinstellung, jedes Preset, jeder Schritt, beide Verfahren, leere Operator-Auswahl, Würfel-Knopf, Permalink-Grenzen, Extremwerte, Experimente auf Abruf, Footer). `tests/test_claims.py` pinnt bewusst nur die Eckwerte jedes Sweeps (nicht jeden einzelnen Punkt) und cacht jede Konfiguration mit `lru_cache` - trotzdem läuft die Testsuite hier mehrere Minuten, deutlich länger als bei jedem anderen Stück dieser Linie (eine ALNS-Iteration kostet ungleich mehr Rechenzeit, siehe Grenzen).
- Übernommener Kern: die vier CVRP-Bausteine aus `vrp-nachbarschaften-demo` (dort bereits gegen Brute-Force/den TSP-Sonderfall kreuzgeprüft); ein Cross-Repo-Regressionstest bestätigt hier zusätzlich, dass die Kopie auf einer festen Instanz bytegleich zum Original rechnet (läuft nur lokal, überspringt in CI/frischen Klones automatisch).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Verfahren-Umschalter, Schritte, Ergebnis, 📐 Sweeps, 🔬 Experimente, 🚧 Grenzen, Mathe |
| `alns_destroy.py` | Vier Destroy-Operatoren (Zufällig, Worst, Shaw, SISR) |
| `alns_repair.py` | Zwei Repair-Operatoren (Greedy, Regret-2) |
| `alns_algorithm.py` | Metropolis-Akzeptanz, adaptive Gewichte, ALNS-Hauptschleife |
| `alns_tour.py`, `alns_construction.py`, `alns_interroute.py` | CVRP-Infrastruktur (Kopie aus `vrp-nachbarschaften-demo`) |
| `alns_scenario.py`, `alns_constants.py` | Instanzen; Konstanten, Presets |
| `alns_evaluation.py` | Kennzahlen, Sweeps, Methoden-/Operator-Vergleiche |
| `alns_presets.py`, `alns_visualization.py` | Permalink/Presets, Plotly-Figuren |
| `tests/` | Destroy/Repair/Algorithmus-Korrektheitskette, CVRP-Kern-Smoke-Tests, Szenario/Auswertung, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
