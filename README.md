# Koordination ohne Absprache: der offensichtliche Platz – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-fokalpunkt-demo.streamlit.app/)** -->

Sechstes Stück der Reihe **Koordinationsspiele** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Vehikel: **zwei Fahrer ohne Funkkontakt, die eine Wechselbrücke an demselben von m Übergabeplätzen übergeben sollen.** Treffen sie denselben Platz, sparen sie Umschlagzeit (1, auf Platz 1 nahe am Tor 1 + δ), sonst nichts. Viele Paare lernen je für sich mit **Softmax-REINFORCE**, bis sie sich treffen; die Demo fragt: **was unterscheidet das Paar, das zusammen gelernt hat, von einem Fremden (Eigenspiel gegen Fremdspiel), was lenkt die Konvention (ein „offensichtlicher“ Platz mit Bonus, eine gemeinsame Vorgeschichte, die Symmetrien des Spiels bei Other-Play) und wie weit trägt das Ergebnis kleiner Stichproben?**

Abgrenzung: `koordinationsspiel-demo` (erstes Stück) zeigt die Gleichgewichte und die Schwelle p* der Hirschjagd mit zwei Handlungen, `gleichgewichtsauswahl-demo` (zweites), welche Konvention auf Dauer gewählt wird, `konventionswechsel-demo` (drittes), wie lange ein Wechsel dauert, `globale-spiele-demo` (viertes), was Unsicherheit über den Zustand ändert, `schwaechstes-glied-demo` (fünftes), warum eine Kette langsam wird. Hier gibt es **m Handlungen, Lernen und die Messgröße Fremdspiel.** `korreliert-demo` lässt einen Vermittler Handlungen empfehlen, hier gibt es **kein** Signal. `policy-gradient-demo` erklärt REINFORCE an einem Lagerroboter, hier ist es nur das Vehikel und wird nicht neu erklärt; `marl-demo` lässt mehrere Agenten unabhängig an einer Kran-Vergabe lernen. `schelling-demo` meint die **Segregation**, nicht Schellings Fokalpunkte: „Fokalpunkt“ heißt hier nur ein ausgezeichneter Platz. **Lernregel und Eigennutz sind Modellannahmen**, keine Aussage über Menschen, Sprachmodelle oder echte Übergabeplätze.

## Kernfrage

*Zwei Paare, die jeweils fast sicher koordinieren, treffen sich über Kreuz nur so oft wie der Zufall: bei vier Plätzen mit 0,250 (Eigenspiel 0,997). Ein „offensichtlicher“ Platz mit 10 % mehr Zeitersparnis (δ = 0,1) hebt das Fremdspiel nur auf 0,262, erst 100 % (δ = 1) bringen 0,750.* Eigenes dieser Demo sind (1) die **exakte Messgröße**: das Fremdspiel wird nicht gezogen, sondern als Σ_a p_i(a)·q_j(a) aus den Politiken gerechnet, und bei festen Konventionen folgt es aus ihrer Verteilung (Σ nₐ(nₐ−1)/(P(P−1))); (2) der **Halbweg-Bonus** statt einer Schwelle, und seine Abhängigkeit vom Lernverfahren; (3) der **Erwartungsfluss gegen den Lauf**; (4) die **Dosis der gemeinsamen Vorgeschichte**; (5) **Other-Play** mit der richtigen und mit einer falschen Symmetriegruppe; (6) die **Streuung kleiner Stichproben**, die den Hook des Plans nicht trägt.

## Modell

Zwei Fahrer eines Paares wählen gleichzeitig einen von m Plätzen. **Belohnung** r = 1 bei gleichem Platz, auf Platz 1 r = 1 + δ, sonst 0. Fahrer k hat Parameter θ_k ∈ ℝ^m und die Politik p_k = softmax(θ_k). **Lernregel** (REINFORCE ohne Basis, Williams 1992): nach jeder Runde θ_k ← θ_k + η·r·(e_a − p_k) mit dem gewählten Platz a. Voreinstellung: η = 0,1, Anfangsparameter normalverteilt mit Streuung σ₀ = 0,1, 6 000 Schritte, 300 Paare.

- **Eigenspiel:** Mittel über die Paare von Σ_a p^A_i(a)·p^B_i(a) (die beiden Fahrer desselben Paares). **Fremdspiel:** Mittel von Σ_a p^A_i(a)·p^B_j(a) über alle Paare i ≠ j (Fahrer A eines Paares gegen Fahrer B eines anderen). Beides exakt aus den Politiken. Bei nur einem Paar ist das Fremdspiel nicht definiert (die App zeigt „–“).
- **Konvention** eines Paares: argmax von p^A + p^B (bei Gleichstand der kleinste Platz). Bei reinen Politiken ist das Fremdspiel Σ nₐ(nₐ−1)/(P(P−1)) mit der Zahl nₐ der Paare auf Platz a; für große P etwa Σ π_a².
- **Zufallsgrenze:** sind die Konventionen gleichverteilt, ist Fremdspiel − 1/m = (X − (m−1))/(m(P−1)) mit X ~ χ²(m−1), also die Standardabweichung √(2(m−1))/(m(P−1)) (0,0020 bei m = 4 und 300 Paaren). Die Verdikte nennen „nur zufällig“ bis vier solche Standardabweichungen über 1/m.
- **Halbweg-Bonus:** der Bonus, bei dem der Anteil der Paare mit Konvention auf Platz 1 die Mitte zwischen 1/m und 1 erreicht, log-interpoliert zwischen den Gitterpunkten δ > 0.
- **Erwartungsfluss:** statt des gezogenen Updates dessen Erwartungswert, E[Δθ_A]/η = p_A ⊙ (p_B ⊙ w − J) mit J = Σ p_A p_B w und w = (1 + δ, 1, …, 1); deterministisch, nur vom Start abhängig.
- **Other-Play** (im Lehrbuchfall): die Marke von Fahrer B wird je Runde durch eine zufällige Umbenennung φ aus einer Gruppe übersetzt, Treffer bei a_A = φ(a_B); getestet wird ohne Umbenennung. Gruppe **alle Plätze** (m! Umbenennungen): J = (1/m)(1 + δ·p₁), unabhängig von B. Gruppe **Platz 1 fest** ((m−1)! Umbenennungen): J = (1 + δ)·p₁q₁ + (1 − p₁)(1 − q₁)/(m − 1); bei m = 2 trivial.
- **Gemeinsame Vorgeschichte:** eine Referenz-Paarung lernt T₀ Schritte; alle Paare ihrer Gruppe starten von deren Parametern und lernen mit dem eigenen Partner weiter. **Gemeinschaften:** G Fahrer lernen zusammen, in jeder Runde bilden sie zufällig Paare.

Alle Herleitungen sind eigene Rechnung, durch Handrechnung, zweite Rechenwege und Simulation gestützt, nicht gegen Literatur geprüft.

## Befunde und Korrekturen gegenüber dem Plan

Alle Zahlen sind in dieser Demo neu gemessen (`tests/test_claims.py`). Die Messreihen (Zellen aus 4 Blöcken zu je 500 Paaren, Vorerfahrung aus 40 Referenzen mit je 25 Paaren je Block) sind vorgerechnet (`data/measurements.json` mit Erzeuger-Hash, `tools/make_measurements.py`), die Live-Experimente rechnen mit den Reglern der Seitenleiste. Simulationen haben feste Zufalls-Seeds (SplitMix64) und werden nur mit Marge verglichen; ± ist der Standardfehler der vier Blöcke. Alle Aussagen gelten für **dieses Lernmodell (Softmax-REINFORCE mit Voreinstellung η = 0,1, σ₀ = 0,1, 6 000 Schritte)**, wo nichts anderes steht.

| Nr. | Hypothese | Antwort | Beleg |
|---|---|---|---|
| H1 | Eigenspiel ≈ 1 und Fremdspiel ≈ 1/m (m = 2, 4, 8, 16) | **Ja** | Eigenspiel ≥ 0,99 in 36 von 36 Zellen (4 Platzzahlen × 9 Boni; kleinster Wert 0,995). Ohne Bonus Fremdspiel für m = 2 / 4 / 8 / 16: 0,501 / 0,250 / 0,125 / 0,063 (1/m: 0,5 / 0,25 / 0,125 / 0,0625) (`test_h1_*`) |
| H2 | Fremdspiel = Σ nₐ(nₐ−1)/(P(P−1)) der Konventionsverteilung (reine Politiken) | **Ja** | größte Abweichung zum gemessenen Fremdspiel in den 36 Zellen 0,0008; exakt für reine Politiken (Handinstanz 5 Paare: 0,4; Orakel gegen Aufzählung aller Paarungen) (`test_h2_*`) |
| H3 | (Plan, V5) δ = 0,2 hebt bei m = 2 das Fremdspiel auf 0,81 | **Nein (Stichproben-Ausreißer)** | 100 Stichproben aus je 20 Paaren (6 000 Schritte): Mittel 0,576, Streuung 0,095, 5.–95. Perzentil 0,479 bis 0,731, größter Wert 0,899, **1 von 100 bei 0,81 oder darüber**; 100 Stichproben aus je 40 Paaren (20 000 Schritte): Mittel 0,577, größter 0,672, 0 von 100. Das Mittel liegt zwischen den Gitterwerten bei δ = 0,1 (0,529) und 0,3 (0,650) (`test_h3_*`) |
| H4 | (Plan, V5) δ = 0,01 und 0,05 ändern kaum etwas | **Ja** | Fremdspiel in 20 Zellen (4 Platzzahlen × δ ≤ 0,03) höchstens 0,0021 über 1/m. δ = 0,05, m = 2 / 4: 0,512 / 0,255 (20 Paare) und 0,502 / 0,253 (40 Paare) (`test_h4_*`) |
| H5 | Der Bonus wirkt als glatte Kurve ohne scharfe Schwelle, der Halbweg-Bonus wächst mit m | **Ja, im Rahmen des Gitters** | Halbweg-Bonus für m = 2 / 4 / 8 / 16: 0,25 / 0,41 / 0,55 / 0,73. Der Anteil auf Platz 1 steigt monoton (Rückgänge höchstens 0,01 im Rauschen); größter Anstieg zwischen zwei Gitterpunkten (Faktor etwa 3 im Bonus) 0,17 / 0,32 / 0,37 / 0,41. Eine Schwelle, die feiner als das Gitter ist, ist nicht ausgeschlossen. Fremdspiel mehr als 0,01 über 1/m erst ab δ = 0,1 (m = 2, 4) bzw. 0,3 (m = 8, 16) (`test_h5_*`) |
| H6 | Die Lage der Kurve hängt am Lernverfahren (η, σ₀), nicht an der Schrittzahl | **Ja** | m = 4, Halbweg-Bonus: η = 0,03 / 0,1 / 0,3: 0,21 / 0,41 / 1,14; σ₀ = 0 / 0,1 / 0,5: 0,37 / 0,41 / 0,98; Schritte 2 000 / 6 000 / 20 000: 0,43 / 0,41 / 0,41 (Faktor unter 1,07). Eigenspiel in allen 54 Zellen über 0,98 (kleinster 0,989) (`test_h6_*`) |
| H7 | Der Erwartungsfluss überschätzt die Wirkung des Bonus gegenüber dem Lauf | **Ja, mit drei Ausnahmen** | In 12 von 15 Zellen (m = 2, 4, 8 × δ = 0,01 bis 1) sieht der Fluss mehr als vier Standardfehler mehr Paare auf Platz 1 als der Lauf; die Ausnahmen sind die drei Zellen mit δ = 0,01. Ohne Bonus gewinnt im Fluss der Start-Favorit bei 1,000 / 0,976 / 0,970 der Paare (m = 2 / 4 / 8), der Lauf folgt ihm nur zu 0,604 / 0,385 / 0,256. Größter Abstand: m = 8, δ = 0,3: Fluss 0,936, Lauf 0,360 (`test_h7_*`) |
| H8 | Gemeinsame Vorgeschichte wirkt dosisabhängig, auch unfertig; zwischen Gruppen bleibt 1/m | **Ja** | m = 4, T₀ = 0 / 25 / 50 / 100 / 200 / 400 / 1 000: Fremdspiel innerhalb der Gruppe 0,286 / 0,431 / 0,494 / 0,712 / 0,876 / 0,994 / 0,998 (monoton); zwischen Gruppen 0,240 bis 0,257 (m = 2: 0,493 bis 0,508; m = 8: 0,124 bis 0,131). Innenspiel erstmals ≥ 0,9 bei T₀ = 100 / 400 / 1 000 für m = 2 / 4 / 8. Die Referenz muss nicht fertig sein: bei T₀ = 400 hat sie selbst Eigenspiel 0,829, die Gruppe trifft sich zu 0,994. T₀ = 0 (gleicher Zufallsstart) hebt innen schon leicht: m = 2 / 4 / 8: 0,536 / 0,286 / 0,151 gegen 0,501 / 0,249 / 0,125 zwischen den Gruppen. Gemeinschaftszeile (m = 4, G = 2 / 4 / 8 / 16 / 32): innerhalb nach 500 Runden 0,913 / 0,867 / 0,800 / 0,705 / 0,595, nach 2 000 Runden mindestens 0,989; zwischen Gemeinschaften 0,25 ± 0,003 (`test_h8_*`) |
| H9 | Other-Play mit der richtigen Gruppe (Platz 1 fest) senkt den nötigen Bonus für m ≥ 4 und kostet Eigenspiel | **Ja** | Halbweg-Bonus Selbstspiel gegen Platz 1 fest, m = 4 / 8 / 16: 0,40 / 0,53 / 0,75 gegen 0,15 / 0,12 / 0,14. Fremdspiel bei δ = 0,3: 0,368 / 0,202 / 0,100 gegen 0,578 / 0,550 / 0,475; höher in 21 von 21 Zellen. Preis: Eigenspiel bei δ = 0: 0,639 / 0,475 / 0,346, bei δ = 0,3 (m = 4, 8): 0,823 / 0,767, bei δ = 1: 0,973 / 0,964 / 0,940 (Selbstspiel in allen 28 Zellen von E5 über 0,99). Die Gruppe setzt selbst einen Fokalpunkt: schon ohne Bonus Fremdspiel 0,302 / 0,209 / 0,138 gegen 0,251 / 0,126 / 0,062 (`test_h9_*`) |
| H10 | Other-Play mit falscher Gruppe bringt keine Koordination über den Zufall hinaus, und bei m = 2 macht „Platz 1 fest“ keinen Unterschied zum Selbstspiel | **Ja** | Gruppe „alle Plätze“: Eigen- und Fremdspiel höchstens 0,016 bzw. 0,013 neben 1/m in 28 von 28 Zellen, obwohl bei δ = 0,3 und m = 4 / 8 die Konvention bei 0,955 / 0,864 der Paare auf Platz 1 liegt (Eigenspiel 0,251 / 0,127). Platz 1 fest bei m = 2: Fremdspiel gegen Selbstspiel höchstens 0,014 Unterschied in 7 von 7 Zellen, derselbe Mechanismus (Gleichheit bei gleichem Seed im Test) (`test_h10_*`) |
| H11 | Lernrate und Rauschen: der Halbweg-Bonus skaliert wie eine einfache Potenz | **Offen** | nur drei Werte je Regler (η: Faktor 10 in der Lernrate, 5,5 im Halbweg-Bonus); keine Gesetzmäßigkeit behauptet |
| H12 | Auf Netzen mit lokaler Interaktion ändert sich die Konvention | **Offen** | nicht Teil dieses Stücks (Netze: `konventionswechsel-demo`) |

### Korrekturen gegenüber dem Plan

- **Hook:** der Plan-Hook „δ = 0,2 hebt bei m = 2 auf 0,81“ (V5) **entfällt.** 0,81 ist ein Wert aus 20 Paaren und einem Zufallsstrom; mit 100 solchen Stichproben liegt das Mittel bei 0,576, nur 1 von 100 erreicht 0,81 (H3). Der Einzelplan fand in seiner Wegwerfmessung als größten von 100 Werten 0,810; hier ist der größte 0,899. Beides sagt: **eine kleine Stichprobe kann einen Bonus-Effekt vortäuschen**. Neuer Hook: siehe Kernfrage. Die erhaltene V5-Rohdatei des Plans (anderer Aufbau: 40 Paare, 20 000 Schritte) passt nicht zum V5-Skript; diese Demo benutzt keines von beiden.
- **Zahlen mit anderem Zufallsstrom:** die Wegwerfmessung des Plans nannte bei m = 4, δ = 0,1 das Fremdspiel 0,280 (Anteil auf Platz 1 0,400); hier 0,262 (0,345). Halbweg-Bonus: Plan 0,25 / 0,39 / 0,55 / 0,74, hier 0,25 / 0,41 / 0,55 / 0,73. Weitere Abweichungen im Bereich der Zellenstreuung (Fremdspiel bei δ = 0,3, m = 16, Platz 1 fest: Plan 0,434, hier 0,475). Der Anteil auf Platz 1 schwankt von Zelle zu Zelle mit den Konventionen der Paare um bis zu etwa 0,05; Aussagen über Rangfolgen stützen sich deshalb auf Zählungen über das Raster.
- **Der Schwelle-Begriff:** der Plan sprach von „Schwelle“ und „S-Kurve“; gemessen wird ein monotoner Anstieg im Gitter mit Faktor 3 im Bonus (H5), der Halbweg-Bonus ist eine Eigenschaft des Lernverfahrens (H6).
- **Neu:** Fremdspiel als exakte Messgröße samt Zufallsgrenze, Erwartungsfluss gegen Lauf, Dosiskurve der Vorgeschichte, Other-Play mit Symmetriegruppe und Gegenbeispiel. Das Gemeinschafts-Experiment des Plans ist nur eine Zeile in E4. Die Messung C3 des Plans (Selbstspiel mit einem Parametersatz) ist nicht übernommen.
- **Port und Name:** Port 9208 (belegt: 9203 bis 9207, 9210, 9211), Präfix `fok`, Linie `koordination`.

## Methodik

- **Läufe:** alle Paare eines Aufrufs laufen gleichzeitig (numpy über Paare, Spieler und Plätze); die Gleichverteilten kommen blockweise aus SplitMix64 (`fok_rng.py`, wörtliche Kopie aus `ab-test-demo`, Kopie-Test). Die Reihenfolge der Zufallszahlen steht im Kopf von `fok_learning.py`; die Blockgröße ändert die Zahlen nicht (Test). Ziehung: die kleinste Platz-Nummer mit u < kumulierte Wahrscheinlichkeit; ein Platz mit Wahrscheinlichkeit 0 wird nicht gezogen, auch nicht bei u = 0 (gefundener Randfall, Test).
- **Zeitbudget:** Paare · Schritte höchstens 3 Millionen (500 Paare bei 6 000 Schritten, 150 bei 20 000); darüber rechnet die App mit weniger Paaren und sagt es.
- **Zellen:** vier unabhängige Blöcke zu je 500 Paaren, Zufalls-Seeds aus der Zellenbezeichnung (FNV-1a); die Reihen sind bei gleichem Code Zahl für Zahl dieselben, auch bei anderer Zahl von Prozessen (bei der Erzeugung einmal von Hand geprüft: zwei Läufe mit 8 und 12 Prozessen ergaben dieselbe Datei; im Test: die Zusammenstellung hängt nicht von der Reihenfolge der Ergebnisse ab).
- **Hauptansicht mit Vorgeschichte:** bei T₀ > 0 bilden die Paare höchstens 8 Gruppen mit je mindestens 2 Paaren; die Referenz lernt im selben Modus wie die Paare.
- **„Nur zufällig“ und „höher“** heißen: das Fremdspiel liegt höchstens bzw. mehr als vier Standardabweichungen der Zufallsgrenze über 1/m; „fällt/überschätzt“ im Vergleich von Fluss und Lauf: mehr als vier Standardfehler (der beiden Mittel) Abstand.

## Ehrliche Grenzen

- **Modell, keine Messung:** Softmax-REINFORCE ohne Basis, ein Parametersatz je Fahrer, Zeitersparnis als Auszahlung. Gemessen ist diese Regel mit diesen Voreinstellungen; über Menschen, Sprachmodelle oder echte Übergabeplätze sagt die Demo nichts. Die Lage der Bonus-Kurve hängt an Lernrate und Anfangsstreuung (H6); andere Lernregeln können andere Zahlen geben.
- **Der „offensichtliche“ Platz ist eine Zahl** (Bonus auf Platz 1), ohne Kultur, Sprache oder Gewohnheit.
- **Reden vor dem Spiel ist nicht modelliert;** die gemeinsame Vorgeschichte ist ein gemeinsamer Anfangszustand der Lernenden, kein Signal.
- **Other-Play nur im Lehrbuchfall** (Training gegen umbenannte Partner im Zuordnungsspiel): kein Nachbau von Hanabi oder Overcooked, keine Aussage über deren Ergebnisse. Other-Play findet nur, was in den Symmetrien des Spiels steckt: die Gruppe muss zum Spiel passen (H10), und die Gruppe „Platz 1 fest“ ist selbst ein Fokalpunkt (H9).
- **Das Gitter ist grob** (Faktor etwa 3 im Bonus); eine feinere Schwelle ist nicht ausgeschlossen (H5). Zellen sind Mittel über vier Blöcke; Live-Läufe sind ein Block mit höchstens 500 Paaren und streuen stark, besonders das Zwischenspiel bei wenigen Referenzen.
- **Kleine Stichproben** (H3): 20 Paare ergeben bei m = 2 und δ = 0,2 eine Streuung des Fremdspiels von 0,095; 40 Paare mit mehr Schritten 0,047.
- **Plan-Werte** der Wegwerfmessung sind keine Messungen dieser Demo (Korrekturen oben).

## Quellen

- **Hu, Lerer, Peysakhovich und Foerster, „Other-Play“ for Zero-Shot Coordination (ICML 2020, arXiv 2003.02979):** laut Abstract (Suchtreffer, nicht im Volltext gelesen) kann Selbstspiel zu stark spezialisierten Konventionen führen, die mit unbekannten Partnern scheitern; Other-Play nutzt bekannte Symmetrien des Problems; geprüft wurde es am Kartenspiel Hanabi. Die Demo baut den Mechanismus im Lehrbuchfall nach und beansprucht keine Übereinstimmung mit der Arbeit; keine Aussage zu Hanabi.
- **Gessler u. a., OvercookedV2 (ICLR 2025, arXiv 2503.17821):** laut Abstract (Suchtreffer, nicht im Volltext gelesen) lässt sich das Zero-Shot-Versagen in Overcooked größtenteils auf schlechte Zustandsabdeckung im Selbstspiel zurückführen, nicht auf anspruchsvollere Koordination. Einordnung: in der Richtung vereinbar mit H8 (geteilte Erfahrung bestimmt die Koordination mit Fremden); kein Nachbau, keine Aussage zu Overcooked.
- **Aharon, La Malfa, Wooldridge und Kraus, Tacit Coordination of Large Language Models (arXiv 2601.22184):** laut Abstract (Suchtreffer, nicht im Volltext gelesen) koordinieren über 20 Modelle ohne Kommunikation oft so gut wie oder besser als Menschen, scheitern aber an Aufgaben mit zahlenbezogenem Alltagswissen oder kulturell geprägter Auffälligkeit. Einordnung: Gegenstück zur Modellannahme dieser Demo, in der der „offensichtliche“ Platz nur ein Bonus ist; keine Aussage über Sprachmodelle.
- **Schelling (1960) und Mehta, Starmer und Sugden (1994):** Abstract nicht geprüft und nicht verwendet; der Begriff Fokalpunkt dient hier nur als Name für einen ausgezeichneten Platz.

Es steht **kein Laborergebnis** als eigene Zahl in dieser README: alle eigenen Zahlen sind Messungen des Lernmodells, die Literaturangaben stehen nur als „laut Abstract“.

## Verifikation

- **Handrechnung** (`tests/conftest.py`): Erfolgswahrscheinlichkeiten (0,5; 0,5; 1/3; mit Bonus 0,55), zwei reine Paare (Eigenspiel 1, Fremdspiel 0), fünf reine Paare (Fremdspiel 8/20 = 0,4), ein REINFORCE-Schritt (0,06 / −0,06), Softmax, Umbenennungen, Halbweg-Bonus (0,316).
- **Orakel** (`tests/test_oracle_fokalpunkt.py`, scipy nur dort): Erfolgswahrscheinlichkeit gegen Ziehungen (Marge vier Standardfehler), Matrixform gegen Doppelschleife, Softmax gegen scipy; Erwartungs-Update durch Aufzählung aller (a, b), gegen zentrale Differenzen und gegen Simulationsmittel; das ganze Training (Selbstspiel und beide Other-Play-Gruppen) und der Erwartungsfluss und das Gemeinschaftslernen gegen Schleifen in reinem Python auf denselben Gleichverteilten; Other-Play-Erwartungen gegen die Aufzählung der ganzen Gruppe und gegen Simulation; Fremdspiel aus der Konventionsverteilung gegen die Aufzählung aller Paarungen.
- **Eigenschaftstests** (`tests/test_eigenschaften_fokalpunkt.py`): Softmax-Verschiebung, Symmetrie der Konventionen ohne Bonus (Verteilung), Reihenfolge der Paare, Rollentausch, mehr Plätze und größerer Bonus in der Verteilung, gleicher Seed in einem zweiten Prozess mit anderer Hash-Randomisierung.
- **Claims** (`tests/test_claims.py`): jede Zahl dieser README, aus den gespeicherten Messreihen oder exakt gerechnet; die Messreihen tragen den Erzeuger-Hash, eine Stichprobe der Zellen wird mit anderen Seeds reproduziert (`tests/test_measurements.py`).
- **Zweig-, Rand- und Verdikt-Tests:** jeder Arm weicht auf mindestens einer Zelle vom Nachbararm ab, Bonus ändert den Anteil auf Platz 1, Vorgeschichte hebt das Innenspiel, m = 2 und 16, ein Paar, zwei Paare, ein Schritt, Überlauf des Softmax; Verdikte als reine Funktionen an handgesetzten Werten (`tests/test_verdicts.py`, `tests/test_edge_cases.py`).
- **Oberfläche:** AppTest (jeder Knopf einzeln, jeder Regler am Minimum und Maximum, Eckfall-Matrix je Modus, jedes Preset, Permalink, PDF), Layout, Footer, Laufzeit-Importe.
- **Plattform:** Softmax-REINFORCE über tausende Schritte ist chaotisch (Winzigkeiten in exp und Summation können einzelne Läufe kippen). Tests prüfen deshalb nur Verteilungsaussagen über viele Paare mit Marge (vier Standardfehler oder große feste Bänder), keine Aussage über ein einzelnes Paar der Simulation und keine Zeiten; die Verdikte der Presets nahe am Zufall sind nur als Art „info oder success“ festgelegt, die Zahlen mit großem Band. Die Messreihen werden aus der Datei gelesen (plattformunabhängig) und mit anderen Seeds nur innerhalb einer Marge reproduziert. Die Linux-CI ist hier nicht lokal nachgestellt (kein Docker).

## Tests

301 Tests (unter Windows, Python 3.14 im Test-venv). Zusammensetzung: Politiken, Symmetriegruppen, Lernen, Vorerfahrung, Konventionen, Orakel, Eigenschaften, Claims, Messreihen, Experimente, Verdikte, Randfälle, Presets, Layout, AppTest-Rauchtests, Visualisierung, Footer, Laufzeit-Importe, SplitMix64-Kopie. Laufzeit der vollen Suite rund zwei Minuten seriell (112 Sekunden).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Hauptansicht, fünf Experimente, Grenzen, Formeln, PDF |
| `fok_constants.py` | Regler-Grenzen, Raster der Experimente, Farben, Presets mit Hilfetexten |
| `fok_policy.py` | Softmax, Ziehung, Erfolgswahrscheinlichkeit, Eigen- und Fremdspiel |
| `fok_symmetry.py` | Symmetriegruppen der Umbenennung (alle Plätze, Platz 1 fest), Other-Play-Erwartung |
| `fok_learning.py` | Zufallsstrom, REINFORCE-Schritt, Training vieler Paare, Erwartungs-Update, Erwartungsfluss |
| `fok_experience.py` | gemeinsame Vorgeschichte (Referenz, Gruppen) und Gemeinschaften |
| `fok_conventions.py` | Konvention eines Paares, Verteilung, Zufallsgrenze, Halbweg-Bonus, Kennzahlen eines Blocks |
| `fok_scenario.py` | Einstellungen (nur mit Schlüsselwörtern), Budget, Zeitschätzung |
| `fok_experiments.py` | Zellen und Reihen der Experimente, vorgerechnete Messreihen mit Erzeuger-Hash |
| `fok_evaluation.py` | Hauptansicht, Formate, Verdikte als reine Funktionen |
| `fok_visualization.py` | Plotly-Abbildungen (Achsen fest) |
| `fok_presets.py` | Permalink, Presets, Zufalls-Seed-Button |
| `fok_layout.py` | fester Kopf |
| `fok_pdf_export.py` | PDF-Export |
| `fok_rng.py` | SplitMix64 (wörtliche Kopie aus `ab-test-demo/abt_rng.py`) |
| `data/measurements.json` | vorgerechnete Messreihen |
| `tools/make_measurements.py` | erzeugt `data/measurements.json` neu (rund drei bis vier Minuten mit 12 Prozessen) |
| `tests/` | alle Tests (siehe oben) |

## Bewusst nicht umgesetzt

- **Reden vor dem Spiel und Signale:** nicht modelliert.
- **Netze mit lokaler Interaktion** (H12): die Konvention auf Netzen behandelt `konventionswechsel-demo`.
- **Hanabi-, Overcooked- und Sprachmodell-Agenten, Menschenexperimente:** die Demo bildet nur das Lehrbuchmodell nach.
- **Andere Lernregeln** als Softmax-REINFORCE ohne Basis (Basislinien, Q-Lernen, beste Antwort).
- **Selbstspiel mit einem gemeinsamen Parametersatz** (Messung C3 des Plans) und eine Potenzgesetz-Anpassung des Halbweg-Bonus (H11).
- **Ein feineres Bonus-Gitter** um den Halbweg-Bonus (Faktor 3 zwischen den Gitterpunkten).

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in der App-Einstellung wählen (getestet mit 3.12 in der CI).

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
