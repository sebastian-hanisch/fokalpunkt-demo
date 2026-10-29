"""Koordination ohne Absprache: der offensichtliche Platz - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Sechstes Stück der Reihe Koordinationsspiele. Zwei Fahrer ohne Funkkontakt sollen eine Wechselbrücke an demselben von m Übergabeplätzen übergeben. Die Demo lässt viele Paare mit
Softmax-REINFORCE zusammen lernen und fragt, was ein Paar, das zusammen gelernt hat, von einem Fremden unterscheidet (Eigenspiel gegen Fremdspiel), was die Konvention lenkt (Bonus für einen
„offensichtlichen“ Platz, gemeinsame Vorgeschichte, die Symmetrien des Spiels) und wie weit man dem Ergebnis kleiner Stichproben trauen darf.

Lauffähig mit: streamlit run app.py
"""

import math

import streamlit as st

import fok_constants as C
import fok_conventions as CV
import fok_evaluation as V
import fok_experiments as XP
import fok_scenario as S
import fok_visualization as Z
from fok_layout import dunkles_design, kopf_css
from fok_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Koordination ohne Absprache: der offensichtliche Platz – Sebastian Hanisch", layout="wide")


# --- Zwischenspeicher (Schlüssel = alle Eingaben inklusive Zufalls-Seed) --------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _meas():
    return XP.load_measurements()


@st.cache_data(show_spinner=False)
def _analyse(s):
    return V.analyse(s)


@st.cache_data(show_spinner=False)
def _e1_live(s):
    return XP.e1_live(s)


@st.cache_data(show_spinner=False)
def _e2_live(s):
    return XP.delta_row(S.with_(s, t0=0), 1, f"e2live|{s.seed}|{s.m}")


@st.cache_data(show_spinner=False)
def _e3_live(s):
    return XP.flow_cell(s, 1, f"e3live|{s.seed}|{s.m}|{s.delta}")


@st.cache_data(show_spinner=False)
def _e4_live(s):
    return XP.e4_live(s)


@st.cache_data(show_spinner=False)
def _e5_live(s):
    return {name: {s.delta: XP.cell(S.with_(s, mode=mode, t0=0), 1, f"e5live|{s.seed}|{name}|{s.m}|{s.delta}")} for name, mode in XP.ARMS.items()}


def show(verdict):
    kind, text = verdict
    getattr(st, kind)(text)


def table(rows):
    st.dataframe(rows, hide_index=True, width="stretch")


def seconds_text(runs: int, s: S.Settings, pairs: int | None = None) -> int:
    """Grobe Sekundenzahl für `runs` Trainingsläufe (nur für die Angabe am Knopf)."""
    p = s.pairs if pairs is None else pairs
    return max(1, math.ceil(runs * min(p, C.RUN_BUDGET // s.steps) * s.steps * (s.m / 4) ** 0.5 / C.PAIR_STEPS_PER_SECOND))


def ms_mean(c: dict, key: str, nd: int = 3) -> str:
    return V.de(c[key]["mean"], nd)


def ms_pm(c: dict, key: str, nd: int = 3) -> str:
    """Mittel ± Standardfehler der Blöcke."""
    return f"{V.de(c[key]['mean'], nd)} ± {V.de(c[key]['se'], nd)}"


delta_label = V.delta_text


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in fok_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧭 Koordination ohne Absprache: der offensichtliche Platz")
st.markdown(
    """
**Sechstes Stück der Reihe Koordinationsspiele.** Zwei Fahrer ohne Funkkontakt sollen eine Wechselbrücke an **demselben** von **m Übergabeplätzen** übergeben; sie haben nie miteinander gesprochen.
Die Stücke davor zeigen die Gleichgewichte (`koordinationsspiel-demo`), welches davon gewählt wird (`gleichgewichtsauswahl-demo`), wie lange ein Wechsel dauert (`konventionswechsel-demo`),
was Unsicherheit ändert (`globale-spiele-demo`) und warum eine Kette langsam wird (`schwaechstes-glied-demo`). Hier lernen **viele Paare** je für sich, bis sie sich treffen, und die Frage ist:
**Was unterscheidet das Paar, das zusammen gelernt hat, von einem Fremden?** Gemessen wird beides, das **Eigenspiel** (mein Partner) und das **Fremdspiel** (ein Fahrer aus einem anderen Paar),
und dazu, was die Wahl lenkt: ein **„offensichtlicher“ Platz** (Platz 1 spart mehr), eine **gemeinsame Vorgeschichte** und die **Symmetrien des Spiels** (Other-Play).

Abgrenzung: `korreliert-demo` lässt einen Vermittler Handlungen empfehlen, hier gibt es **kein** Signal; `policy-gradient-demo` erklärt REINFORCE an einem Lagerroboter, hier ist es nur das Vehikel und
wird nicht neu erklärt; `marl-demo` lässt mehrere Agenten unabhängig an einer Kran-Vergabe lernen. `schelling-demo` meint die Segregation, nicht Schellings Fokalpunkte: „Fokalpunkt“ heißt hier
nur ein ausgezeichneter Platz. **Lernregel und Eigennutz sind Modellannahmen (Softmax-REINFORCE)**, keine Aussage über Menschen, Sprachmodelle oder echte Übergabeplätze.

Weiter unten: fünf Tabs mit je einem **🔬 Experiment**, die Abschnitte **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Das Spiel und der Fokalpunkt in einem Absatz", expanded=True):
    st.markdown(
        r"""
1. **Zuordnungsspiel.** Beide Fahrer wählen gleichzeitig einen von $m$ Plätzen. Treffen sie denselben Platz, sparen sie Umschlagzeit: **1**, auf Platz 1 (nahe am Tor) **1 + δ**. Sonst 0.

| | B wählt Platz 1 | B wählt Platz $b \neq 1$ |
|---|---|---|
| **A wählt Platz 1** | $(1+\delta,\ 1+\delta)$ | $(0,\ 0)$ |
| **A wählt Platz $a \neq 1$** | $(0,\ 0)$ | $(1,\ 1)$ falls $a = b$, sonst $(0,\ 0)$ |

2. **Jeder Platz ist ein Gleichgewicht.** Im Spiel selbst zeichnet nichts einen Platz vor dem anderen aus (δ = 0); ein ausgezeichneter („offensichtlicher“) Platz, ein **Fokalpunkt**, ist das, was Fahrer
   ohne Absprache zusammenbringen könnte. Hier ist er nur eine Zahl: der Bonus δ auf Platz 1. Kultur, Sprache oder Gewohnheit sind nicht modelliert.
3. **Lernen.** Jeder Fahrer hat Parameter $\theta \in \mathbb{R}^m$ und wählt Platz $a$ mit Wahrscheinlichkeit $\mathrm{softmax}(\theta)_a$. Nach jeder Runde sieht er nur seinen Platz und die Belohnung und verschiebt $\theta$
   zu dem gewählten Platz hin, wenn er getroffen hat (REINFORCE ohne Basis). Wie, zeigt die Erklärtabelle.
4. **Eigenspiel und Fremdspiel.** *Eigenspiel*: die beiden Fahrer desselben Paares, die zusammen gelernt haben. *Fremdspiel*: Fahrer A eines Paares gegen Fahrer B eines anderen. Beide sind
   **exakt aus den Politiken** gerechnet (Summe über die Plätze von $p(a)\cdot q(a)$), nicht aus Ziehungen.
        """
    )

with st.expander("Erklärtabelle: was die Auswahl „Modus“ und die Regler tun"):
    st.markdown(
        """
| Modus / Regler | Was er tut | Wirkung auf die Frage |
|---|---|---|
| **Selbstspiel** | Beide Fahrer eines Paares lernen gegeneinander, wie sie sind. | Jedes Paar legt sich fest, aber auf irgendeinen Platz. |
| **Other-Play: alle Plätze** | Je Runde wird der Platz des Partners durch eine zufällige Umbenennung **aller** Plätze übersetzt; Treffer, wenn A den übersetzten Platz wählt. | Richtig, wenn kein Platz ausgezeichnet ist (δ = 0). **Mit Bonus ist die Gruppe falsch**: Gegenbeispiel. |
| **Other-Play: Platz 1 fest** | Wie oben, aber Platz 1 bleibt fest, die übrigen Plätze werden vertauscht: genau die Symmetrien des Spiels **mit** Bonus. | Bei m = 2 wirkungslos (nichts zu vertauschen). Bei größerem m braucht es einen kleineren Bonus und kostet Eigenspiel. |
| **Bonus δ** | Platz 1 bringt 1 + δ statt 1. | Lenkt die Konvention auf Platz 1; die Wirkung ist eine glatte Kurve, keine Schwelle. |
| **Lernrate η, Anfangsstreuung σ₀** | Wie stark ein Treffer die Parameter verschiebt; wie verschieden die Fahrer starten. | Verschieben, wie groß δ sein muss (E2). |
| **Gemeinsame Vorgeschichte T₀** | Die Paare starten in Gruppen von den Parametern einer Referenz-Paarung, die T₀ Schritte gelernt hat (0: unabhängiger Start). | Macht Fremde in der Gruppe zu Bekannten (E4). |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("🧭 Spiel und Modus")
    m = st.slider("Zahl der Übergabeplätze m", *bounds("m_slider"), key="m_slider", step=1, help="Wie viele Plätze zur Wahl stehen. Wählen die Paare ihre Plätze gleichverteilt, trifft ein Fremder im Mittel mit 1/m.")
    delta = st.slider("Bonus δ für Platz 1", *bounds("delta_slider"), key="delta_slider", step=0.01,
                      help="Platz 1 liegt näher am Tor und spart 1 + δ statt 1 Umschlagzeit. Mit 0 ist kein Platz ausgezeichnet.")
    mode = st.selectbox("Modus", C.MODES, key="mode_select",
                        help="Selbstspiel: die Paare lernen gegeneinander. Other-Play: der Partner wird je Runde mit zufällig umbenannten Plätzen angetroffen; die Gruppe der Umbenennungen sollte zu den Symmetrien des Spiels passen.")
    st.header("🎚️ Lernen")
    eta = st.slider("Lernrate η", *bounds("eta_slider"), key="eta_slider", step=0.01, help="Wie stark ein Treffer die Parameter eines Fahrers verschiebt.")
    sigma0 = st.slider("Anfangsstreuung σ₀", *bounds("sigma_slider"), key="sigma_slider", step=0.05,
                       help="Streuung der Anfangsparameter. Mit 0 starten alle Fahrer mit gleichverteilter Wahl; der Zufall der Ziehungen entscheidet dann allein.")
    steps = st.slider("Schritte (Runden)", *bounds("steps_slider"), key="steps_slider", step=500, help=f"Wie viele Runden jedes Paar lernt. Paare · Schritte ist auf {V.thousands(C.RUN_BUDGET)} begrenzt.")
    st.header("👥 Paare und Vorgeschichte")
    pairs = st.slider("Zahl der Paare", *bounds("pairs_slider"), key="pairs_slider", step=1,
                      help="Unabhängig lernende Paare. Das Fremdspiel braucht mindestens zwei; kleine Zahlen streuen stark (E2).")
    t0 = st.slider("Gemeinsame Vorgeschichte T₀ (Schritte)", *bounds("t0_slider"), key="t0_slider", step=25,
                   help="0: die Paare starten unabhängig. Sonst starten sie in Gruppen von den Parametern einer Referenz-Paarung, die T₀ Schritte gelernt hat.")
    st.header("🎲 Lauf")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Zufalls-Seed ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"m_slider": int(m), "delta_slider": float(delta), "eta_slider": float(eta), "sigma_slider": float(sigma0), "steps_slider": int(steps), "pairs_slider": int(pairs),
                   "t0_slider": int(t0), "mode_select": mode, "seed_input": int(seed)})

settings = S.Settings(m=int(m), delta=float(delta), eta=float(eta), sigma0=float(sigma0), steps=int(steps), pairs=int(pairs), t0=int(t0), mode=mode, seed=int(seed))
used_pairs = S.budget_pairs(settings)
with st.spinner(f"Trainiere {V.thousands(used_pairs)} Paare über {V.thousands(settings.steps)} Schritte (rund {max(1, math.ceil(S.estimated_seconds(settings)))} Sekunden) …"):
    view = _analyse(settings)
verdicts = []   # Verdikte für den PDF-Export

# --- Hauptansicht --------------------------------------------------------------------------------------------------------------------------
st.markdown("## 📊 Mein Partner und ein Fremder")
if used_pairs < settings.pairs:
    st.info(f"Gerechnet mit {V.thousands(used_pairs)} statt {V.thousands(settings.pairs)} Paaren: Paare · Schritte ist auf {V.thousands(C.RUN_BUDGET)} begrenzt.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Eigenspiel (mein Partner)", V.de(view.own, 3), help="Mittel über die Paare der Erfolgswahrscheinlichkeit Σ p_A(a)·p_B(a) der beiden Fahrer desselben Paares.")
m2.metric("Fremdspiel (ein Fremder)", "–" if view.cross is None else V.de(view.cross, 3),
          help="Mittel von Σ p_A,i(a)·p_B,j(a) über alle Paare i ≠ j, exakt aus den Politiken. Mit einem Paar nicht definiert.")
m3.metric("Zufallswert 1/m", V.de(view.chance, 3), help="Das Fremdspiel, wenn die Paare ihre Plätze gleichverteilt wählen.")
m4.metric("Paare mit Konvention Platz 1", V.pct(view.frac0, 1), help="Anteil der Paare, deren größte Neigung (p_A + p_B) auf Platz 1 liegt; bei Gleichstand gilt der kleinste Platz.")

st.markdown("#### ❓ Kernfrage: Trifft ein Fremder meinen Partner?")
v_cross = V.verdict_cross(view)
show(v_cross)
verdicts.append(v_cross)

col_mat, col_conv = st.columns(2)
with col_mat:
    st.plotly_chart(Z.build_matrix(view), width="stretch", key="matrix_chart")
    st.caption(f"Spieler A des Paares i (Zeile) gegen Spieler B des Paares j (Spalte), die ersten {view.matrix.shape[0]} Paare, exakt. **Diagonale = Eigenspiel**, alles andere Fremdspiel."
               + (" Weiße Linien trennen Gruppen mit gleicher Vorgeschichte." if view.group is not None else ""))
with col_conv:
    st.plotly_chart(Z.build_conventions(view), width="stretch", key="conv_chart")
    st.caption("Auf welchen Platz sich die Paare festgelegt haben. Bei festen Konventionen ist das Fremdspiel die Summe der Quadrate dieser Anteile (genauer: Σ nₐ(nₐ−1)/(P(P−1)) über die Zahl der Paare je Platz).")
st.plotly_chart(Z.build_trace(view), width="stretch", key="trace_chart")
st.caption("Das erste Paar beim Lernen: die Wahrscheinlichkeit jedes Platzes über die Schritte, Spieler A durchgezogen, Spieler B gepunktet. Beide legen sich auf denselben Platz fest; welcher, entscheidet der Zufall des Trainings (und der Bonus).")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["1 Eigenspiel gegen Fremdspiel", "2 Wie groß muss der Bonus sein?", "3 Erwartungsfluss gegen Lauf", "4 Gemeinsame Vorgeschichte", "5 Other-Play"])
meas = _meas()
if meas is None:
    st.warning("Die vorgerechneten Messreihen (data/measurements.json) fehlen oder gehören zu einem anderen Rechencode: `python tools/make_measurements.py` ausführen. Die Experimente zeigen dann keine Reihen.")
else:
    blocks_text = f"{meas['blocks']} Blöcke zu je {meas['pairs_per_block']} Paaren"

# --- Tab 1 ---------------------------------------------------------------------------------------------------------------------------------
with tab1:
    st.subheader("🔬 Experiment E1: Eigenspiel gegen Fremdspiel")
    st.caption("Wie gut trifft ein Paar sich selbst, wie gut einen Fremden, wenn der Bonus wächst? Eigenspiel und Fremdspiel exakt aus den Politiken; die Reihe ist vorgerechnet. "
               + (f"Je Zelle {blocks_text} (Selbstspiel, Lernrate 0,1, Anfangsstreuung 0,1, 6 000 Schritte); ± ist der Standardfehler der Blöcke." if meas else ""))
    e1_m = st.selectbox("Zahl der Plätze m", C.MS, index=1, key="e1_m")
    if meas:
        row1 = V.float_keys(meas["e1"][str(e1_m)])
        ds = list(row1)
        st.plotly_chart(Z.build_own_cross(ds, [row1[d]["own"]["mean"] for d in ds], [row1[d]["cross"]["mean"] for d in ds], [row1[d]["frac0"]["mean"] for d in ds], e1_m,
                                          cross_se=[row1[d]["cross"]["se"] for d in ds]), width="stretch", key="e1_chart")
        table([{"Bonus δ": delta_label(d), "Eigenspiel": ms_mean(c, "own"), "Fremdspiel": ms_pm(c, "cross"), "Fremdspiel aus der Verteilung der Konventionen": ms_mean(c, "pred"),
                "Paare mit Konvention Platz 1": V.pct(c["frac0"]["mean"], 1), "Paare sicher (Eigenspiel > 0,9)": V.pct(c["sure"]["mean"], 1)} for d, c in row1.items()])
        v1 = V.verdict_e1(row1, e1_m)
        show(v1)
        verdicts.append(v1)
    st.markdown("**Mit den Reglern der Seitenleiste nachrechnen** (Lernrate, Anfangsstreuung, Schritte, Paare und Modus aus der Seitenleiste; m läuft durch 2 bis 16, δ durch 0 / 0,1 / 1, ohne Vorgeschichte, ein Block)")
    est1 = seconds_text(len(C.MS) * 3, settings)
    if st.button("▶️ Plätze und Bonus durchrechnen", key="e1_button", help=f"{len(C.MS) * 3} Trainingsläufe mit je {V.thousands(used_pairs)} Paaren; rechnet rund {est1} Sekunden."):
        st.session_state["e1_signature"] = settings
    if st.session_state.get("e1_signature") == settings:
        with st.spinner(f"Rechne {len(C.MS) * 3} Trainingsläufe (rund {est1} Sekunden) …"):
            live1 = _e1_live(S.with_(settings, pairs=used_pairs))
        table([{"m": mm, "Bonus δ": delta_label(d), "Eigenspiel": ms_mean(c, "own"), "Fremdspiel": ms_mean(c, "cross") if c["cross"]["mean"] is not None else "–",
                "Zufallswert 1/m": V.de(1.0 / mm, 3)} for mm, rows in live1.items() for d, c in rows.items()])
        st.caption(f"{V.thousands(used_pairs)} Paare je Zelle; das Fremdspiel streut bei so wenigen Paaren um die Größenordnung, die E2 zeigt.")

# --- Tab 2 ---------------------------------------------------------------------------------------------------------------------------------
with tab2:
    st.subheader("🔬 Experiment E2: Wie groß muss der Bonus sein?")
    st.caption("Der Anteil der Paare, die sich auf Platz 1 festlegen, steigt **glatt** mit dem Bonus, es gibt keine Schwelle. Als Maß dient der **Halbweg-Bonus**: der Bonus, bei dem dieser Anteil die Mitte zwischen 1/m und 1 erreicht "
               "(log-interpoliert zwischen den Gitterpunkten). Darunter: wovon er abhängt und wie stark kleine Stichproben streuen.")
    if meas:
        e1_all = {int(mm): V.float_keys(r) for mm, r in meas["e1"].items()}
        half = {mm: CV.half_way_delta(list(r), [r[d]["frac0"]["mean"] for d in r], mm) for mm, r in e1_all.items()}
        st.plotly_chart(Z.build_halfway(list(half), {"Selbstspiel": [half[mm][1] for mm in half]}), width="stretch", key="e2_half_chart")
        table([{"m": mm, "Halbweg-Bonus δ": V.de(half[mm][1], 2) if half[mm][1] is not None else "nie erreicht", "Anteil Platz 1 bei δ = 1": V.pct(e1_all[mm][1.0]["frac0"]["mean"], 1),
                "Fremdspiel bei δ = 1": ms_mean(e1_all[mm][1.0], "cross")} for mm in half])
        v2h = V.verdict_e2_halfway(half)
        show(v2h)
        verdicts.append(v2h)
        st.markdown("**Wovon hängt der Halbweg-Bonus ab?** (m = 4; Lernrate η, Anfangsstreuung σ₀, Schritte einzeln verändert, sonst η = 0,1, σ₀ = 0,1, 6 000 Schritte)")
        rows2 = {"η = 0,1, σ₀ = 0,1, T = 6.000 (Standard)": e1_all[4]}
        names = {"0.03|0.1|6000": "η = 0,03", "0.3|0.1|6000": "η = 0,3", "0.1|0.0|6000": "σ₀ = 0", "0.1|0.5|6000": "σ₀ = 0,5", "0.1|0.1|2000": "T = 2.000", "0.1|0.1|20000": "T = 20.000"}
        for k, label in names.items():
            rows2[label] = V.float_keys(meas["e2"][k])
        half2 = {label: CV.half_way_delta(list(r), [r[d]["frac0"]["mean"] for d in r], 4) for label, r in rows2.items()}
        dl = list(next(iter(rows2.values())))
        st.plotly_chart(Z.build_lines({label: (dl, [r[d]["frac0"]["mean"] for d in dl]) for label, r in rows2.items()}, "Bonus δ", "Anteil der Paare mit Konvention Platz 1",
                                      xticks=[delta_label(d) for d in dl], hline=0.25, hline_text="1/m = 0,25"), width="stretch", key="e2_sens_chart")
        table([{"Einstellung": label, "Halbweg-Bonus δ": V.de(half2[label][1], 2) if half2[label][1] is not None else "nie erreicht",
                "Anteil Platz 1 bei δ = 0,3": V.pct(rows2[label][0.3]["frac0"]["mean"], 1), "kleinstes Eigenspiel im Gitter": V.de(min(c["own"]["mean"] for c in rows2[label].values()), 3)} for label in rows2])
        v2s = V.verdict_e2_sensitivity(half2)
        show(v2s)
        verdicts.append(v2s)
        st.markdown("**Streuung kleiner Stichproben** (100 Stichproben aus je 20 Paaren mit 6.000 Schritten, und aus je 40 Paaren mit 20.000 Schritten)")
        sa, sb = st.columns(2)
        sm = sa.selectbox("Zahl der Plätze m", (2, 4), key="e2_small_m")
        sd_ = sb.selectbox("Bonus δ", C.SMALL_DELTAS, index=3, key="e2_small_delta", format_func=lambda x: V.de(x, 2))
        small_a = meas["small"]["20|6000"][str(sm)][str(sd_)]
        small_b = meas["small"]["40|20000"][str(sm)][str(sd_)]
        st.plotly_chart(Z.build_small_sample(small_a["values"], small_b["values"], ("20 Paare, 6.000 Schritte", "40 Paare, 20.000 Schritte"), 1.0 / sm, mark=0.81 if sm == 2 and sd_ == 0.2 else None),
                        width="stretch", key="e2_small_chart")
        for cell_ in (small_a, small_b):
            v2k = V.verdict_small_sample(cell_)
            show(v2k)
            verdicts.append(v2k)
    st.markdown("**Mit den Reglern der Seitenleiste nachrechnen** (m, Lernrate, Anfangsstreuung, Schritte, Paare und Modus aus der Seitenleiste; δ läuft durch das Gitter, ein Block)")
    est2 = seconds_text(len(C.DELTAS), settings)
    if st.button("▶️ Bonus-Gitter durchrechnen", key="e2_button", help=f"{len(C.DELTAS)} Trainingsläufe mit je {V.thousands(used_pairs)} Paaren; rechnet rund {est2} Sekunden."):
        st.session_state["e2_signature"] = settings
    if st.session_state.get("e2_signature") == settings:
        with st.spinner(f"Rechne {len(C.DELTAS)} Trainingsläufe (rund {est2} Sekunden) …"):
            live2 = _e2_live(S.with_(settings, pairs=used_pairs))
        d2 = list(live2)
        st.plotly_chart(Z.build_own_cross(d2, [live2[d]["own"]["mean"] for d in d2], [live2[d]["cross"]["mean"] for d in d2], [live2[d]["frac0"]["mean"] for d in d2], settings.m),
                        width="stretch", key="e2_live_chart")
        kind2, val2 = CV.half_way_delta(d2, [live2[d]["frac0"]["mean"] for d in d2], settings.m)
        st.caption(f"m = {settings.m}, {V.thousands(used_pairs)} Paare, ein Block. Halbweg-Bonus aus diesem einen Lauf: "
                   + (V.de(val2, 2) if val2 is not None else "nicht erreicht") + (" (schon beim ersten Gitterpunkt über dem Niveau)" if kind2 == "unter" else "") + "."
                   + (" Im Modus „alle Plätze“ steigt der Anteil auf Platz 1 ohne Koordination (E5)." if settings.mode == C.MODE_ALL else ""))

# --- Tab 3 ---------------------------------------------------------------------------------------------------------------------------------
with tab3:
    st.subheader("🔬 Experiment E3: Erwartungsfluss gegen Lauf")
    st.caption("Der **Erwartungsfluss** wendet statt des gezogenen Updates dessen Erwartungswert an (deterministisch, nur vom Start abhängig). Beide beginnen an denselben Anfangsparametern. "
               "Gemessen: bei wie vielen Paaren gewinnt Platz 1, und folgt der Lauf dem Fluss? "
               + (f"Vorgerechnet: {blocks_text}, Lernrate 0,1, Anfangsstreuung 0,1, 6 000 Schritte." if meas else ""))
    e3_m = st.selectbox("Zahl der Plätze m", C.E3_MS, index=1, key="e3_m")
    if meas:
        row3 = V.float_keys(meas["e3"][str(e3_m)])
        ds3 = list(row3)
        st.plotly_chart(Z.build_lines({"Platz 1 gewinnt im Fluss": (ds3, [row3[d]["det0"]["mean"] for d in ds3]), "Platz 1 gewinnt im Lauf": (ds3, [row3[d]["sto0"]["mean"] for d in ds3]),
                                       "Lauf = Fluss (gleiche Siegermarke)": (ds3, [row3[d]["agree"]["mean"] for d in ds3])}, "Bonus δ", "Anteil der Paare", xticks=[delta_label(d) for d in ds3],
                                      hline=1.0 / e3_m, hline_text=f"1/m = {V.de(1.0 / e3_m, 3)}"), width="stretch", key="e3_chart")
        table([{"Bonus δ": delta_label(d), "Platz 1 im Fluss": V.pct(c["det0"]["mean"], 1), "Platz 1 im Lauf": V.pct(c["sto0"]["mean"], 1), "Start-Favorit = Platz 1": V.pct(c["init0"]["mean"], 1),
                "Fluss = Lauf": V.pct(c["agree"]["mean"], 1), "Start-Favorit = Fluss": V.pct(c["init_det"]["mean"], 1), "Start-Favorit = Lauf": V.pct(c["init_sto"]["mean"], 1)} for d, c in row3.items()])
        v3 = V.verdict_e3(row3, e3_m)
        show(v3)
        verdicts.append(v3)
    st.markdown("**Mit den Reglern der Seitenleiste nachrechnen** (m, δ, Lernrate, Anfangsstreuung, Schritte und Paare aus der Seitenleiste; ohne Vorgeschichte und im Selbstspiel, ein Block)")
    est3 = seconds_text(2, settings)
    if st.button("▶️ Fluss und Lauf rechnen", key="e3_button", help=f"Ein Erwartungsfluss und ein Trainingslauf mit je {V.thousands(used_pairs)} Paaren; rechnet rund {est3} Sekunden."):
        st.session_state["e3_signature"] = settings
    if st.session_state.get("e3_signature") == settings:
        with st.spinner(f"Rechne Fluss und Lauf (rund {est3} Sekunden) …"):
            live3 = _e3_live(S.with_(settings, pairs=used_pairs))
        table([{"Platz 1 im Fluss": V.pct(live3["det0"]["mean"], 1), "Platz 1 im Lauf": V.pct(live3["sto0"]["mean"], 1), "Start-Favorit = Platz 1": V.pct(live3["init0"]["mean"], 1),
                "Fluss = Lauf": V.pct(live3["agree"]["mean"], 1), "Start-Favorit = Fluss": V.pct(live3["init_det"]["mean"], 1), "Start-Favorit = Lauf": V.pct(live3["init_sto"]["mean"], 1)}])
        st.caption(f"m = {settings.m}, δ = {V.de(settings.delta, 2)}, {V.thousands(used_pairs)} Paare, ein Block.")

# --- Tab 4 ---------------------------------------------------------------------------------------------------------------------------------
with tab4:
    st.subheader("🔬 Experiment E4: Gemeinsame Vorgeschichte")
    st.caption("Eine Referenz-Paarung lernt T₀ Schritte; jedes Paar einer Gruppe startet von ihren Parametern und lernt dann mit dem eigenen Partner weiter. Verglichen wird das Fremdspiel **innerhalb** einer Gruppe "
               "(gleiche Vorgeschichte) mit dem **zwischen** Gruppen. Die Referenz muss nicht fertig gelernt haben. T₀ = 0 heißt hier: gleicher Zufallsstart in der Gruppe (in der Seitenleiste: kein gemeinsamer Start). "
               + (f"Vorgerechnet: {meas['blocks']} Blöcke zu je {C.E4_REFS} Referenzen mit je {C.E4_PER_REF} Paaren, δ = 0, 6.000 Schritte beim Weiterlernen." if meas else ""))
    e4_m = st.selectbox("Zahl der Plätze m", C.E4_MS, index=1, key="e4_m")
    if meas:
        row4 = V.int_keys(meas["e4"][f"{e4_m}|0.0"])
        ts = list(row4)
        st.plotly_chart(Z.build_lines({"innerhalb der Gruppe (gleiche Vorgeschichte)": (ts, [row4[t]["within"]["mean"] for t in ts]), "zwischen Gruppen": (ts, [row4[t]["between"]["mean"] for t in ts]),
                                       "Eigenspiel der Referenz": (ts, [row4[t]["ref_own"]["mean"] for t in ts])}, "Länge der gemeinsamen Vorgeschichte T₀", "Erfolgswahrscheinlichkeit", xticks=ts,
                                      hline=1.0 / e4_m, hline_text=f"1/m = {V.de(1.0 / e4_m, 3)}"), width="stretch", key="e4_chart")
        table([{"T₀": t, "Fremdspiel innerhalb": ms_pm(c, "within"), "Fremdspiel zwischen": ms_pm(c, "between"), "Eigenspiel der Paare": ms_mean(c, "own"), "Eigenspiel der Referenz": ms_mean(c, "ref_own"),
                "Paare bei der Konvention der Referenz": V.pct(c["follow"]["mean"], 1)} for t, c in row4.items()])
        v4 = V.verdict_e4(row4, e4_m)
        show(v4)
        verdicts.append(v4)
        st.markdown("**Gemeinschaften statt Paare** (eine Zeile: m = 4, G Fahrer lernen zusammen, in jeder Runde bilden sie zufällig Paare, δ = 0)")
        comm = {int(g): c for g, c in meas["e4comm"].items()}
        table([{"Gemeinschaftsgröße G": g, **{f"innerhalb nach {t} Runden": V.de(c[t]["within"]["mean"], 3) for t in ("500", "1000", "2000")},
                "Gemeinschaften > 0,9 nach 500 Runden": V.pct(c["500"]["sure"]["mean"], 1), "zwischen Gemeinschaften nach 2.000 Runden": V.de(c["2000"]["between"]["mean"], 3)} for g, c in comm.items()])
        st.caption(f"{comm[2]['communities']} Gemeinschaften je Größe. Größere Gemeinschaften brauchen anfangs länger, holen aber auf; der Befund ist eine Zeile, kein eigenes Experiment.")
    st.markdown("**Mit den Reglern der Seitenleiste nachrechnen** (m, δ, Lernrate, Anfangsstreuung und Schritte aus der Seitenleiste; 8 Referenzen mit je 20 Paaren, ein Block, Selbstspiel)")
    est4 = seconds_text(len(C.E4_T0S), settings, pairs=C.E4_LIVE_REFS * C.E4_LIVE_PER_REF)
    if st.button("▶️ Vorgeschichten durchrechnen", key="e4_button", help=f"{len(C.E4_T0S)} Werte von T₀, je {C.E4_LIVE_REFS * C.E4_LIVE_PER_REF} Paare; rechnet rund {est4} Sekunden."):
        st.session_state["e4_signature"] = settings
    if st.session_state.get("e4_signature") == settings:
        with st.spinner(f"Rechne {len(C.E4_T0S)} Vorgeschichten (rund {est4} Sekunden) …"):
            live4 = _e4_live(settings)
        table([{"T₀": t, "Fremdspiel innerhalb": ms_mean(c, "within"), "Fremdspiel zwischen": ms_mean(c, "between"), "Eigenspiel der Paare": ms_mean(c, "own")} for t, c in live4.items()])
        st.caption(f"m = {settings.m}, δ = {V.de(settings.delta, 2)}; {C.E4_LIVE_REFS} Referenzen mit je {C.E4_LIVE_PER_REF} Paaren. Das Zwischenspiel streut bei nur {C.E4_LIVE_REFS} Referenzen stark.")

# --- Tab 5 ---------------------------------------------------------------------------------------------------------------------------------
with tab5:
    st.subheader("🔬 Experiment E5: Other-Play")
    st.caption("Other-Play trainiert gegen einen Partner, dessen Plätze je Runde durch eine zufällige Umbenennung übersetzt werden. Das erzwingt Konventionen, die unter den Umbenennungen bestehen. "
               "Drei Arme: **Selbstspiel**, **alle Plätze** (jede Umbenennung) und **Platz 1 fest** (nur Umbenennungen, die Platz 1 stehen lassen). Getestet wird **ohne** Umbenennung. "
               + (f"Vorgerechnet: {blocks_text}, Lernrate 0,1, Anfangsstreuung 0,1, 6 000 Schritte." if meas else ""))
    e5a, e5b = st.columns(2)
    e5_m = e5a.selectbox("Zahl der Plätze m", C.MS, index=2, key="e5_m")
    e5_d = e5b.selectbox("Bonus δ der Auswahl", C.OP_DELTAS, index=5, key="e5_delta", format_func=lambda x: V.de(x, 3))
    if meas:
        arms = {name: V.float_keys(meas["e5"][name][str(e5_m)]) for name in XP.ARMS}
        d5 = list(arms["Selbstspiel"])
        st.plotly_chart(Z.build_arms([delta_label(d) for d in d5], {name: ([arms[name][d]["cross"]["mean"] for d in d5], [arms[name][d]["own"]["mean"] for d in d5]) for name in arms}, 1.0 / e5_m),
                        width="stretch", key="e5_chart")
        table([{"Arm": name, "Eigenspiel": ms_mean(arms[name][e5_d], "own"), "Fremdspiel": ms_pm(arms[name][e5_d], "cross"), "Paare mit Konvention Platz 1": V.pct(arms[name][e5_d]["frac0"]["mean"], 1),
                "mittleres größtes p": ms_mean(arms[name][e5_d], "pmax")} for name in arms])
        v5 = V.verdict_e5(arms, e5_m, e5_d)
        show(v5)
        verdicts.append(v5)
        half5 = {name: {int(mm): CV.half_way_delta(C.OP_DELTAS, [V.float_keys(meas["e5"][name][str(mm)])[d]["frac0"]["mean"] for d in C.OP_DELTAS], int(mm)) for mm in C.MS} for name in XP.ARMS}
        st.markdown("**Halbweg-Bonus je Arm** (der Bonus, bei dem der Anteil der Paare auf Platz 1 die Mitte zwischen 1/m und 1 erreicht; „alle Plätze“: der Anteil steigt, aber ohne Koordination)")
        table([{"Arm": name, **{f"m = {mm}": (V.de(h[1], 2) if h[1] is not None else "nie") for mm, h in half5[name].items()}} for name in half5])
        st.plotly_chart(Z.build_halfway(list(C.MS), {name: [half5[name][mm][1] for mm in C.MS] for name in ("Selbstspiel", "Platz 1 fest")}), width="stretch", key="e5_half_chart")
    st.markdown("**Mit den Reglern der Seitenleiste nachrechnen** (m, δ, Lernrate, Anfangsstreuung, Schritte und Paare aus der Seitenleiste; die drei Arme, ein Block, ohne Vorgeschichte)")
    est5 = seconds_text(3, settings)
    if st.button("▶️ Drei Arme rechnen", key="e5_button", help=f"Drei Trainingsläufe mit je {V.thousands(used_pairs)} Paaren; rechnet rund {est5} Sekunden."):
        st.session_state["e5_signature"] = settings
    if st.session_state.get("e5_signature") == settings:
        with st.spinner(f"Rechne drei Arme (rund {est5} Sekunden) …"):
            live5 = _e5_live(S.with_(settings, pairs=used_pairs))
        table([{"Arm": name, "Eigenspiel": ms_mean(row[settings.delta], "own"), "Fremdspiel": ms_mean(row[settings.delta], "cross") if row[settings.delta]["cross"]["mean"] is not None else "–",
                "Paare mit Konvention Platz 1": V.pct(row[settings.delta]["frac0"]["mean"], 1)} for name, row in live5.items()])
        st.caption(f"m = {settings.m}, δ = {V.de(settings.delta, 2)}, {V.thousands(used_pairs)} Paare, ein Block. Other-Play findet nur, was in den Symmetrien des Spiels steckt: Platz 1 ist nur dann ausgezeichnet, wenn δ > 0.")

# --- Verwandte Demos -----------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔗 Verwandte Demos")
st.markdown(
    """
- **`koordinationsspiel-demo`** (Stück 1 der Reihe): die Gleichgewichte und die Schwelle p* der Hirschjagd mit zwei Spielern und zwei Handlungen; hier m Handlungen und Lernen.
- **`gleichgewichtsauswahl-demo`** (Stück 2): welche Konvention auf Dauer gewählt wird, mit Rauschen im 2×2-Fall.
- **`konventionswechsel-demo`** (Stück 3): wie lange ein Wechsel dauert; hier ist die Konvention noch nicht gelernt.
- **`globale-spiele-demo`** (Stück 4): Unsicherheit über den Zustand; hier ist das Spiel bekannt.
- **`schwaechstes-glied-demo`** (Stück 5): vier Lernregeln auf dem Minimum-Spiel; hier eine Lernregel auf dem Zuordnungsspiel.
- **`korreliert-demo`**: ein Vermittler empfiehlt Handlungen; hier gibt es kein Signal.
- **`policy-gradient-demo`**: REINFORCE als Softmax-Regression am Lagerroboter.
- **`marl-demo`**: mehrere Agenten lernen unabhängig an einer Kran-Vergabe.
"""
)

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Lernregel ist eine Modellannahme** (Softmax-REINFORCE ohne Basis, ein Parametersatz je Fahrer) | Gemessen ist diese Regel. Andere Lernregeln, Basislinien oder Netzparametrisierungen können andere Zahlen geben; über Menschen, Sprachmodelle oder echte Übergabeplätze sagt die Demo nichts. | `policy-gradient-demo` |
| **Die Auszahlung ist Zeitersparnis** (1 bei gleichem Platz, 1 + δ auf Platz 1) | Echte Plätze unterscheiden sich in vielem; der „offensichtliche“ Platz ist hier eine einzige Zahl, ohne Kultur, Sprache oder Gewohnheit. | – |
| **Schrittzahl und Streuung sind Voreinstellungen** (6.000 Schritte, Lernrate 0,1, Anfangsstreuung 0,1) | Die Lage der Bonus-Kurve hängt an Lernrate und Anfangsstreuung (E2), nicht an der Schrittzahl; andere Werte verschieben sie. | – |
| **Zwei Fahrer, gleichzeitige Wahl** | Mehr als zwei Fahrer, Netze und wiederholtes Treffen mit wechselnden Partnern sind nicht gerechnet (Netze: `konventionswechsel-demo`). | `konventionswechsel-demo` |
| **Keine Absprache** | Reden vor dem Spiel ist nicht modelliert; die gemeinsame Vorgeschichte ist ein gemeinsamer Anfangszustand der Lernenden, kein Signal. | – |
| **Other-Play nur im Lehrbuchfall** | Das Experiment baut den Mechanismus (Training gegen umbenannte Partner) im Zuordnungsspiel nach. Es ist kein Nachbau von Hanabi oder Overcooked und sagt nichts über deren Ergebnisse. Die Gruppe der Umbenennungen muss zu den Symmetrien des Spiels passen. | – |
| **Fokalpunkte sind hier eine Zahl** | Laut Abstract eines Preprints (Aharon u. a., arXiv 2601.22184) koordinieren über 20 Sprachmodelle ohne Kommunikation oft so gut wie Menschen, scheitern aber an Aufgaben mit zahlenbezogenem Alltagswissen oder kulturell geprägter Auffälligkeit; hier ist der „offensichtliche“ Platz ein Bonus. Über Sprachmodelle sagt die Demo nichts (Abstract, nicht im Volltext gelesen). | – |
| **Gemeinsame Erfahrung** | Laut Abstract von OvercookedV2 (Gessler u. a., arXiv 2503.17821) liegt das Zero-Shot-Versagen in Overcooked größtenteils an schlechter Zustandsabdeckung im Selbstspiel: in der Richtung vereinbar mit E4, aber kein Nachbau und keine Aussage zu Overcooked (Abstract, nicht im Volltext gelesen). | – |
| **Stichproben sind endlich** | Zellen sind Mittel über 4 Blöcke zu je 500 Paaren, Live-Läufe haben ein Block mit höchstens 500 Paaren; das Fremdspiel kleiner Stichproben streut stark (E2). | – |
    """
)

st.markdown("---")
with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $m$ Plätze, ein Bonus $\delta \ge 0$ auf Platz 1 und $P$ Paare von Fahrern, die je mit Softmax-REINFORCE lernen; gesucht sind das Eigenspiel (Partner), das Fremdspiel (Fremder) und was die gelernte
Konvention bestimmt. Die Herleitungen sind eigene Rechnung, durch Handrechnung, zweite Rechenwege und Simulation gestützt, nicht gegen Literatur geprüft.

| Symbol | Bedeutung |
|---|---|
| $\theta_k,\ p_k$ | Parameter und Politik von Fahrer $k$ (A oder B eines Paares) |
| $a,\ r$ | gewählter Platz und Belohnung |
| $\delta,\ w(a)$ | Bonus und Treffer-Belohnung $w(1) = 1 + \delta$, sonst 1 |
| $\eta,\ \sigma_0$ | Lernrate und Streuung der Anfangsparameter |
| $p^A_i,\ p^B_i$ | Politiken von Spieler A und B des Paares $i$ |
| $c_i,\ n_a,\ \pi_a$ | Konvention des Paares $i$, Zahl der Paare auf Platz $a$ und Anteil $n_a/P$ |
| $\varphi$ | Umbenennung der Plätze (Other-Play) |

**Politik und Belohnung** eines Fahrers, bei Treffer von Platz $a$:

$$p_k(a) = \frac{e^{\theta_k(a)}}{\sum_{a'} e^{\theta_k(a')}}, \qquad r = w(a)\cdot [\,a_A = a_B\,]$$

**Lernregel** (REINFORCE ohne Basis, $e_a$ der Einheitsvektor des gewählten Platzes):

$$\theta_k \leftarrow \theta_k + \eta\, r\,(e_a - p_k)$$

**Erfolgswahrscheinlichkeit** zweier unabhängiger Wahlen aus $p$ und $q$ (die einzige Formel hinter Eigen- und Fremdspiel):

$$\Pr(\text{Treffer}) = \sum_{a=1}^{m} p(a)\, q(a)$$

**Eigenspiel** und **Fremdspiel** (exakt aus den Politiken, Mittel über die Paare):

$$\text{Eigen} = \frac{1}{P} \sum_{i=1}^{P} \sum_a p^A_i(a)\, p^B_i(a)$$

$$\text{Fremd} = \frac{1}{P(P-1)} \sum_{i \ne j} \sum_a p^A_i(a)\, p^B_j(a)$$

**Fremdspiel aus der Verteilung der Konventionen.** Sind die Politiken rein ($p^A_i = p^B_i = e_{c_i}$), trifft Paar $i$ Paar $j$ genau dann, wenn $c_i = c_j$:

$$\text{Fremd} = \frac{\sum_a n_a (n_a - 1)}{P(P-1)} \;\approx\; \sum_a \pi_a^2$$

**Zufallsgrenze.** Sind die Konventionen gleichverteilt über die $m$ Plätze, so ist $\text{Fremd} - 1/m = (X - (m-1)) / (m(P-1))$ mit $X \sim \chi^2_{m-1}$, also die Standardabweichung

$$\mathrm{sd} = \frac{\sqrt{2(m-1)}}{m\,(P-1)}$$

**Erwartungswert des Updates** (Spieler A, $J$ die erwartete Belohnung, $\odot$ elementweise); der Erwartungsfluss wendet ihn statt des gezogenen Updates an:

$$J = \sum_a p_A(a)\, p_B(a)\, w(a)$$

$$\frac{E[\Delta\theta_A]}{\eta} = p_A \odot \bigl(p_B \odot w - J\bigr)$$

**Halbweg-Bonus.** Der Bonus, bei dem der Anteil der Paare mit Konvention auf Platz 1 die Mitte zwischen $1/m$ und $1$ erreicht, log-interpoliert zwischen den Gitterpunkten:

$$\pi_1(\delta^{1/2}) = \frac{1/m + 1}{2}$$

**Other-Play.** Die Marke von B wird durch $\varphi$ aus einer Gruppe übersetzt, Treffer bei $a_A = \varphi(a_B)$. Erwartete Belohnung von A bei Politiken $p$ (A) und $q$ (B): für **alle Plätze** ($m!$ Umbenennungen)

$$J_{\text{alle}} = \frac{1}{m}\,\bigl(1 + \delta\, p_1\bigr)$$

und für **Platz 1 fest** ($(m-1)!$ Umbenennungen, die Platz 1 stehen lassen)

$$J_{\text{fest}} = (1+\delta)\, p_1 q_1 + \frac{(1 - p_1)(1 - q_1)}{m - 1}$$

**Literatur.** Die Quellen stehen in der README mit Stand der Prüfung; Formeln und Zahlen dieser Demo sind eigene Rechnung.
        """
    )

st.markdown("---")
st.subheader("📄 Ergebnisse als PDF")
if st.button("PDF erstellen", key="pdf_button"):
    from fok_pdf_export import build_pdf
    st.session_state["pdf_bytes"] = build_pdf(view, verdicts)
if st.session_state.get("pdf_bytes"):
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="fokalpunkt.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
