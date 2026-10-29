"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Knopf, fährt jeden Regler an seine Grenzen und prüft, dass
kein Python-Fehler auftritt - insbesondere `StreamlitDuplicateElementId` (mehrere st.plotly_chart-Aufrufe ohne eindeutiges key=) und `StreamlitAPIException` bei Reglern mit
berechneten Grenzen. Die ▶️-Knöpfe laufen einzeln (die Schleife lässt sie aus)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import fok_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer
AUTOPLAY_MARKER = "▶"  # ▶: Knöpfe mit Rechenlast, einzeln getestet
PRESET_LABELS = set(C.PRESETS)


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join([w.value for w in at.warning] + [s.value for s in at.success] + [i.value for i in at.info] + [e.value for e in at.error])


def test_app_loads_without_exception():
    at = _fresh_app()
    assert "Ein Fremder trifft" in _texts(at)                    # Standard: m = 4, kein Bonus (nur zufällig oder, selten, knapp darüber)
    assert not [w for w in at.warning if "vorgerechneten Messreihen" in w.value]       # die Messdatei passt zum Code


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label or label in PRESET_LABELS:
            continue                                                 # Presets und Rechenknöpfe laufen einzeln
        at = _fresh_app()
        at.button[index].click().run()
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            if slider.disabled:
                continue
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run()
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("mode", C.MODES)
@pytest.mark.parametrize("edge", ["min", "max"])
def test_corner_matrix_all_sidebar_sliders_at_their_minima_or_maxima(mode, edge):
    """Eckfall-Matrix (DEMO-PLAYBOOK §1d): je Modus alle Regler zugleich am Minimum bzw. Maximum (m = 2 bzw. 16, δ 0 bzw. 3, η 0,03 bzw. 0,3, σ₀ 0 bzw. 0,5, Schritte 500 bzw. 20 000,
    Paare 1 bzw. 500, T₀ 0 bzw. 2 000). Das Maximum rechnet mit dem Budget (150 Paare bei 20 000 Schritten) und kann einige Sekunden dauern."""
    at = _fresh_app()
    at.selectbox(key="mode_select").set_value(mode).run()
    for slider in at.sidebar.slider:
        if slider.disabled:
            continue
        slider.set_value(slider.min if edge == "min" else slider.max)
    at.run()
    assert not at.exception, (mode, edge, [str(e) for e in at.exception])


def test_one_pair_says_there_is_no_stranger_and_shows_no_cross_play():
    at = _fresh_app()
    at.slider(key="pairs_slider").set_value(1).run()
    assert not at.exception and "Mit einem Paar gibt es keinen Fremden" in _texts(at)
    assert [m.value for m in at.metric if m.label.startswith("Fremdspiel")] == ["–"]


def test_largest_run_is_cut_to_the_budget_and_says_so():
    at = _fresh_app()
    at.slider(key="steps_slider").set_value(C.STEPS_MAX)
    at.slider(key="pairs_slider").set_value(C.PAIRS_MAX)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert any("Gerechnet mit 150 statt 500 Paaren" in i.value for i in at.info)


def test_default_run_is_not_cut():
    at = _fresh_app()
    assert not [i for i in at.info if i.value.startswith("Gerechnet mit")]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_loads_and_shows_a_verdict(name):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert _texts(at)
    assert at.session_state["m_slider"] == C.PRESETS[name]["m"] and at.session_state["mode_select"] == C.PRESETS[name]["mode"] and at.session_state["t0_slider"] == C.PRESETS[name]["t0"]


def test_history_preset_shows_groups_in_the_matrix_caption():
    at = _fresh_app()
    at.sidebar.button(key="preset_Gleiche Vorgeschichte (T₀ = 400)").click().run()
    assert any("Weiße Linien trennen Gruppen" in c.value for c in at.caption)
    assert not any("Weiße Linien" in c.value for c in _fresh_app().caption)


def _tables(at, column):
    return [df for df in at.dataframe if column in df.value.columns.tolist()]


def test_the_play_buttons_work_individually():
    at = _fresh_app()
    at.button(key="e1_button").click().run()
    assert not at.exception and len(_tables(at, "Zufallswert 1/m")) == 1 and len(_tables(at, "Zufallswert 1/m")[0].value) == 12

    at = _fresh_app()
    at.button(key="e2_button").click().run()
    assert not at.exception and any("Halbweg-Bonus aus diesem einen Lauf" in c.value for c in at.caption)

    at = _fresh_app()
    at.button(key="e3_button").click().run()
    assert not at.exception and any("ein Block" in c.value and "m = 4" in c.value for c in at.caption)

    at = _fresh_app()
    at.button(key="e4_button").click().run()
    assert not at.exception and len(_tables(at, "Eigenspiel der Paare")) == 2 and any("8 Referenzen mit je 20 Paaren" in c.value for c in at.caption)

    at = _fresh_app()
    at.button(key="e5_button").click().run()
    assert not at.exception and any(df.value["Arm"].tolist() == ["Selbstspiel", "alle Plätze", "Platz 1 fest"] for df in _tables(at, "Arm"))


def test_play_results_are_hidden_when_the_sidebar_settings_change():
    at = _fresh_app()
    at.button(key="e1_button").click().run()
    assert len(_tables(at, "Zufallswert 1/m")) == 1
    at.slider(key="m_slider").set_value(5).run()
    assert not at.exception and _tables(at, "Zufallswert 1/m") == []


def test_experiment_tabs_show_their_results_without_a_button():
    at = _fresh_app()
    texts = _texts(at)
    assert "**m = 4:** Eigenspiel ≥ 0,99 in 9 von 9 Zellen" in texts                    # E1
    assert "Halbweg-Bonus" in texts and "Wovon hängt der Halbweg-Bonus ab?" in texts and "100 Stichproben mit je 20 Paaren" in texts and "100 Stichproben mit je 40 Paaren" in texts          # E2
    assert "Ohne Bonus gewinnt im Fluss der Start-Favorit" in texts                         # E3
    assert "Fremdspiel innerhalb einer Gruppe mit gleicher Vorgeschichte" in texts          # E4
    assert "Selbstspiel: Eigenspiel" in texts and "Platz 1 fest: Eigenspiel" in texts       # E5
    at.selectbox(key="e5_delta").set_value(0.3).run()
    assert not at.exception and "braucht die richtige Gruppe" in _texts(at)


def test_selectboxes_of_the_experiments_work_for_every_value():
    for key, values in (("e1_m", C.MS), ("e3_m", C.E3_MS), ("e4_m", C.E4_MS), ("e5_m", C.MS), ("e2_small_m", (2, 4)), ("e2_small_delta", C.SMALL_DELTAS), ("e5_delta", C.OP_DELTAS)):
        for v in values:
            at = _fresh_app()
            at.selectbox(key=key).set_value(v).run()
            assert not at.exception, (key, v, [str(e) for e in at.exception])


def test_pdf_button_creates_the_download():
    at = _fresh_app()
    at.button(key="pdf_button").click().run()
    assert not at.exception and at.session_state["pdf_bytes"][:4] == b"%PDF"


def test_math_expander_has_one_display_block_per_equation():
    at = _fresh_app()
    exp = next(e for e in at.expander if e.label == "📐 Mathematische Formulierung")
    text = "\n".join(m.value for m in exp.markdown)
    blocks = text.count("$$") // 2
    assert text.count("$$") % 2 == 0 and blocks == 12
    chunks = text.split("$$")[1::2]
    assert all("\n" not in c.strip() and "\\\\" not in c for c in chunks)        # je Gleichung ein einzeiliger Block, kein Zeilenumbruch \\ darin (KaTeX-Fehler im Browser)


def test_permalink_sets_the_regulators_and_clamps_to_the_bounds():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["modus"] = C.MODE_FIX0
    at.query_params["m"] = "8"
    at.query_params["delta"] = "9"                       # über der Grenze: wird auf 3,0 begrenzt
    at.query_params["t"] = "1"                           # unter der Grenze: wird auf 500 begrenzt
    at.query_params["paare"] = "100"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["mode_select"] == C.MODE_FIX0 and at.session_state["m_slider"] == 8 and at.session_state["delta_slider"] == 3.0
    assert at.session_state["steps_slider"] == 500 and at.session_state["pairs_slider"] == 100


def test_permalink_ignores_values_outside_the_allowed_choices_and_garbage():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["modus"] = "Quatsch"
    at.query_params["delta"] = "nan"
    at.query_params["m"] = "abc"
    at.query_params["eta"] = ""
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["mode_select"] == C.MODE_SELF and at.session_state["delta_slider"] == C.DEFAULT_DELTA and at.session_state["m_slider"] == C.DEFAULT_M
    assert at.session_state["eta_slider"] == C.DEFAULT_ETA


def test_permalink_round_trip_after_a_preset_click():
    at = _fresh_app()
    at.sidebar.button(key="preset_Großer Bonus (δ = 1)").click().run()
    assert at.query_params["delta"] in (["1.0"], "1.0")
