"""Presets: Vollständigkeit, gültige Werte, erstes Preset = Standardeinstellungen, Permalink-Grenzen, und je Preset die erwartete Verdikt-Zeile (DEMO-PLAYBOOK §11a, Punkt 7)."""

import pytest

import fok_constants as C
import fok_evaluation as V
import fok_presets as P
import fok_scenario as S


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name].strip(), name


def test_preset_values_are_inside_the_slider_bounds():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.choices is not None:
                assert preset[key] in spec.choices, (name, key)
            else:
                assert spec.lo <= preset[key] <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    assert S.Settings(**C.PRESETS[first]) == S.Settings()
    assert first.startswith("Standard")


def test_setting_specs_cover_every_preset_key_and_permalink_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_no_two_presets_are_identical():
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)


def test_defaults_in_the_specs_are_the_base_settings():
    for key, state_key in P.PRESET_KEYS.items():
        assert P.SETTING_SPECS[state_key].default == C.BASE[key], key


def test_preset_names_match_their_values():
    """Vertauschte Beschriftungen (DEMO-PLAYBOOK §11, Klasse 5): die Zahlen im Namen sind die Werte des Presets."""
    assert C.PRESETS["Fremde Paare (m = 16)"]["m"] == 16
    assert C.PRESETS["Kleiner Bonus (δ = 0,1)"]["delta"] == 0.1 and C.PRESETS["Großer Bonus (δ = 1)"]["delta"] == 1.0
    assert C.PRESETS["Gleiche Vorgeschichte (T₀ = 400)"]["t0"] == 400
    fest = C.PRESETS["Other-Play, Platz 1 fest (m = 8, δ = 0,3)"]
    assert (fest["m"], fest["delta"], fest["mode"]) == (8, 0.3, C.MODE_FIX0)
    falsch = C.PRESETS["Other-Play, falsche Gruppe (m = 4, δ = 0,3)"]
    assert (falsch["m"], falsch["delta"], falsch["mode"]) == (4, 0.3, C.MODE_ALL)


# --- Verdikt je Preset (kein Preset ohne Effekt darf einen Effekt melden) ---------------------------------------------------------------------------
# Plattformregel: Softmax-REINFORCE ist über tausende Schritte chaotisch, ein anderer numpy-Stand kann die Konventionen der 300 Paare anders verteilen. Fest verlangt werden deshalb nur Aussagen,
# die weit von der Verdikt-Schwelle (vier Standardabweichungen) entfernt sind; bei den Presets nahe am Zufall gilt die Art des Verdikts als „info oder success“ und die Zahl mit großer Marge.
EXPECTED = {
    "Standard (m = 4, ohne Bonus)": (("info", "success"), None),
    "Fremde Paare (m = 16)": (("info", "success"), None),
    "Kleiner Bonus (δ = 0,1)": (("info", "success"), None),
    "Großer Bonus (δ = 1)": (("success",), "häufiger als der Zufall"),
    "Gleiche Vorgeschichte (T₀ = 400)": (("success",), "Fremde zu Bekannten"),
    "Other-Play, Platz 1 fest (m = 8, δ = 0,3)": (("info",), "Weniger Festlegung, aber mehr Treffer mit Fremden"),
    "Other-Play, falsche Gruppe (m = 4, δ = 0,3)": (("warning",), "nicht festgelegt"),
}
# Fremdspiel je Preset: (Wert aus den vorgerechneten Reihen der README, erlaubte Abweichung) mit großer Marge (mehrere Standardabweichungen einer Stichprobe von 300 Paaren); Eigenspiel über 0,99
CROSS_BAND = {"Standard (m = 4, ohne Bonus)": (0.25, 0.03), "Fremde Paare (m = 16)": (0.0625, 0.015), "Kleiner Bonus (δ = 0,1)": (0.262, 0.04), "Großer Bonus (δ = 1)": (0.75, 0.12)}


def test_every_preset_has_an_expected_verdict():
    assert set(EXPECTED) == set(C.PRESETS)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_each_preset_shows_the_expected_verdict(name):
    view = V.analyse(S.settings_of_preset(name))
    kind, text = V.verdict_cross(view)
    kinds, want_text = EXPECTED[name]
    assert kind in kinds and (want_text is None or want_text in text), (name, text)
    if name in CROSS_BAND:
        centre, tol = CROSS_BAND[name]
        assert view.own > 0.99 and abs(view.cross - centre) < tol, (name, view.own, view.cross)
