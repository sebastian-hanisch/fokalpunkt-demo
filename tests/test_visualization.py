"""Figuren: bauen ohne Fehler, feste Achsen (fixedrange), richtige Zahl der Spuren und Werte aus der Auswertung."""

import numpy as np
import pytest

import fok_constants as C
import fok_evaluation as V
import fok_scenario as S
import fok_visualization as Z


@pytest.fixture(scope="module")
def view():
    return V.analyse(S.Settings(m=4, pairs=30, steps=300, seed=3))


@pytest.fixture(scope="module")
def view_groups():
    return V.analyse(S.Settings(m=4, pairs=30, steps=300, seed=3, t0=50))


def fixed(fig):
    return fig.layout.xaxis.fixedrange is True and fig.layout.yaxis.fixedrange is True


def test_matrix_is_square_with_the_exact_values_and_no_group_lines_without_history(view):
    fig = Z.build_matrix(view)
    k = min(30, C.MATRIX_SHOW)
    assert fixed(fig) and np.array(fig.data[0].z).shape == (k, k) and np.array_equal(np.array(fig.data[0].z), view.matrix) and len(fig.layout.shapes) == 0


def test_matrix_draws_lines_between_the_groups(view_groups):
    fig = Z.build_matrix(view_groups)
    boundaries = int((view_groups.group[1:] != view_groups.group[:-1]).sum())
    assert len(fig.layout.shapes) == 2 * boundaries and boundaries >= 1


def test_conventions_bars_sum_to_one_hundred_and_have_the_chance_line(view):
    fig = Z.build_conventions(view)
    assert len(fig.data[0].x) == 4 and sum(fig.data[0].y) == pytest.approx(100.0) and len(fig.layout.shapes) == 1 and fixed(fig)


def test_trace_has_two_lines_per_place(view):
    fig = Z.build_trace(view)
    assert len(fig.data) == 8 and fixed(fig) and len(fig.data[0].x) == len(view.trace_steps)
    assert fig.layout.yaxis.range == (-0.02, 1.02)


def test_trace_hides_the_second_player_legend_for_many_places():
    v = V.analyse(S.Settings(m=8, pairs=4, steps=100))
    fig = Z.build_trace(v)
    assert len(fig.data) == 16 and sum(1 for t in fig.data if t.showlegend is not False) == 8


def test_own_cross_figure_has_three_lines_and_a_reference(view):
    ds = [0.0, 0.1, 1.0]
    fig = Z.build_own_cross(ds, [0.99, 0.99, 0.99], [0.25, 0.26, 0.75], [0.25, 0.34, 0.86], 4, cross_se=[0.0, 0.001, 0.002])
    assert len(fig.data) == 3 and list(fig.data[0].x) == ["0", "0,1", "1"] and len(fig.layout.shapes) == 1 and fixed(fig)
    assert list(fig.data[1].error_y.array) == [0.0, 0.001, 0.002]


def test_line_figure_uses_category_axis_on_request():
    fig = Z.build_lines({"a": ([0, 25, 50], [0.1, 0.2, 0.3])}, "T₀", "Wert", xticks=[0, 25, 50], hline=0.25, hline_text="1/m")
    assert fig.layout.xaxis.type == "category" and list(fig.data[0].x) == ["0", "25", "50"] and len(fig.layout.shapes) == 1
    assert Z.build_lines({"a": ([1, 2], [0.1, 0.2])}, "x", "y").layout.xaxis.type == "linear"


def test_arms_figure_has_a_solid_cross_line_and_a_dotted_own_line_per_arm():
    fig = Z.build_arms(["0", "0,3"], {"Selbstspiel": ([0.2, 0.3], [0.99, 0.99]), "Platz 1 fest": ([0.3, 0.5], [0.6, 0.8])}, 0.125)
    assert len(fig.data) == 4 and [t.line.dash for t in fig.data] == [None, "dot", None, "dot"] and fig.data[0].line.color == fig.data[1].line.color and len(fig.layout.shapes) == 1 and fixed(fig)


def test_line_labels_come_from_the_ticks_so_the_decimal_comma_is_kept():
    fig = Z.build_lines({"a": ([0.0, 0.03], [0.1, 0.2])}, "δ", "y", xticks=["0", "0,03"])
    assert list(fig.data[0].x) == ["0", "0,03"]


def test_half_way_figure_leaves_gaps_for_missing_values_and_uses_a_log_axis():
    fig = Z.build_halfway([2, 4, 8], {"Selbstspiel": [0.25, None, 0.55]})
    assert fig.layout.yaxis.type == "log" and fig.data[0].connectgaps is False and fixed(fig)


def test_small_sample_histogram_marks_the_chance_value_and_the_plan_value():
    fig = Z.build_small_sample([0.5, 0.52, 0.81], [0.5, 0.51], ("a", "b"), 0.5, mark=0.81)
    assert len(fig.data) == 2 and len(fig.layout.shapes) == 2 and fixed(fig)
    assert len(Z.build_small_sample([0.5], [0.5], ("a", "b"), 0.5).layout.shapes) == 1


def test_bars_figure():
    fig = Z.build_bars(["a", "b"], {"x": [0.1, 0.2], "y": [0.3, 0.4]}, "Wert", chance=0.25, chance_text="1/m")
    assert len(fig.data) == 2 and len(fig.layout.shapes) == 1 and fixed(fig)


def test_delta_labels_on_the_category_axis_use_the_german_comma():
    assert [Z._cat_label(d) for d in (0.0, 0.001, 0.03, 0.1, 1.0, 3.0)] == ["0", "0,001", "0,03", "0,1", "1", "3"]
