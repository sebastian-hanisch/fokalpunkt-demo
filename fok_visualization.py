"""Plotly-Figuren (alle Achsen fest, fixedrange). Nur Darstellung: jede Zahl kommt fertig aus Kern, Auswertung oder Messreihe."""

from __future__ import annotations

import plotly.graph_objects as go

import fok_constants as C
import fok_evaluation as V


def _base(fig: go.Figure, title: str | None = None, height: int = 360) -> go.Figure:
    fig.update_layout(title=title, height=height, margin=dict(l=55, r=15, t=45 if title else 15, b=55), legend=dict(orientation="h", y=-0.28, x=0))
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _cat_label(x: float) -> str:
    """Beschriftung eines Gitterpunkts auf der Kategorieachse (deutsches Komma)."""
    return V.de(x, 3).rstrip("0").rstrip(",") if x else "0"


# --- Hauptansicht ---------------------------------------------------------------------------------------------------------------------------
def build_matrix(view: V.MainView) -> go.Figure:
    """Erfolgswahrscheinlichkeit Spieler A des Paares i (Zeile) gegen Spieler B des Paares j (Spalte): die Diagonale ist das Eigenspiel, alles andere Fremdspiel."""
    k = view.matrix.shape[0]
    labels = [str(i + 1) for i in range(k)]
    fig = go.Figure(go.Heatmap(z=view.matrix, x=labels, y=labels, zmin=0, zmax=1, colorscale="Viridis", colorbar=dict(title="Erfolg", thickness=12),
                               hovertemplate="Spieler A von Paar %{y} gegen Spieler B von Paar %{x}: %{z:.3f}<extra></extra>"))
    fig.update_xaxes(title="Spieler B des Paares …", type="category")
    fig.update_yaxes(title="Spieler A des Paares …", type="category", autorange="reversed")
    if view.group is not None:
        for i in range(1, k):
            if view.group[i] != view.group[i - 1]:
                fig.add_shape(type="line", x0=i - 0.5, x1=i - 0.5, y0=-0.5, y1=k - 0.5, line=dict(color="white", width=2))
                fig.add_shape(type="line", y0=i - 0.5, y1=i - 0.5, x0=-0.5, x1=k - 0.5, line=dict(color="white", width=2))
    return _base(fig, height=420)


def build_conventions(view: V.MainView) -> go.Figure:
    """Anteil der Paare je Konventionsmarke; gestrichelt 1/m (jede Marke gleich häufig)."""
    m = len(view.counts)
    share = view.counts / view.counts.sum()
    fig = go.Figure(go.Bar(x=[str(a + 1) for a in range(m)], y=100 * share, marker_color=[C.MARK_COLORS[a % len(C.MARK_COLORS)] for a in range(m)],
                           text=[V.pct(x, 0) for x in share], textposition="outside"))
    fig.add_hline(y=100.0 / m, line=dict(color=C.COLOR_GREY, dash="dash"), annotation_text=f"gleichverteilt: {V.pct(1.0 / m, 1)}", annotation_position="top right")
    fig.update_xaxes(title="Konvention des Paares (Platz)", type="category")
    fig.update_yaxes(title="Anteil der Paare (%)", range=[0, min(118, 100 * max(share.max(), 1.0 / m) + 22)])
    return _base(fig, height=320)


def build_trace(view: V.MainView) -> go.Figure:
    """Politik des ersten Paares über die Schritte: je Platz eine Farbe, Spieler A durchgezogen, Spieler B gepunktet."""
    m = view.trace.shape[2]
    fig = go.Figure()
    for a in range(m):
        col = C.MARK_COLORS[a % len(C.MARK_COLORS)]
        fig.add_trace(go.Scatter(x=view.trace_steps, y=view.trace[:, 0, a], mode="lines", name=f"Platz {a + 1}, Spieler A", line=dict(color=col, width=2), legendgroup=str(a)))
        fig.add_trace(go.Scatter(x=view.trace_steps, y=view.trace[:, 1, a], mode="lines", name=f"Platz {a + 1}, Spieler B", line=dict(color=col, width=2, dash="dot"), legendgroup=str(a),
                                 showlegend=m <= 4))
    fig.update_xaxes(title="Schritt")
    fig.update_yaxes(title="Wahrscheinlichkeit, den Platz zu wählen", range=[-0.02, 1.02])
    return _base(fig)


# --- Experimente ----------------------------------------------------------------------------------------------------------------------------
def build_own_cross(deltas: list[float], own: list[float], cross: list[float], frac0: list[float], m: int, own_se=None, cross_se=None) -> go.Figure:
    """E1/E2: Eigenspiel, Fremdspiel und Anteil der Paare auf Platz 1 gegen den Bonus (Kategorieachse, damit 0 und kleine Werte lesbar bleiben); gestrichelt 1/m."""
    x = [_cat_label(d) for d in deltas]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=own, mode="lines+markers", name="Eigenspiel (das eigene Paar)", line=dict(color=C.COLOR_OWN, width=3),
                             error_y=dict(type="data", array=own_se, visible=own_se is not None)))
    fig.add_trace(go.Scatter(x=x, y=cross, mode="lines+markers", name="Fremdspiel (Spieler aus anderen Paaren)", line=dict(color=C.COLOR_CROSS, width=3),
                             error_y=dict(type="data", array=cross_se, visible=cross_se is not None)))
    fig.add_trace(go.Scatter(x=x, y=frac0, mode="lines+markers", name="Anteil der Paare mit Konvention Platz 1", line=dict(color=C.COLOR_LINE, width=2, dash="dot")))
    fig.add_hline(y=1.0 / m, line=dict(color=C.COLOR_GREY, dash="dash"), annotation_text=f"1/m = {V.de(1.0 / m, 3)}", annotation_position="bottom right")
    fig.update_xaxes(title="Bonus δ", type="category")
    fig.update_yaxes(title="Erfolgswahrscheinlichkeit", range=[0, 1.04])
    return _base(fig)


def build_lines(series: dict, xtitle: str, ytitle: str, xticks: list | None = None, yrange=(0, 1.04), height: int = 360, hline: float | None = None, hline_text: str = "") -> go.Figure:
    """Mehrere Linien {Name: (x, y)} auf einer Kategorieachse (xticks) oder numerisch."""
    fig = go.Figure()
    palette = [C.COLOR_OWN, C.COLOR_CROSS, C.COLOR_LINE, C.COLOR_RED, "#9467bd", "#8c564b", "#7f7f7f"]
    for i, (name, (xs, ys)) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=[str(t) for t in xticks] if xticks is not None else xs, y=ys, mode="lines+markers", name=name, line=dict(color=palette[i % len(palette)], width=2.5)))
    if hline is not None:
        fig.add_hline(y=hline, line=dict(color=C.COLOR_GREY, dash="dash"), annotation_text=hline_text, annotation_position="bottom right")
    fig.update_xaxes(title=xtitle, type="category" if xticks is not None else "linear")
    fig.update_yaxes(title=ytitle, range=list(yrange))
    return _base(fig, height=height)


def build_arms(xticks: list[str], arms: dict[str, tuple[list[float], list[float]]], chance: float, height: int = 400) -> go.Figure:
    """E5: je Arm eine Farbe, Fremdspiel durchgezogen, Eigenspiel gepunktet; arms = {Name: (Fremdspiel, Eigenspiel)} über das Bonus-Gitter (Kategorieachse xticks)."""
    fig = go.Figure()
    palette = [C.COLOR_OWN, C.COLOR_CROSS, C.COLOR_LINE, C.COLOR_RED]
    for i, (name, (cross, own)) in enumerate(arms.items()):
        fig.add_trace(go.Scatter(x=xticks, y=cross, mode="lines+markers", name=f"{name}: Fremdspiel", line=dict(color=palette[i % 4], width=3), legendgroup=name))
        fig.add_trace(go.Scatter(x=xticks, y=own, mode="lines", name=f"{name}: Eigenspiel", line=dict(color=palette[i % 4], width=2, dash="dot"), legendgroup=name))
    fig.add_hline(y=chance, line=dict(color=C.COLOR_GREY, dash="dash"), annotation_text=f"1/m = {V.de(chance, 3)}", annotation_position="bottom right")
    fig.update_xaxes(title="Bonus δ", type="category")
    fig.update_yaxes(title="Erfolgswahrscheinlichkeit", range=[0, 1.04])
    return _base(fig, height=height)


def build_halfway(ms: list[int], values: dict[str, list[float | None]]) -> go.Figure:
    """Halbweg-Bonus (der Bonus, bei dem der Anteil auf Platz 1 den halben Weg erreicht) gegen m, je Reihe eine Linie; fehlende Werte (nie erreicht) bleiben Lücken."""
    fig = go.Figure()
    palette = [C.COLOR_CROSS, C.COLOR_LINE, C.COLOR_OWN, C.COLOR_RED]
    for i, (name, ys) in enumerate(values.items()):
        fig.add_trace(go.Scatter(x=[str(m) for m in ms], y=ys, mode="lines+markers", name=name, line=dict(color=palette[i % len(palette)], width=2.5), connectgaps=False))
    fig.update_xaxes(title="Zahl der Plätze m", type="category")
    fig.update_yaxes(title="Halbweg-Bonus δ", type="log")
    return _base(fig)


def build_small_sample(values_a: list[float], values_b: list[float], names: tuple[str, str], chance: float, mark: float | None = None) -> go.Figure:
    """Streuung des Fremdspiels über viele kleine Stichproben (Histogramm zweier Aufbauten); gestrichelt der Wert ohne Bonus (1/m), gepunktet eine markierte Zahl (z. B. 0,81)."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values_a, name=names[0], marker_color=C.COLOR_CROSS, opacity=0.65, xbins=dict(start=0.4, end=1.0, size=0.025)))
    fig.add_trace(go.Histogram(x=values_b, name=names[1], marker_color=C.COLOR_LINE, opacity=0.65, xbins=dict(start=0.4, end=1.0, size=0.025)))
    fig.update_layout(barmode="overlay")
    fig.add_vline(x=chance, line=dict(color=C.COLOR_GREY, dash="dash"), annotation_text=f"Erwartung ohne Bonus: {V.de(chance, 2)}", annotation_position="top right")
    if mark is not None:
        fig.add_vline(x=mark, line=dict(color=C.COLOR_RED, dash="dot"), annotation_text=f"{V.de(mark, 2)}", annotation_position="top left")
    fig.update_xaxes(title="Fremdspiel einer Stichprobe", range=[0.4, 1.0])
    fig.update_yaxes(title="Zahl der Stichproben")
    return _base(fig, height=320)


def build_bars(categories: list[str], series: dict[str, list[float]], ytitle: str, chance: float | None = None, chance_text: str = "") -> go.Figure:
    """Gruppierte Balken {Name: Werte} über Kategorien."""
    palette = [C.COLOR_CROSS, C.COLOR_LINE, C.COLOR_OWN, C.COLOR_RED]
    fig = go.Figure()
    for i, (name, ys) in enumerate(series.items()):
        fig.add_trace(go.Bar(x=categories, y=ys, name=name, marker_color=palette[i % len(palette)]))
    if chance is not None:
        fig.add_hline(y=chance, line=dict(color=C.COLOR_GREY, dash="dash"), annotation_text=chance_text, annotation_position="top right")
    fig.update_layout(barmode="group")
    fig.update_xaxes(type="category")
    fig.update_yaxes(title=ytitle, range=[0, 1.04])
    return _base(fig, height=320)
