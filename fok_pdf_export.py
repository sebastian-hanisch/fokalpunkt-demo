"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Kennzahlen und die Verdikte. Erst nach Klick erzeugt (DEMO-PLAYBOOK §1c)."""

from __future__ import annotations

from fpdf import FPDF

import fok_evaluation as V

_REPLACE = {"–": "-", "—": "-", "≤": "<=", "≥": ">=", "≠": "!=", "√": "Wurzel", "²": "2", "Σ": "Summe", "°": " Grad", "·": "*", "−": "-", "„": '"', "“": '"', "∞": "unendlich",
            "→": "->", "ε": "eps", "β": "beta", "α": "alpha", "θ": "theta", "κ": "kappa", "ρ": "rho", "λ": "lambda", "Λ": "Lambda", "φ": "phi", "σ": "sigma", "π": "pi", "δ": "delta",
            "η": "eta", "Φ": "Phi", "Δ": "Delta", "×": "x", "↔": "<->", "≈": "~", "∝": "~", "‖": "||", "½": "1/2", "⁻": "^-", "¹": "1", "⁵": "5", "⁶": "6", "⁰": "0", "⁴": "4", "³": "3",
            "₀": "0", "₁": "1", "⁸": "8", "⁷": "7", "⁹": "9", "∈": "in", "…": "...", "‚": ",", "‘": "'", "’": "'", "”": '"', "µ": "mu", "⊙": "*"}


def _latin(text: str) -> str:
    """Kernschriften kennen nur Latin-1: Sonderzeichen ersetzen, Markdown-Sterne entfernen."""
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.replace("**", "").replace("`", "").encode("latin-1", "replace").decode("latin-1")


def build_pdf(view: V.MainView, verdicts: list[tuple[str, str]]) -> bytes:
    s = view.settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _latin("Koordination ohne Absprache: der offensichtliche Platz"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _latin(f"{s.mode}; m = {s.m} Uebergabeplaetze; Bonus delta = {V.de(s.delta, 2)}; Lernrate eta = {V.de(s.eta, 2)}; Anfangsstreuung sigma0 = {V.de(s.sigma0, 2)}; "
                                f"{s.steps} Schritte; {view.pairs_used} Paare; gemeinsame Vorgeschichte T0 = {s.t0}; Zufalls-Seed {s.seed}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Kennzahlen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    cross = "nicht definiert (ein Paar)" if view.cross is None else V.de(view.cross, 3)
    pdf.multi_cell(0, 5, _latin(f"Eigenspiel: {V.de(view.own, 3)}; Fremdspiel: {cross}; Zufallswert 1/m: {V.de(view.chance, 3)}; Anteil der Paare mit Konvention Platz 1: {V.pct(view.frac0, 1)}; "
                                f"Anteil sicherer Paare (Eigenspiel > 0,9): {V.pct(view.sure, 1)}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Verdikte", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for _, text in verdicts:
        pdf.multi_cell(0, 5, _latin(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _latin("Lernregel (Softmax-REINFORCE) und Eigennutz sind Modellannahmen, keine Aussagen ueber Menschen, Sprachmodelle oder echte Uebergabeplaetze. Alle Zahlen sind eigene "
                                "Rechnung, durch Handrechnung, zweite Rechenwege und Simulation gestuetzt."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
