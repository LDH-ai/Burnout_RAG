from __future__ import annotations


PRIMARY = "#4F7C5A"
MINT    = "#7FB69A"
YELLOW  = "#D7A84F"
CORAL   = "#CB7568"
BLUE    = "#6F91B6"

LIGHT_THEME: dict[str, str] = {
    "BG":       "#F6F1E8",
    "BG_SOFT":  "#ECE3D5",
    "CARD":     "#FFFDF8",
    "CARD_ALT": "#F8F1E6",
    "TEXT":     "#2F2A24",
    "MUTED":    "#776E62",
    "BORDER":   "rgba(77, 68, 56, 0.14)",
    "PRIMARY":  PRIMARY,
    "MINT":     MINT,
    "YELLOW":   YELLOW,
    "CORAL":    CORAL,
    "BLUE":     BLUE,
    "SHADOW":   "0 18px 48px rgba(72, 61, 47, 0.12)",
    "GRADIENT": (
        "radial-gradient(ellipse at 15% 15%, rgba(127,182,154,0.30) 0%, transparent 55%), "
        "radial-gradient(ellipse at 85% 80%, rgba(215,168,79,0.18) 0%, transparent 50%), "
        "radial-gradient(ellipse at 70% 10%, rgba(111,145,182,0.16) 0%, transparent 45%), "
        "radial-gradient(ellipse at 30% 85%, rgba(203,117,104,0.10) 0%, transparent 40%), "
        "linear-gradient(150deg, #F0EDE4 0%, #E8DFD0 45%, #E2ECE5 100%)"
    ),
}


def get_theme() -> dict[str, str]:
    return LIGHT_THEME
