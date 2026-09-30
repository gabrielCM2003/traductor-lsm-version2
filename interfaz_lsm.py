"""Interfaz para el usuario del traductor: tema visual, componentes de la
ventana y textos de retroalimentacion.

La ventana (SignLanguageApp en senas.py) arma su pantalla con estos
componentes. Aqui no hay nada de MediaPipe ni de reconocimiento: la parte de
retroalimentacion (feedback_for_result, guidance_feedback, how_to_sign) son
funciones puras que traducen lo que reporta HandTrackingThread (resultado de
cada sena y estado de cada frame) a mensajes para la persona, y se prueban
sin ventana (tests/test_interfaz.py).

Los consejos de correccion salen de medir las plantillas de cada palabra
(SignStats en senas.py): donde queda la muneca respecto a los hombros, que
tan cerca de la boca queda la punta del indice, si se usan las dos manos y
cuanto dura. Asi, por ejemplo, un HOLA hecho frente al pecho recibe "HOLA se
hace a la altura de la cabeza", en vez de un "no reconocido" sin mas.
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QTabWidget, QVBoxLayout, QWidget,
)

# --------------------------------------------------------------------------- #
# Tema
# --------------------------------------------------------------------------- #

COLORS = {
    "bg": "#0f172a",          # fondo de la ventana
    "card": "#1e293b",        # tarjetas
    "card_alt": "#273449",
    "border": "#334155",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
    "accent": "#38bdf8",      # azul: haciendo la sena
    "ok": "#22c55e",          # verde: reconocida
    "warn": "#f59e0b",        # ambar: corregir / repetir
    "error": "#ef4444",
    "info": "#a78bfa",        # morado: reconociendo
}

def rgba(hex_color: str, alpha: float) -> str:
    """'#rrggbb' -> 'rgba(r, g, b, a)'. En hojas de estilo de Qt un hex de 8
    digitos es #AARRGGBB (no #RRGGBBAA), asi que la transparencia va asi."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"


# Color del marco del video y de la etiqueta de estado segun la fase.
PHASE_STYLE = {
    "detenido": ("Cámara detenida", COLORS["border"]),
    "reposo": ("Listo", COLORS["border"]),
    "seña": ("Haciendo seña…", COLORS["accent"]),
    "clasificando": ("Reconociendo…", COLORS["info"]),
    "ok": ("Reconocida", COLORS["ok"]),
    "repetir": ("Repite la seña", COLORS["warn"]),
}

LEVEL_STYLE = {
    "ok": ("✓", COLORS["ok"]),
    "tip": ("💡", COLORS["accent"]),
    "warn": ("!", COLORS["warn"]),
    "error": ("✕", COLORS["error"]),
}

STYLESHEET = f"""
QMainWindow, QDialog {{ background: {COLORS['bg']}; }}
QWidget {{ color: {COLORS['text']}; font-size: 14px; }}
QToolTip {{ background: {COLORS['card_alt']}; color: {COLORS['text']}; border: 1px solid {COLORS['border']}; }}
QFrame#Card {{ background: {COLORS['card']}; border: 1px solid {COLORS['border']}; border-radius: 14px; }}
QLabel#CardTitle {{ color: {COLORS['muted']}; font-size: 12px; font-weight: 600; letter-spacing: 1px; }}
QLabel#AppTitle {{ font-size: 22px; font-weight: 700; }}
QLabel#AppSubtitle {{ color: {COLORS['muted']}; font-size: 13px; }}
QLabel#Muted {{ color: {COLORS['muted']}; }}
QPushButton {{
    background: {COLORS['card_alt']}; border: 1px solid {COLORS['border']}; border-radius: 10px;
    padding: 8px 14px;
}}
QPushButton:hover {{ border-color: {COLORS['accent']}; }}
QPushButton:disabled {{ color: {COLORS['muted']}; }}
QPushButton#Primary {{ background: {COLORS['accent']}; color: #0b1220; border: none; font-weight: 700; }}
QPushButton#Primary:hover {{ background: #7dd3fc; }}
QPushButton#Danger {{ background: transparent; border-color: {COLORS['error']}; color: {COLORS['error']}; }}
QComboBox {{
    background: {COLORS['card_alt']}; border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 5px 10px;
}}
QComboBox QAbstractItemView {{ background: {COLORS['card']}; selection-background-color: {COLORS['border']}; }}
QProgressBar {{ background: {COLORS['card_alt']}; border: none; border-radius: 5px; height: 10px; }}
QProgressBar::chunk {{ background: {COLORS['accent']}; border-radius: 5px; }}
QSlider::groove:horizontal {{ height: 6px; background: {COLORS['card_alt']}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {COLORS['accent']}; width: 16px; margin: -6px 0; border-radius: 8px; }}
QCheckBox {{ spacing: 8px; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QPlainTextEdit {{ background: {COLORS['card_alt']}; border: 1px solid {COLORS['border']}; border-radius: 8px; }}
QStatusBar {{ background: {COLORS['card']}; color: {COLORS['muted']}; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{
    background: {COLORS['card']}; color: {COLORS['muted']}; padding: 9px 18px; margin-right: 4px;
    border-top-left-radius: 10px; border-top-right-radius: 10px; font-weight: 600;
}}
QTabBar::tab:selected {{ background: {COLORS['card_alt']}; color: {COLORS['text']}; }}
QLabel#Hero {{ font-size: 40px; font-weight: 800; }}
QPushButton#Big {{ font-size: 18px; padding: 14px 26px; }}
QStatusBar QLabel {{ color: {COLORS['muted']}; padding: 0 8px; }}
"""

# --------------------------------------------------------------------------- #
# Retroalimentacion (funciones puras)
# --------------------------------------------------------------------------- #


@dataclass
class Feedback:
    level: str        # "ok" | "tip" | "warn" | "error"
    title: str
    body: str = ""


def _get(stats: Any, name: str, default: float = 0.0) -> float:
    if stats is None:
        return default
    if isinstance(stats, Mapping):
        return float(stats.get(name, default))
    return float(getattr(stats, name, default))


def zone_of(stats: Any) -> str:
    """Zona del cuerpo donde se hizo la sena, a partir de SignStats (todo en
    anchos de hombro): muneca respecto al centro de los hombros (dy > 0 es
    hacia abajo) y punta del indice respecto a la boca. Umbrales elegidos con
    las plantillas: MAMA queda a 0.22 de la boca, HOLA con la muneca 0.39 sobre
    los hombros, AYUDA/GRACIAS/POR FAVOR entre 0.45 y 0.69 bajo ellos."""
    if stats is None or not bool(_get(stats, "has_body", 1.0)):
        return ""
    if _get(stats, "tip_mouth", 9.0) <= 0.32:
        return "junto a la boca"
    dy = _get(stats, "wrist_dy")
    if dy <= -0.2:
        return "a la altura de la cabeza"
    if dy <= 0.3:
        return "a la altura de los hombros"
    if dy <= 0.95:
        return "frente al pecho"
    return "a la altura del abdomen"


def hands_of(stats: Any) -> str:
    """"con las dos manos" solo si se vieron dos manos. Lo contrario no se
    afirma: con las palmas juntas (POR FAVOR) MediaPipe suele ver una sola
    mano, asi que no ver dos no quiere decir que se haga con una."""
    return "con las dos manos" if _get(stats, "two_hands") >= 0.35 else ""


def how_to_sign(label: str, profile: Any) -> str:
    """Una linea de la guia: como se hace la palabra segun sus plantillas."""
    parts = [p for p in (hands_of(profile), zone_of(profile)) if p]
    text = ", ".join(parts) or "haz la seña completa"
    return f"{label}: {text[0].upper() + text[1:]} (~{_get(profile, 'duration_s', 2.0):.0f} s)."


def compare_to(label: str, profile: Any, stats: Any) -> list[str]:
    """Diferencias concretas entre como se hizo la sena (stats) y como se
    hace `label` (profile), en frases cortas para la persona."""
    tips: list[str] = []
    if profile is None or stats is None:
        return tips
    want, got = _get(profile, "two_hands"), _get(stats, "two_hands")
    if want >= 0.35 and got < 0.15:
        tips.append(f"{label} se hace con las dos manos.")
    # Al reves ("se hace con una sola mano") no se dice: ver hands_of.
    zone_want, zone_got = zone_of(profile), zone_of(stats)
    if zone_want and zone_got and zone_want != zone_got:
        tips.append(f"Tu mano quedó {zone_got}; {label} se hace {zone_want}.")
    dur_want, dur_got = _get(profile, "duration_s", 0.0), _get(stats, "duration_s", 0.0)
    if dur_want > 0 and 0 < dur_got < 0.45 * dur_want:
        tips.append("La hiciste muy rápido: haz el movimiento completo.")
    return tips


def feedback_for_result(info: Mapping[str, Any], profiles: Mapping[str, Any]) -> Optional[Feedback]:
    """Mensaje para el resultado de una sena con movimiento (auto_result_signal
    de HandTrackingThread). profiles: {PALABRA mostrada: SignStats}."""
    code = info.get("code", "")
    label = info.get("label", "")
    topk = list(info.get("topk") or [])
    stats = info.get("stats")
    too_long = bool(info.get("too_long"))
    long_tip = "Baja las manos al terminar cada seña." if too_long else ""

    if code == "error":
        return Feedback("error", "No pude analizar la seña", "Inténtalo de nuevo.")
    if code == "corta":
        return Feedback("warn", "Seña muy corta",
                        "Haz la seña completa antes de bajar las manos.")
    if code in ("deletreo", "deletreo_largo"):
        return None
    if info.get("kind") == "palabra":
        if code == "ok":
            conf = topk[0][1] if topk else 0.0
            body = ""
            if conf < 0.8 and len(topk) > 1:
                rival = topk[1][0]
                tips = compare_to(label, profiles.get(label), stats)
                body = f"Se pareció un poco a {rival}." + (" " + tips[0] if tips else "")
            return Feedback("ok", f"{label}", body)
        if code == "ambigua" and len(topk) > 1:
            a, b = topk[0][0], topk[1][0]
            tips = compare_to(a, profiles.get(a), stats)
            hint = tips[0] if tips else (
                f"{how_to_sign(a, profiles.get(a))} {how_to_sign(b, profiles.get(b))}"
                if profiles.get(a) is not None and profiles.get(b) is not None else "")
            body = " ".join(x for x in (hint, long_tip) if x) or "Repite la seña más marcada."
            return Feedback("warn", f"¿{a} o {b}?", body)
        # desconocida (o cualquier otro rechazo de palabra)
        closest = topk[0][0] if topk else ""
        tips = compare_to(closest, profiles.get(closest), stats) if closest else []
        body = " ".join(tips[:2]) or (f"La más parecida fue {closest}. " if closest else "") + \
            "Haz la seña completa y más despacio."
        if long_tip:
            body = f"{body} {long_tip}"
        return Feedback("warn", "No reconocí la seña", body)
    # letras con movimiento (J, K, Ñ, Q, X, Z)
    if code == "ok":
        return Feedback("ok", label, "")
    if topk:
        return Feedback("warn", f"¿{topk[0][0]}?",
                        "Haz el movimiento de la letra más marcado y baja la mano al terminar."
                        + (f" {long_tip}" if long_tip else ""))
    return None


def guidance_feedback(state: Mapping[str, Any], held_s: Mapping[str, float]) -> Optional[Feedback]:
    """Consejo en vivo segun el estado del frame (guidance_signal) y cuanto
    tiempo lleva cada condicion (held_s, en segundos, lo lleva la ventana).
    Devuelve el mas importante o None."""
    if state.get("too_close"):
        return Feedback("warn", "Mano muy cerca de la cámara", "Aléjala un poco para no perderla.")
    if state.get("hands", 0) > 0 and state.get("body_tracking") and not state.get("body_visible") \
            and held_s.get("sin_cuerpo", 0.0) >= 2.0:
        return Feedback("tip", "No veo tus hombros",
                        "Aléjate un poco o ajusta la cámara: las palabras necesitan ver tu cuerpo.")
    if state.get("two_raised_still") and held_s.get("dos_manos", 0.0) >= 1.5:
        return Feedback("tip", "Para deletrear usa una sola mano", "Las letras se hacen con una mano.")
    unsure = state.get("static_unsure")
    if unsure and held_s.get("duda", 0.0) >= 1.5:
        names = " o ".join(str(c[0]) for c in unsure[:2])
        return Feedback("tip", f"¿{names}?", "Mantén la mano quieta y ajusta la forma de los dedos.")
    if state.get("moving_letter") and held_s.get("moviendo", 0.0) >= 3.0:
        return Feedback("tip", "Mantén la mano quieta", "Una letra fija se escribe cuando la mano se detiene.")
    return None


# --------------------------------------------------------------------------- #
# Componentes
# --------------------------------------------------------------------------- #


class Card(QFrame):
    """Tarjeta con titulo; el contenido va en .body (QVBoxLayout)."""

    def __init__(self, title: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("Card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 14)
        outer.setSpacing(8)
        if title:
            t = QLabel(title.upper())
            t.setObjectName("CardTitle")
            outer.addWidget(t)
        self.body = QVBoxLayout()
        self.body.setSpacing(8)
        outer.addLayout(self.body)


class Pill(QLabel):
    """Etiqueta redondeada de color (estado de la sena, de la camara...)."""

    def __init__(self, text: str = "", color: str = COLORS["border"], parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set(text, color)

    def set(self, text: str, color: str) -> None:
        self.setText(text)
        self.setStyleSheet(
            f"background: {rgba(color, 0.13)}; color: {color}; border: 1px solid {color};"
            "border-radius: 11px; padding: 3px 12px; font-weight: 600; font-size: 13px;"
        )


class CandidateBars(QWidget):
    """Top-3 de la ultima sena: nombre, barra y porcentaje."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._rows = []
        for _ in range(3):
            row = QHBoxLayout()
            name = QLabel("")
            name.setMinimumWidth(96)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            bar.setFixedHeight(10)
            pct = QLabel("")
            pct.setObjectName("Muted")
            pct.setMinimumWidth(44)
            pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(name)
            row.addWidget(bar, stretch=1)
            row.addWidget(pct)
            layout.addLayout(row)
            self._rows.append((name, bar, pct))
        self.set_candidates([])

    def set_candidates(self, topk: Sequence[tuple[str, float]], winner_color: str = COLORS["accent"]) -> None:
        for i, (name, bar, pct) in enumerate(self._rows):
            visible = i < len(topk)
            for w in (name, bar, pct):
                w.setVisible(visible)
            if not visible:
                continue
            label, conf = topk[i]
            name.setText(str(label))
            bar.setValue(int(round(conf * 100)))
            pct.setText(f"{conf * 100:.0f}%")
            color = winner_color if i == 0 else COLORS["muted"]
            bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; border-radius: 5px; }}")


class FeedbackItem(QFrame):
    def __init__(self, fb: Feedback, parent: Optional[QWidget] = None):
        super().__init__(parent)
        icon, color = LEVEL_STYLE.get(fb.level, LEVEL_STYLE["tip"])
        self.setStyleSheet(
            f"QFrame {{ background: {rgba(color, 0.10)}; border-left: 4px solid {color}; border-radius: 8px; }}"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        ic = QLabel(icon)
        ic.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: 700; border: none; background: transparent;")
        ic.setFixedWidth(24)
        layout.addWidget(ic, alignment=Qt.AlignmentFlag.AlignTop)
        text = QLabel(
            f"<b style='color:{color}'>{html.escape(fb.title)}</b>"
            + (f"<br><span style='color:{COLORS['text']}'>{html.escape(fb.body)}</span>" if fb.body else "")
        )
        text.setWordWrap(True)
        text.setStyleSheet("border: none; background: transparent;")
        layout.addWidget(text, stretch=1)


class FeedbackPanel(QWidget):
    """Consejo en vivo (fijo arriba) + ultimos mensajes, el mas nuevo arriba."""

    MAX_ITEMS = 4

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)
        self._live: Optional[FeedbackItem] = None
        self._live_key: Optional[tuple] = None
        self._items: list[FeedbackItem] = []
        self._empty = QLabel("Aquí verás si la seña salió bien y cómo corregirla.")
        self._empty.setObjectName("Muted")
        self._empty.setWordWrap(True)
        self._layout.addWidget(self._empty)
        self._layout.addStretch()

    def set_live(self, fb: Optional[Feedback]) -> None:
        key = None if fb is None else (fb.level, fb.title, fb.body)
        if key == self._live_key:
            return
        self._live_key = key
        if self._live is not None:
            self._live.setParent(None)
            self._live.deleteLater()
            self._live = None
        if fb is not None:
            self._live = FeedbackItem(fb)
            self._layout.insertWidget(0, self._live)
        self._refresh_empty()

    def add(self, fb: Feedback) -> None:
        item = FeedbackItem(fb)
        self._layout.insertWidget(1 if self._live is not None else 0, item)
        self._items.insert(0, item)
        while len(self._items) > self.MAX_ITEMS:
            old = self._items.pop()
            old.setParent(None)
            old.deleteLater()
        self._refresh_empty()

    def clear(self) -> None:
        for item in self._items:
            item.setParent(None)
            item.deleteLater()
        self._items.clear()
        self.set_live(None)
        self._refresh_empty()

    def _refresh_empty(self) -> None:
        self._empty.setVisible(self._live is None and not self._items)


def sentence_html(history: Sequence[str], current: str, pending: int, max_words: int = 24) -> str:
    """Texto traducido: palabras anteriores en gris, la palabra en curso en
    blanco y las letras pendientes (de la sena en curso, que una palabra
    todavia puede reemplazar) en azul claro."""
    words = [html.escape(w) for w in history[-max_words:]]
    prev = f"<span style='color:{COLORS['muted']}'>{' '.join(words)}</span>" if words else ""
    pending = max(0, min(pending, len(current)))
    fixed, tail = (current[:-pending], current[-pending:]) if pending else (current, "")
    cur = (
        f"<span style='color:{COLORS['text']}'>{html.escape(fixed)}</span>"
        f"<span style='color:{COLORS['accent']}; text-decoration: underline'>{html.escape(tail)}</span>"
        f"<span style='color:{COLORS['accent']}'>▏</span>"
    )
    if not prev and not current:
        return f"<span style='color:{COLORS['muted']}'>Lo que signes aparecerá aquí.</span>"
    return f"{prev} {cur}" if prev else cur


GUIDE_STEPS = [
    ("1. Colócate", "Que la cámara vea tu cara, tus hombros y tus manos. Con las manos "
                    "abajo, en reposo, no se reconoce nada."),
    ("2. Letras fijas (A–Y)", "Sube una mano sobre la línea punteada y mantenla quieta: "
                              "la letra se escribe en un momento."),
    ("3. Letras con movimiento (J, K, Ñ, Q, X, Z)", "Sube la mano, haz el movimiento de la letra "
                                                   "de corrido y bájala."),
    ("4. Palabras", "Sube las manos, haz la seña completa y bájalas: la palabra se escribe sola."),
    ("5. Terminar una palabra", "Baja las manos un momento o presiona Enter."),
]

ALPHABET = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "Ñ",
            "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
DYNAMIC = {"J", "K", "Ñ", "Q", "X", "Z"}


class SpriteLabel(QLabel):
    """Imagen animada a partir de una tira horizontal de cuadros (los _anim.png
    de manual/). Sin tira, muestra la imagen fija. La anima ManualWidget."""

    def __init__(self, still: Optional[QPixmap], strip: Optional[QPixmap], height: int,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(height)
        self._frames: list[QPixmap] = []
        self._i = 0
        if strip is not None and still is not None and still.width() > 0:
            fw = still.width()
            for k in range(strip.width() // fw):
                frame = strip.copy(k * fw, 0, fw, strip.height())
                self._frames.append(frame.scaledToHeight(height, Qt.TransformationMode.SmoothTransformation))
        self._still = still.scaledToHeight(height, Qt.TransformationMode.SmoothTransformation) if still else None
        if self._still is not None:
            self.setPixmap(self._still)

    @property
    def animated(self) -> bool:
        return len(self._frames) > 1

    def step(self) -> None:
        if self._frames:
            self._i = (self._i + 1) % len(self._frames)
            self.setPixmap(self._frames[self._i])


def _badge(text: str, color: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"background: {rgba(color, 0.15)}; color: {color}; border-radius: 8px; padding: 2px 8px;"
        "font-size: 12px; font-weight: 600;"
    )
    return lbl


class ManualWidget(QWidget):
    """Manual de señas: abecedario (fijas y con movimiento), palabras y cómo
    usar el traductor. Las ilustraciones salen de manual/ (generar_manual.py)
    y las animaciones solo corren mientras el manual se ve."""

    def __init__(self, manual_dir: Path, word_descriptions: Mapping[str, str],
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._manual_dir = manual_dir
        self._sprites: dict[int, list[SpriteLabel]] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        note = QLabel("Las señas se ven en espejo, como te verás en la pantalla del traductor.")
        note.setObjectName("Muted")
        layout.addWidget(note)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._scroll(self._letters_page()), "Abecedario")
        self.tabs.addTab(self._scroll(self._words_page(word_descriptions)), "Palabras")
        self.tabs.addTab(self._scroll(self._guide_page()), "Cómo usar")
        layout.addWidget(self.tabs, stretch=1)
        credits = QLabel("Letras con movimiento: plantillas del dataset del CICESE (CC BY 4.0). "
                         "Letras fijas y palabras: grabaciones del equipo Chili Mix.")
        credits.setObjectName("Muted")
        credits.setWordWrap(True)
        credits.setStyleSheet("font-size: 12px;")
        layout.addWidget(credits)
        self._timer = QTimer(self)
        self._timer.setInterval(110)
        self._timer.timeout.connect(self._tick)

    # ---- paginas ----------------------------------------------------------

    def _scroll(self, inner: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(inner)
        return area

    def _pixmap(self, path: Path) -> Optional[QPixmap]:
        if not path.exists():
            return None
        pm = QPixmap(str(path))
        return None if pm.isNull() else pm

    def _letters_page(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setSpacing(12)
        cols = 5
        for n, letter in enumerate(ALPHABET):
            card = Card()
            card.setMinimumWidth(170)
            top = QHBoxLayout()
            big = QLabel(letter)
            big.setStyleSheet("font-size: 34px; font-weight: 800;")
            top.addWidget(big)
            top.addStretch()
            still = self._pixmap(self._manual_dir / "letras" / f"{letter}.png")
            strip = self._pixmap(self._manual_dir / "letras" / f"{letter}_anim.png")
            if letter in DYNAMIC:
                top.addWidget(_badge("con movimiento", COLORS["accent"]))
            elif still is not None:
                top.addWidget(_badge("fija", COLORS["muted"]))
            card.body.addLayout(top)
            if still is None:
                pending = QLabel("Ilustración pendiente")
                pending.setAlignment(Qt.AlignmentFlag.AlignCenter)
                pending.setMinimumHeight(150)
                pending.setObjectName("Muted")
                card.body.addWidget(pending)
            else:
                sprite = SpriteLabel(still, strip if letter in DYNAMIC else None, 150)
                card.body.addWidget(sprite)
                if sprite.animated:
                    self._sprites.setdefault(0, []).append(sprite)
            hint = QLabel("Haz el movimiento de corrido" if letter in DYNAMIC else "Mantén la mano quieta")
            hint.setObjectName("Muted")
            hint.setStyleSheet("font-size: 12px;")
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card.body.addWidget(hint)
            grid.addWidget(card, n // cols, n % cols)
        return page

    def _words_page(self, descriptions: Mapping[str, str]) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setSpacing(12)
        words_dir = self._manual_dir / "palabras"
        files = sorted(words_dir.glob("*.png")) if words_dir.is_dir() else []
        names = [f.stem for f in files if not f.stem.endswith("_anim")]
        if not names:
            empty = QLabel("Todavía no hay ilustraciones de palabras (generar_manual.py --palabras).")
            empty.setObjectName("Muted")
            grid.addWidget(empty, 0, 0)
        for n, name in enumerate(names):
            label = name.replace("_", " ")
            card = Card()
            title = QLabel(label)
            title.setStyleSheet("font-size: 26px; font-weight: 800;")
            card.body.addWidget(title)
            sprite = SpriteLabel(self._pixmap(words_dir / f"{name}.png"),
                                 self._pixmap(words_dir / f"{name}_anim.png"), 250)
            card.body.addWidget(sprite)
            if sprite.animated:
                self._sprites.setdefault(1, []).append(sprite)
            desc = descriptions.get(label, "")
            if desc:
                d = QLabel(desc)
                d.setWordWrap(True)
                d.setObjectName("Muted")
                card.body.addWidget(d)
            grid.addWidget(card, n // 3, n % 3)
        return page

    def _guide_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        for title, text in GUIDE_STEPS:
            card = Card()
            lbl = QLabel(f"<b>{html.escape(title)}</b><br>{html.escape(text)}")
            lbl.setWordWrap(True)
            card.body.addWidget(lbl)
            layout.addWidget(card)
        layout.addStretch()
        return page

    # ---- animacion --------------------------------------------------------

    def _tick(self) -> None:
        for sprite in self._sprites.get(self.tabs.currentIndex(), []):
            sprite.step()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._timer.start()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._timer.stop()     # en la Raspberry Pi no se gasta CPU con el manual cerrado


class ManualWindow(QDialog):
    """El manual en su propia ventana, para consultarlo con el traductor
    corriendo (no es modal: la camara sigue)."""

    def __init__(self, manual_dir: Path, word_descriptions: Mapping[str, str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Manual de señas")
        self.setModal(False)
        self.resize(1100, 780)
        layout = QVBoxLayout(self)
        self.manual = ManualWidget(manual_dir, word_descriptions)
        layout.addWidget(self.manual, stretch=1)
        layout.addLayout(close_row(self))


class StartPage(QWidget):
    """Menu inicial."""

    start_requested = pyqtSignal()
    manual_requested = pyqtSignal()
    quit_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.addStretch()
        box = Card()
        box.setMaximumWidth(620)
        hero = QLabel("Traductor LSM")
        hero.setObjectName("Hero")
        hero.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub = QLabel("Lengua de Señas Mexicana a texto y voz")
        sub.setObjectName("AppSubtitle")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text = QLabel("Reconoce el abecedario completo y las palabras HOLA, GRACIAS, POR FAVOR, "
                      "AYUDA y MAMÁ. Antes de empezar, revisa en el manual cómo se hace cada seña.")
        text.setWordWrap(True)
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box.body.addWidget(hero)
        box.body.addWidget(sub)
        box.body.addSpacing(10)
        box.body.addWidget(text)
        box.body.addSpacing(14)
        for label, name, signal in (("▶  Iniciar programa", "Primary", self.start_requested),
                                    ("📖  Ver manual de señas", "", self.manual_requested),
                                    ("Salir", "", self.quit_requested)):
            b = QPushButton(label)
            b.setObjectName(name or "Big")
            if name:
                b.setStyleSheet("font-size: 18px; padding: 14px 26px;")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(signal.emit)
            box.body.addWidget(b)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(box, stretch=1)
        row.addStretch()
        outer.addLayout(row)
        outer.addStretch()


class ManualPage(QWidget):
    """El manual a pantalla completa, con los botones para volver al menu o
    seguir al traductor (lo que se muestra despues de Iniciar programa)."""

    back_requested = pyqtSignal()
    continue_requested = pyqtSignal()

    def __init__(self, manual_dir: Path, word_descriptions: Mapping[str, str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        top = QHBoxLayout()
        back = QPushButton("←  Menú")
        back.clicked.connect(self.back_requested.emit)
        title = QLabel("Manual de señas")
        title.setObjectName("AppTitle")
        go = QPushButton("Continuar al traductor  ▶")
        go.setObjectName("Primary")
        go.setMinimumWidth(240)
        go.clicked.connect(self.continue_requested.emit)
        for w in (back, go):
            w.setCursor(Qt.CursorShape.PointingHandCursor)
        top.addWidget(back)
        top.addSpacing(12)
        top.addWidget(title)
        top.addStretch()
        top.addWidget(go)
        layout.addLayout(top)
        self.manual = ManualWidget(manual_dir, word_descriptions)
        layout.addWidget(self.manual, stretch=1)


class SettingsDialog(QDialog):
    """Ajustes tecnicos (camara, umbrales, dibujo, mano, voz, diagnostico).
    La ventana le pasa los controles ya creados, conectados a sus handlers."""

    def __init__(self, sections: Sequence[tuple[str, Sequence[Any]]], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Ajustes")
        self.setMinimumWidth(460)
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        for title, rows in sections:
            card = Card(title)
            for row in rows:
                if isinstance(row, QWidget):
                    card.body.addWidget(row)
                else:
                    card.body.addLayout(row)
            layout.addWidget(card)
        layout.addStretch()
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        outer.addLayout(close_row(self))


def close_row(dialog: QDialog) -> QHBoxLayout:
    """Boton "Cerrar" alineado a la derecha (los botones estandar de Qt
    salen en ingles sin un traductor instalado)."""
    row = QHBoxLayout()
    row.addStretch()
    button = QPushButton("Cerrar")
    button.setObjectName("Primary")
    button.clicked.connect(dialog.accept)
    row.addWidget(button)
    return row


def big_button(text: str, tooltip: str = "", object_name: str = "") -> QPushButton:
    b = QPushButton(text)
    if tooltip:
        b.setToolTip(tooltip)
    if object_name:
        b.setObjectName(object_name)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return b
