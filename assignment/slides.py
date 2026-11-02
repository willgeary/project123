"""16:9 PDF slides drawn with matplotlib, in Vi's visual style, shared by every deck.

Style from vi.co: warm off-white pages, black text, a violet accent, Book/Medium weights of one sans-serif.
Vi's PP Neue Montreal is a licensed font, so this uses Helvetica Neue (macOS) with Vi's own Arial fallback.

Text uses named roles from one type scale, so every slide shares the same hierarchy:
    kicker > title > lead > heading > body > small > caption   (display: dark slides and stat numbers)

Coordinates are inches from the bottom-left. A deck's footer is its PDF title metadata:
    with PdfPages(path, metadata={"Title": "Vi Labs · Meridian · ..."}) as pdf: ...
"""

import logging
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

# Palette: vi.co CSS (--bs-body-bg, --bs-violet, --bs-majorelle-blue, beige), plus warm neutrals and one warning hue.
BG, PAPER, INK, MUTED, LINE = "#f8f5f2", "#ffffff", "#000000", "#5f5a54", "#e4ddd5"
ACCENT, ACCENT_2, BEIGE = "#4f31f5", "#5d47ed", "#ded0c2"
GRAY, LIGHT, ORANGE = "#3b3835", "#b5aca2", "#eb6834"  # chart: observed/reference, baseline, bias
TINT = "#e7e2fd"  # violet tint for highlighted nodes

MEDIUM, REGULAR = 500, 400
ROLES = {  # role: (size, weight, color)
    "kicker": (11, MEDIUM, ACCENT),
    "display": (40, MEDIUM, INK),
    "title": (30, MEDIUM, INK),
    "lead": (18, MEDIUM, INK),
    "subtitle": (18, REGULAR, INK),
    "heading": (16, MEDIUM, INK),
    "body": (14, REGULAR, INK),
    "small": (12, REGULAR, MUTED),
    "caption": (10, REGULAR, MUTED),
}
SIZE = {role: size for role, (size, _, _) in ROLES.items()}

plt.rcParams.update(
    {
        "text.parse_math": False,  # "$5–$10" is money, not mathtext
        # A list, not "sans-serif", so matplotlib falls back per glyph: DejaVu supplies the arrows Helvetica Neue lacks.
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "text.color": INK,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.fontsize": 10,
    }
)

# Arial and DejaVu have no Medium (500); falling back to their Book weight is expected, not worth a warning per glyph.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

W, H = 13.333, 7.5
TOP = 5.85  # content top edge; kicker and title sit above it


def money(x: float) -> str:
    """Dollars in thousands: 6800 -> "$6.8k"."""
    return f"${x / 1000:,.1f}k"


def section(pdf: PdfPages, label: str) -> None:
    """Set the kicker shown above every following slide title (e.g. "2 · How the list is ranked")."""
    pdf.kicker = label


def page(pdf: PdfPages, title: str, dark: bool = False) -> plt.Figure:
    """New slide: dark title slide, or a content slide with kicker, title, footer and page number."""
    fig = plt.figure(figsize=(W, H), facecolor=INK if dark else BG)
    if dark:
        text(fig, 0.6, 4.3, title, "display", color=BG, va="bottom", linespacing=1.15)
        return fig
    if getattr(pdf, "kicker", ""):
        text(fig, 0.6, H - 0.45, pdf.kicker.upper(), "kicker")
    long_title = max(len(line) for line in title.splitlines()) > 60  # longer one-liners would run off the slide
    text(fig, 0.6, H - 0.75, title, "title", size=24 if long_title else None)
    _footer(fig, pdf)
    return fig


def divider(pdf: PdfPages, number: int, title: str, subtitle: str) -> None:
    """Section opener: a large violet number, the section title and one line on what it answers.

    Also sets the kicker for the section's slides.
    """
    section(pdf, f"{number} · {title}")
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    text(fig, 0.6, 5.6, f"{number:02d}", "display", size=96, color=ACCENT)
    text(fig, 0.6, 3.75, title, "display")
    text(fig, 0.6, 2.95, subtitle, "subtitle", color=MUTED)
    _footer(fig, pdf)
    emit(pdf, fig)


def _footer(fig: plt.Figure, pdf: PdfPages) -> None:
    text(fig, 0.6, 0.42, pdf.infodict().get("Title", ""), "caption", va="bottom")
    text(fig, W - 0.6, 0.42, str(pdf.get_pagecount() + 1), "caption", ha="right", va="bottom")


def emit(pdf: PdfPages, fig: plt.Figure) -> None:
    pdf.savefig(fig)
    plt.close(fig)


def box(fig: plt.Figure, x: float, y: float, w: float, h: float) -> plt.Axes:
    """Axes for a content block."""
    return fig.add_axes((x / W, y / H, w / W, h / H))


def text(fig: plt.Figure, x: float, y: float, s: str, role: str = "body", size: int | None = None, **kw) -> None:
    """Top-anchored text in a type role. Break lines explicitly: matplotlib's wrapping loses bullet indents."""
    role_size, weight, color = ROLES[role]
    kw = {"color": color, "va": "top", **kw}
    fig.text(x / W, y / H, s, fontsize=size or role_size, weight=weight, **kw)


def _surface(ax: plt.Axes, color: str = PAPER) -> None:
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0,rounding_size=0.04",
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="none",
        )
    )


def card(fig: plt.Figure, x: float, y: float, w: float, h: float, big: str, label: str, color: str = ACCENT) -> None:
    """Stat callout: a big number with a short label."""
    _surface(box(fig, x, y, w, h))
    text(fig, x + 0.3, y + h - 0.3, big, "display", color=color)
    text(fig, x + 0.3, y + h - 1.25, textwrap.fill(label, 36), "small", linespacing=1.4)


def panel(
    fig: plt.Figure, x: float, y: float, w: float, h: float, title: str, body: str, width: int = 34, role: str = "body"
) -> None:
    """White card with a heading and wrapped body text."""
    _surface(box(fig, x, y, w, h))
    text(fig, x + 0.25, y + h - 0.22, title, "heading")
    text(fig, x + 0.25, y + h - 0.62, textwrap.fill(body, width), role, color=INK, linespacing=1.5)


def table(
    fig: plt.Figure,
    x: float,
    y: float,
    w: float,
    h: float,
    header: list[str],
    rows: list[list[str]],
    col_widths: list[float],
) -> None:
    """Table with a black header row on white cells, filling the given box."""
    ax = box(fig, x, y, w, h)
    ax.axis("off")
    t = ax.table(
        cellText=rows, colLabels=header, colWidths=col_widths, loc="upper left", cellLoc="left", bbox=(0, 0, 1, 1)
    )
    t.auto_set_font_size(False)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(LINE)
        cell.set_text_props(fontsize=SIZE["small"], color=BG if r == 0 else INK, weight=MEDIUM if r == 0 else REGULAR)
        cell.set_facecolor(INK if r == 0 else PAPER)
        cell.PAD = 0.1 / (w * col_widths[c] / sum(col_widths))  # 0.1 in, whatever the column width


def connect(ax: plt.Axes, a, b, color: str = ACCENT, ls: str = "-", lw: float = 2.0, alpha: float = 1.0) -> None:
    """Arrow from text node a to text node b, clipped at their boxes."""
    ax.annotate(
        "",
        xy=(0.5, 0.5),
        xytext=(0.5, 0.5),
        xycoords=b,
        textcoords=a,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            ls=ls,
            alpha=alpha,
            patchA=a.get_bbox_patch(),
            patchB=b.get_bbox_patch(),
            shrinkA=2,
            shrinkB=2,
        ),
    )


def style(ax: plt.Axes) -> None:
    """Quiet chart frame: no top/right spines, light warm grid, frameless legend."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(LIGHT)
    ax.grid(color=LINE, lw=0.6)
    ax.set_axisbelow(True)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
