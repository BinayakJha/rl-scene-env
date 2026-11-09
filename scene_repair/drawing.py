"""Draw a scene on a matplotlib axis: the 5x5 board, the center 3x3, the objects and goals.

Used for the worked-example figure and the episode GIF. Pure matplotlib, no images.
"""
from matplotlib.patches import FancyBboxPatch, Rectangle

from .grid import GRID_SIZE

COLORS = ["#e07a5f", "#3d85c6", "#81b29a", "#f2cc8f", "#9b72cf"]  # box, cat, book, lamp, cup


def draw_state(ax, state, title="", show_goals=True):
    ax.set_xlim(-0.1, GRID_SIZE + 0.1)
    ax.set_ylim(GRID_SIZE + 0.1, -0.1)  # flip so y grows DOWNWARD (south), like the proposal
    ax.set_aspect("equal")
    ax.axis("off")
    # the center 3x3 region, lightly shaded
    ax.add_patch(Rectangle((1, 1), 3, 3, facecolor="#fff3d6", edgecolor="none", zorder=0))
    for n in range(GRID_SIZE + 1):
        ax.plot([0, GRID_SIZE], [n, n], color="#c8c8c8", lw=1, zorder=1)
        ax.plot([n, n], [0, GRID_SIZE], color="#c8c8c8", lw=1, zorder=1)
    for n in range(GRID_SIZE):  # coordinate labels
        ax.text(n + 0.5, -0.03, str(n), ha="center", va="bottom", fontsize=7, color="#888")
        ax.text(-0.05, n + 0.5, str(n), ha="right", va="center", fontsize=7, color="#888")
    for k, (x, y) in enumerate(state["layout"]):
        ax.add_patch(FancyBboxPatch((x + 0.1, y + 0.1), 0.8, 0.8,
                                    boxstyle="round,pad=0,rounding_size=0.15",
                                    facecolor=COLORS[k], edgecolor="#333", lw=1, zorder=2))
        ax.text(x + 0.5, y + 0.5, state["names"][k], ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=3)
    if title:
        ax.set_title(title, fontsize=9, pad=12)
    if show_goals:
        lines = []
        for c, ok in zip(state["constraints"], state["mask"]):
            lines.append(("✓ " if ok else "✗ ") + c.describe(state["names"]))
        lines.append(f"budget left: {state['budget']}")
        ax.text(0, GRID_SIZE + 0.25, "\n".join(lines), ha="left", va="top", fontsize=7.5,
                family="DejaVu Sans")
