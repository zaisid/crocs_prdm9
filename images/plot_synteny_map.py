#!/usr/bin/env python3
"""
plot_synteny_map.py

Draws a gene-order synteny map from the output folder of
prdm9_synteny_pipeline_v2.py:

  * top row  : the reference (turtle) PRDM9 neighbourhood, PRDM9 in purple
  * other rows: each species' matching genes, in that species' own gene order
  * colour   : every gene is coloured by its position in the turtle
               neighbourhood, so the same gene has the same colour everywhere.
               A smooth colour run left->right = order conserved; a reversed
               run = inversion; scrambled colours = rearranged.
  * arrows   : gene direction (strand)
  * grey     : genes in the species that are not part of the turtle set
               (long runs are collapsed to a "// n genes" marker)
  * dashed purple box/band: where PRDM9 would be expected, i.e. between the
               two anchor genes that flank it in the turtle. If the anchors
               sit next to each other, an empty "?" slot is drawn between them.

Two layouts:
  --layout stacked  one shared turtle row on top, species rows below (compact)
  --layout paired   a turtle row above EVERY species, joined by curved lines

Usage:
  python plot_synteny_map.py --results results --out synteny_map
  python plot_synteny_map.py --results results --out synteny_map --layout paired \
         --species amer_gat chin_gat aus_croc

Needs: pandas, numpy, matplotlib.
Output: <out>.png (transparent, 300 dpi) and <out>.svg (text as outlines).
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, PathPatch, Patch, Polygon
from matplotlib.path import Path as MPath

# --------------------------------------------------------------------------
# Edit these to taste
# --------------------------------------------------------------------------
GREEN_DARK, GREEN_LIGHT = "#12501b", "#b4e5a2"          # your poster colours
PURPLE, PURPLE_FILL = "#5b4a8a", "#d9cdec"
GREY, GREY_EDGE = "#dedcd3", "#9a978c"
GRADIENT = [GREEN_DARK, GREEN_LIGHT, "#f1e3b8", "#e0a279"]   # turtle order -> colour

# sheet name -> label shown on the left. Names not listed here are prettified.
DISPLAY_NAMES = {
    "turtle": "Painted turtle",
    "amer_gat": "American alligator",
    "chin_gat": "Chinese alligator",
    "aus_croc": "Australian saltwater crocodile",
    "gharial": "Indian gharial",
}

SLOT_IN = 0.52      # inches per gene slot (width)
ROW_IN = 0.55       # inches per y-unit (height)
LABEL_W = 6.2       # slots reserved on the left for species labels
GENE_W, GENE_H, HEAD = 0.88, 0.50, 0.30


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def pretty(name):
    return DISPLAY_NAMES.get(name, name.replace("_", " ").capitalize())


def load_reference(results, gene):
    f = results / "reference" / "reference_neighbourhood.tsv"
    if not f.exists():
        sys.exit(f"Missing {f} (run the pipeline first)")
    hood = pd.read_csv(f, sep="\t").sort_values("order").reset_index(drop=True)
    target = hood[hood.symbol == gene]
    if target.empty:
        sys.exit(f"{gene} not found in {f}")
    return hood, float(target.iloc[0]["order"])


def load_species(results, name, target_order, summary):
    """Return the table of matched genes on the species' synteny chromosome."""
    f = results / name / "flank_best_hits.tsv"
    if not f.exists():
        return None
    hits = pd.read_csv(f, sep="\t")
    if hits.empty:
        return None

    chrom = None
    if summary is not None and name in summary.index and "synteny_chrom" in summary.columns:
        v = summary.loc[name, "synteny_chrom"]
        chrom = v if isinstance(v, str) and v else None
    if chrom is None:                          # fall back: chromosome with most hits
        chrom = hits.c_chrom.value_counts().index[0]

    sub = (hits[hits.c_chrom == chrom]
           .sort_values("bitscore", ascending=False)
           .drop_duplicates("c_geneid")        # one croc gene = one slot
           .sort_values("c_order"))
    return chrom, sub, len(hits)


# --------------------------------------------------------------------------
# turning tables into "items" (one item = one slot on a row)
# --------------------------------------------------------------------------
def reference_items(hood, gene):
    return [dict(kind="gene", t_order=float(r.order), strand=r.strand, label=r.symbol,
                 target=(r.symbol == gene)) for r in hood.itertuples()]


def species_items(sub, target_order, max_gap, orient=True):
    """
    Returns (items, band, rho, flipped)
      items : list of dicts for the row, left to right
      band  : (i, j) item indices spanning the PRDM9-expected region, or None
    """
    rho = (sub.t_order.rank().corr(sub.c_order.rank()) if len(sub) >= 3 else np.nan)
    by_order = {int(r.c_order): r for r in sub.itertuples()}
    lo, hi = min(by_order), max(by_order)

    items, run = [], 0

    def flush():
        nonlocal run
        if run > max_gap:
            items.append(dict(kind="gap", n=run))
        else:
            items.extend(dict(kind="grey") for _ in range(run))
        run = 0

    for o in range(lo, hi + 1):
        if o in by_order:
            flush()
            r = by_order[o]
            items.append(dict(kind="gene", t_order=float(r.t_order), strand=r.c_strand,
                              label=r.t_symbol, c_order=o))
        else:
            run += 1
    flush()

    flipped = bool(orient and pd.notna(rho) and rho < 0)
    if flipped:                                    # show inverted species right-way-round
        items.reverse()
        for it in items:
            if it["kind"] == "gene":
                it["strand"] = "-" if it["strand"] == "+" else "+"

    # anchors = nearest turtle neighbours on each side of PRDM9 that were recovered
    left = sub[sub.t_order < target_order].sort_values("t_order").tail(1)
    right = sub[sub.t_order > target_order].sort_values("t_order").head(1)
    band = None
    if len(left) and len(right):
        def idx(c_order):
            return next(i for i, it in enumerate(items)
                        if it["kind"] == "gene" and it["c_order"] == c_order)
        i1, i2 = idx(int(left.iloc[0].c_order)), idx(int(right.iloc[0].c_order))
        a, b = min(i1, i2), max(i1, i2)
        if b - a == 1:                             # anchors adjacent: show the empty slot
            items.insert(b, dict(kind="slot"))     # slot now sits at index b, right anchor at b+1
            band = (a, b + 1)
        else:
            band = (a, b)
    return items, band, rho, flipped


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------
def gene_polygon(cx, y, strand):
    l, r = cx - GENE_W / 2, cx + GENE_W / 2
    h = GENE_H / 2
    if strand == "-":
        pts = [(r, y - h), (l + HEAD, y - h), (l, y), (l + HEAD, y + h), (r, y + h)]
    else:
        pts = [(l, y - h), (r - HEAD, y - h), (r, y), (r - HEAD, y + h), (l, y + h)]
    return pts


def draw_row(ax, y, items, cmap, norm, labels=False, band=None):
    """Draw one row. Returns {t_order: x_centre} for coloured genes."""
    centres = {}
    if band is not None:                           # PRDM9-expected region, behind genes
        a, b = band
        ax.add_patch(FancyBboxPatch(
            (a + 0.04, y - 0.52), (b + 1 - a) - 0.08, 1.04,
            boxstyle="round,pad=0,rounding_size=0.18", fc=PURPLE_FILL, ec=PURPLE,
            lw=1.6, ls=(0, (4, 3)), alpha=0.45, zorder=1))
        ax.text((a + b + 1) / 2, y - 0.78, "PRDM9 expected here", ha="center", va="top",
                fontsize=8.5, color=PURPLE, style="italic")

    for i, it in enumerate(items):
        cx = i + 0.5
        k = it["kind"]
        if k == "gene":
            target = it.get("target", False)
            fc = PURPLE if target else cmap(norm(it["t_order"]))
            ax.add_patch(Polygon(gene_polygon(cx, y, it["strand"]), closed=True, fc=fc,
                                 ec=GREEN_DARK, lw=2.0 if target else 1.2, zorder=3,
                                 joinstyle="round"))
            centres[it["t_order"]] = cx
            if labels:
                ax.text(cx - 0.1, y + 0.40, it["label"], rotation=60, ha="left", va="bottom",
                        fontsize=8.5, color=PURPLE if target else GREEN_DARK,
                        fontweight="bold" if target else "normal", rotation_mode="anchor")
        elif k == "grey":
            ax.add_patch(Polygon([(cx - .32, y - .13), (cx + .32, y - .13),
                                  (cx + .32, y + .13), (cx - .32, y + .13)], closed=True,
                                 fc=GREY, ec=GREY_EDGE, lw=0.8, zorder=2))
        elif k == "gap":
            ax.text(cx, y + 0.02, "//", ha="center", va="center", fontsize=14,
                    color=GREY_EDGE, fontweight="bold")
            ax.text(cx, y - 0.28, f"{it['n']} genes", ha="center", va="top",
                    fontsize=6.5, color=GREY_EDGE)
        elif k == "slot":
            ax.add_patch(FancyBboxPatch((cx - GENE_W / 2, y - GENE_H / 2), GENE_W, GENE_H,
                                        boxstyle="round,pad=0,rounding_size=0.1",
                                        fc="none", ec=PURPLE, lw=1.6, ls=(0, (3, 2)), zorder=3))
            ax.text(cx, y - 0.02, "?", ha="center", va="center", fontsize=13,
                    color=PURPLE, fontweight="bold", zorder=4)
    return centres


def ribbons(ax, y_top, top_centres, y_bot, bot_centres, cmap, norm):
    for t_order, xb in bot_centres.items():
        if t_order not in top_centres:
            continue
        xt = top_centres[t_order]
        y1, y2 = y_top - GENE_H / 2 - 0.03, y_bot + GENE_H / 2 + 0.03
        ym = (y1 + y2) / 2
        path = MPath([(xt, y1), (xt, ym), (xb, ym), (xb, y2)],
                     [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
        ax.add_patch(PathPatch(path, fc="none", ec=GREEN_DARK, lw=3.6, alpha=0.35,
                               zorder=2, capstyle="round"))          # dark underlay for contrast
        ax.add_patch(PathPatch(path, fc="none", ec=cmap(norm(t_order)), lw=2.2,
                               alpha=1.0, zorder=2.1, capstyle="round"))


def row_label(ax, y, title, subtitle):
    ax.text(-0.5, y + 0.12, title, ha="right", va="bottom", fontsize=11.5,
            fontweight="bold", color=GREEN_DARK)
    ax.text(-0.5, y - 0.06, subtitle, ha="right", va="top", fontsize=8.5, color="#4d6b50")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", default="synteny_map", help="output file stem (no extension)")
    ap.add_argument("--gene", default="PRDM9")
    ap.add_argument("--ref-name", default="turtle")
    ap.add_argument("--species", nargs="*", help="species folders to draw, in order (default: all in summary.tsv)")
    ap.add_argument("--layout", choices=["stacked", "paired"], default="stacked")
    ap.add_argument("--max-gap", type=int, default=3,
                    help="runs of unmatched genes longer than this are collapsed to '// n genes'")
    ap.add_argument("--no-flip", action="store_true", help="don't flip species whose order is reversed")
    args = ap.parse_args()

    hood, target_order = load_reference(args.results, args.gene)
    ref_chrom = hood.chrom.iloc[0]
    n_ref = len(hood) - 1

    summary = None
    sf = args.results / "summary.tsv"
    if sf.exists():
        summary = pd.read_csv(sf, sep="\t").set_index("species")
    species = args.species or (list(summary.index) if summary is not None else
                               [p.name for p in args.results.iterdir() if (p / "flank_best_hits.tsv").exists()])

    cmap = LinearSegmentedColormap.from_list("turtle_order", GRADIENT)
    norm = Normalize(hood.order.min(), hood.order.max())
    ref_items = reference_items(hood, args.gene)

    panels, skipped = [], []
    for sp in species:
        loaded = load_species(args.results, sp, target_order, summary)
        if loaded is None:
            skipped.append(sp)
            continue
        chrom, sub, _ = loaded
        items, band, rho, flipped = species_items(sub, target_order, args.max_gap, not args.no_flip)
        sub_txt = f"{chrom}  ·  {len(sub)} of {n_ref} neighbours"
        if pd.notna(rho):
            sub_txt += f"  ·  ρ = {rho:.2f}"
        if flipped:
            sub_txt += "  ·  shown flipped"
        panels.append(dict(name=sp, items=items, band=band, sub=sub_txt))
    if skipped:
        print("No usable flank hits for:", ", ".join(skipped))
    if not panels:
        sys.exit("Nothing to draw.")

    n_slots = max([len(ref_items)] + [len(p["items"]) for p in panels])

    # vertical positions
    ys = {}
    if args.layout == "stacked":
        ys["ref"] = [0.0]
        ys["sp"] = [-1.9 * (i + 1) - 0.3 for i in range(len(panels))]
        y_min = ys["sp"][-1] - 1.3
    else:
        step = 4.0
        ys["ref"] = [-step * i for i in range(len(panels))]
        ys["sp"] = [y - 2.1 for y in ys["ref"]]
        y_min = ys["sp"][-1] - 1.3
    y_max = 2.4  # room for the rotated turtle gene names

    x_min, x_max = -LABEL_W, n_slots + 2.2
    fig_w = (x_max - x_min) * SLOT_IN
    fig_h = (y_max - y_min) * ROW_IN + 0.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min - 0.9, y_max)
    ax.axis("off")

    # reference row(s)
    ref_centres = {}
    for i, y in enumerate(ys["ref"]):
        ref_centres = draw_row(ax, y, ref_items, cmap, norm, labels=(i == 0))
        row_label(ax, y, pretty(args.ref_name) + " (reference)", f"{ref_chrom}  ·  {args.gene} neighbourhood")

    # species rows
    for i, p in enumerate(panels):
        y = ys["sp"][i]
        centres = draw_row(ax, y, p["items"], cmap, norm, band=p["band"])
        row_label(ax, y, pretty(p["name"]), p["sub"])
        if args.layout == "paired":
            ribbons(ax, ys["ref"][i], ref_centres, y, centres, cmap, norm)

    # legend
    handles = [
        Patch(fc=cmap(0.15), ec=GREEN_DARK, label="gene (colour = position in turtle neighbourhood)"),
        Patch(fc=PURPLE, ec=GREEN_DARK, label=args.gene),
        Patch(fc=GREY, ec=GREY_EDGE, label="other gene in this species"),
        Patch(fc=PURPLE_FILL, ec=PURPLE, ls="--", label=f"where {args.gene} is expected"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2,
              frameon=False, fontsize=9, labelcolor=GREEN_DARK, handlelength=1.6)

    fig.savefig(f"{args.out}.png", dpi=300, transparent=True, bbox_inches="tight")
    fig.savefig(f"{args.out}.svg", transparent=True, bbox_inches="tight")
    print(f"Wrote {args.out}.png and {args.out}.svg  ({len(panels)} species, layout={args.layout})")


if __name__ == "__main__":
    main()
