#!/usr/bin/env python3
"""
prdm9_synteny_pipeline.py

Looks for the neighbourhood of PRDM9 (from a reference genome, e.g. painted
turtle) in one or more other genomes (e.g. crocodilians), then searches the
region where PRDM9 "should" be.

Steps (per species):
  0. (once) Parse reference GFF -> find PRDM9 -> take +/- N protein-coding
     neighbours -> write their proteins to a FASTA.
  1. Download genome / GFF / proteins from NCBI (needs the `datasets` CLI),
     unless you give your own file paths in the sheet.
  2. blastp: reference neighbourhood proteins vs the species proteome.
  3. Map protein hits -> genes using both GFFs, keep best species gene per
     reference gene (gene level, not isoform level).
  4. Synteny check: which chromosome do the neighbours land on, and is the
     gene order preserved (Spearman rho of gene order; negative = inverted).
  5. Pick the closest left/right neighbours of PRDM9 that sit on that
     chromosome ("anchors"); the region between them is where PRDM9 would be.
     Lists every annotated gene in that region.
  6. Cut that region out of the genome and run tblastn with the reference
     PRDM9 protein against it.
  7. Write one summary row per species. Judging presence/absence from the
     tblastn hits is still a manual step.

Usage:
  python prdm9_synteny_pipeline.py --sheet species.tsv --outdir results --threads 4

Sheet = tab-separated, header row, columns:
  name  accession                (required; accession may be blank if you give paths)
  gff  protein  genome           (optional; give all three or none)
One row must have name == --ref-name (default "turtle"); that row is the reference.

Requires: python3, pandas, biopython, matplotlib, BLAST+ (makeblastdb, blastp,
tblastn) on your PATH, and the NCBI `datasets` CLI if you want auto-download.
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote

import matplotlib

matplotlib.use("Agg")  # no display needed; figures are saved to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO

BLASTP_COLS = ["qseqid", "sseqid", "pident", "length", "evalue", "bitscore", "qlen", "slen"]
TBLASTN_COLS = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
                "qstart", "qend", "sstart", "send", "evalue", "bitscore"]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def run(cmd):
    cmd = [str(c) for c in cmd]
    print("   $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def need_tools(*names):
    missing = [n for n in names if shutil.which(n) is None]
    if missing:
        sys.exit(f"Missing required program(s) on PATH: {', '.join(missing)}")


def read_blast(path, cols):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=cols)
    return pd.read_csv(path, sep="\t", names=cols)


def clean_seqid(x):
    """With makeblastdb -parse_seqids, sseqid looks like 'ref|XP_123.1|'."""
    parts = str(x).split("|")
    return parts[1] if len(parts) > 1 else parts[0]


def union_length(intervals):
    """Total length covered by a set of (start, end) intervals (1-based, inclusive)."""
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None:
            cur_s, cur_e = s, e
        elif s > cur_e + 1:
            total += cur_e - cur_s + 1
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s + 1
    return total



# --------------------------------------------------------------------------
# sheet reading
# --------------------------------------------------------------------------
SHEET_COLS = ["name", "accession", "gff", "protein", "genome"]


def read_sheet(path):
    """
    Read the species sheet. Delimiter (tab / comma / semicolon) is detected from
    the file's contents, not its extension. Header names are lower-cased and
    stripped, every value is stripped, blank rows are dropped, and missing
    optional columns are filled with "".
    """
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Sheet not found: {path}")
    with open(path, encoding="utf-8-sig") as f:   # utf-8-sig removes an Excel BOM
        header = f.readline()
    counts = {d: header.count(d) for d in ("\t", ",", ";")}
    sep = max(counts, key=counts.get)
    if counts[sep] == 0:
        raise RuntimeError(f"Could not find a tab, comma or semicolon in the header row of {path}: {header!r}")

    df = pd.read_csv(path, sep=sep, dtype=str, encoding="utf-8-sig", skipinitialspace=True)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "name" not in df.columns:
        raise RuntimeError(f"Sheet has no 'name' column. Columns found: {list(df.columns)} "
                           f"(delimiter detected: {sep!r}). Expected: {SHEET_COLS}")
    for c in SHEET_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[SHEET_COLS].fillna("")
    df = df.apply(lambda col: col.str.strip())
    df = df[df["name"] != ""]                      # drop blank / trailing rows
    dupes = df["name"][df["name"].duplicated()].tolist()
    if dupes:
        raise RuntimeError(f"Duplicate names in sheet: {dupes}")
    return df.to_dict("records")

# --------------------------------------------------------------------------
# files: download / locate
# --------------------------------------------------------------------------
def download_ncbi(name, accession, datadir):
    if shutil.which("datasets") is None:
        raise RuntimeError(f"[{name}] need the NCBI 'datasets' CLI to download {accession} "
                           f"(or give file paths in the sheet)")
    d = datadir / name
    d.mkdir(parents=True, exist_ok=True)
    zpath = d / f"{accession}.zip"
    run(["datasets", "download", "genome", "accession", accession,
         "--include", "genome,gff3,protein", "--filename", zpath])
    with zipfile.ZipFile(zpath) as z:
        z.extractall(d)


def _must_exist(path, label, name):
    if not Path(path).is_file():
        raise RuntimeError(f"[{name}] {label} file not found: {path}  "
                           f"(relative paths are relative to where you ran the script: {Path.cwd()})")
    return Path(path)


def resolve_files(row, datadir, skip_download, need=("gff", "protein", "genome")):
    """
    Work out the gff / protein / genome files for one sheet row.
      * any path given in the sheet is used as-is (and must exist)
      * a GenBank file (.gbff etc.) in the `gff` column supplies proteins + genome too
      * anything still missing is looked for in data/<name>/, then downloaded
        from NCBI using the accession (if allowed)
    `need` lists which files are actually required (the reference species does
    not need its genome). Raises RuntimeError (never sys.exit) so one bad
    species cannot stop the whole loop.
    """
    name = row["name"]
    explicit = {k: row.get(k, "") for k in ("gff", "protein", "genome")}
    for k, v in explicit.items():
        if v:
            _must_exist(v, k, name)
    d = datadir / name

    # ---- GenBank flat file -------------------------------------------------
    if explicit["gff"] and is_gbff(explicit["gff"]):
        gb = Path(explicit["gff"])
        d.mkdir(parents=True, exist_ok=True)
        faa = Path(explicit["protein"]) if explicit["protein"] else d / "from_gbff_protein.faa"
        fna = Path(explicit["genome"]) if explicit["genome"] else d / "from_gbff_genome.fna"
        want_faa = not explicit["protein"] and not faa.exists()
        want_fna = ("genome" in need) and not explicit["genome"] and not fna.exists()
        if want_faa or want_fna:
            print(f"[{name}] extracting {'proteins ' if want_faa else ''}"
                  f"{'genome ' if want_fna else ''}from GBFF (one streaming pass, may take a few minutes)",
                  flush=True)
            # write to .part files first so an interrupted run never leaves a half-written FASTA
            faa_part = faa.with_name(faa.name + ".part") if want_faa else None
            fna_part = fna.with_name(fna.name + ".part") if want_fna else None
            parse_gbff(gb, faa_out=faa_part, fna_out=fna_part)
            if faa_part: faa_part.replace(faa)
            if fna_part: fna_part.replace(fna)
        return {"gff": gb, "protein": faa, "genome": fna}

    # ---- GFF + FASTAs: explicit paths first, then data/<name>/, then download ----
    patterns = {"gff": "**/genomic.gff", "protein": "**/protein.faa", "genome": "**/*_genomic.fna"}
    out = {k: Path(v) for k, v in explicit.items() if v}
    missing = [k for k in need if k not in out]

    def look():
        for k in list(missing):
            found = sorted(d.glob(patterns[k])) if d.exists() else []
            if found:
                out[k] = found[0]
                missing.remove(k)

    look()
    if missing:
        acc = row.get("accession", "")
        if skip_download or not acc:
            raise RuntimeError(f"[{name}] missing {missing}: nothing in {d} and download not possible "
                               f"(give the path(s) in the sheet, or an accession and drop --skip-download)")
        download_ncbi(name, acc, datadir)
        look()
    if missing:
        hint = ""
        if "gff" in missing and row.get("accession", "").startswith("GCA_"):
            hint = (" GenBank (GCA_) packages often have no GFF: download the .gbff for this "
                    "assembly and put its path in the 'gff' column.")
        raise RuntimeError(f"[{name}] could not find {missing} after looking in {d}.{hint}")
    return out


# --------------------------------------------------------------------------
# GFF parsing
# --------------------------------------------------------------------------
def parse_attrs(s):
    out = {}
    for kv in s.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = unquote(v)
    return out


def load_gff(path):
    """
    Returns (genes, cds)
      genes: one row per gene (all biotypes) with chrom/start/end/strand/symbol/
             description/biotype, plus `order` = rank along the chromosome among
             PROTEIN-CODING genes only (NaN for other biotypes).
      cds:   protein_id -> geneid lookup (this is the "lookup table" between
             XP_ accessions and genes).
    """
    genes, cds = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9:
                continue
            a = parse_attrs(c[8])
            geneid = next((x.split(":", 1)[1] for x in a.get("Dbxref", "").split(",")
                           if x.startswith("GeneID:")), None)
            if geneid is None:
                continue
            if c[2] in ("gene", "pseudogene"):
                genes.append(dict(geneid=geneid, symbol=a.get("Name"),
                                  description=a.get("description", ""),
                                  biotype=a.get("gene_biotype", ""),
                                  chrom=c[0], start=int(c[3]), end=int(c[4]), strand=c[6]))
            elif c[2] == "CDS" and a.get("protein_id"):
                cds.append(dict(protein=a["protein_id"], geneid=geneid))

    genes = pd.DataFrame(genes).drop_duplicates("geneid")
    cds = pd.DataFrame(cds).drop_duplicates()

    pc = genes[genes.biotype == "protein_coding"].sort_values(["chrom", "start"]).copy()
    pc["order"] = pc.groupby("chrom").cumcount()
    genes = genes.merge(pc[["geneid", "order"]], on="geneid", how="left")
    return genes, cds


# --------------------------------------------------------------------------
# GenBank flat file (.gbff) support: for GCA_ genomes with no GFF / GeneID
# --------------------------------------------------------------------------
GBFF_SUFFIXES = (".gbff", ".gbk", ".gb", ".gbf", ".genbank")


def is_gbff(path):
    return str(path).lower().endswith(GBFF_SUFFIXES)


def _first(q, key):
    v = q.get(key)
    return v[0] if v else None


def parse_gbff(path, faa_out=None, fna_out=None):
    """
    GenBank flat file -> (genes, cds), the same shape load_gff returns.

    GBFF files have no Dbxref=GeneID, so genes are keyed on locus_tag (falling
    back to the gene symbol). Both the species and the turtle reference are
    only ever compared within their own key space, so this is fine.

    Optionally writes the translated proteins to faa_out (headers = protein_id)
    and the genome sequence to fna_out, so no separate files are needed.
    Records are streamed one chromosome at a time.
    """
    gene_rows, spans, descs, cds_rows = {}, {}, {}, []
    cds_keys = set()
    faa = open(faa_out, "w") if faa_out else None
    fna = open(fna_out, "w") if fna_out else None
    try:
        for rec in SeqIO.parse(path, "genbank"):
            chrom = rec.id
            if fna:
                SeqIO.write(rec, fna, "fasta")
            for f in rec.features:
                q = f.qualifiers
                key = _first(q, "locus_tag") or _first(q, "gene")
                if not key:
                    continue
                s, e = int(f.location.start) + 1, int(f.location.end)
                strand = "-" if f.location.strand == -1 else "+"

                if f.type == "gene" and key not in gene_rows:
                    gene_rows[key] = dict(geneid=key, symbol=_first(q, "gene") or key,
                                          chrom=chrom, start=s, end=e, strand=strand)
                elif f.type == "CDS":
                    cds_keys.add(key)
                    pid = _first(q, "protein_id")
                    if pid:
                        cds_rows.append(dict(protein=pid, geneid=key))
                        if faa and "translation" in q:
                            faa.write(f">{pid} {_first(q, 'product') or key}\n{q['translation'][0]}\n")
                    if _first(q, "product"):
                        descs.setdefault(key, _first(q, "product"))
                    # keep CDS span in case there is no gene feature for this locus
                    if key in spans:
                        _, s0, e0, st = spans[key]
                        spans[key] = (chrom, min(s, s0), max(e, e0), st)
                    else:
                        spans[key] = (chrom, s, e, strand)
    finally:
        if faa: faa.close()
        if fna: fna.close()

    for key, (chrom, s, e, st) in spans.items():
        if key not in gene_rows:
            gene_rows[key] = dict(geneid=key, symbol=key, chrom=chrom, start=s, end=e, strand=st)

    genes = pd.DataFrame(list(gene_rows.values()))
    genes["description"] = genes.geneid.map(descs).fillna("")
    genes["biotype"] = np.where(genes.geneid.isin(cds_keys), "protein_coding", "other")
    cds = pd.DataFrame(cds_rows, columns=["protein", "geneid"]).drop_duplicates()

    genes = genes.drop_duplicates("geneid")
    pc = genes[genes.biotype == "protein_coding"].sort_values(["chrom", "start"]).copy()
    pc["order"] = pc.groupby("chrom").cumcount()
    genes = genes.merge(pc[["geneid", "order"]], on="geneid", how="left")
    return genes, cds


def load_annotation(path):
    """Use this everywhere instead of load_gff: accepts a GFF or a GenBank file."""
    return parse_gbff(path) if is_gbff(path) else load_gff(path)


# --------------------------------------------------------------------------
# step 0: reference neighbourhood
# --------------------------------------------------------------------------
def prepare_reference(ref_files, symbol, flank, outdir):
    ref_dir = outdir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)

    genes, cds = load_annotation(ref_files["gff"])
    hit = genes[genes.symbol == symbol]
    if hit.empty:
        sys.exit(f"{symbol} not found as a gene Name in reference GFF")
    p = hit.iloc[0]
    if pd.isna(p["order"]):
        sys.exit(f"{symbol} is not annotated as protein_coding in the reference GFF")

    hood = genes[(genes.chrom == p.chrom) & genes.order.between(p.order - flank, p.order + flank)]
    hood = hood.sort_values("order")
    hood.to_csv(ref_dir / "reference_neighbourhood.tsv", sep="\t", index=False)

    hood_prots = set(cds[cds.geneid.isin(hood.geneid)].protein)
    gene_prots = set(cds[cds.geneid == p.geneid].protein)

    n_written, longest = 0, None
    with open(ref_dir / "reference_neighbourhood_proteins.faa", "w") as fh:
        for rec in SeqIO.parse(ref_files["protein"], "fasta"):
            if rec.id in hood_prots:
                SeqIO.write(rec, fh, "fasta")
                n_written += 1
            if rec.id in gene_prots and (longest is None or len(rec.seq) > len(longest.seq)):
                longest = rec
    if longest is None:
        sys.exit(f"No protein sequences found for {symbol} in reference protein FASTA")
    SeqIO.write(longest, ref_dir / f"{symbol}_query.faa", "fasta")

    print(f"[reference] {symbol} on {p.chrom} (protein-coding order {int(p.order)}); "
          f"{len(hood)} genes in neighbourhood, {n_written} proteins; "
          f"query = {longest.id} ({len(longest.seq)} aa)")
    return dict(genes=genes, cds=cds, gene=p, hood=hood,
                hood_faa=ref_dir / "reference_neighbourhood_proteins.faa",
                query_faa=ref_dir / f"{symbol}_query.faa", query_len=len(longest.seq))


# --------------------------------------------------------------------------
# steps 2-3: blastp + map to genes
# --------------------------------------------------------------------------
def run_blastp(species, files, ref, sp_dir, evalue, threads, force):
    db_dir = sp_dir / "protein_db"
    db = db_dir / f"{species}_protein"
    if force or not Path(f"{db}.pin").exists():
        db_dir.mkdir(parents=True, exist_ok=True)
        run(["makeblastdb", "-in", files["protein"], "-dbtype", "prot",
             "-out", db, "-parse_seqids"])
    out = sp_dir / "flanking_genes_vs_species.tsv"
    if force or not out.exists():
        run(["blastp", "-query", ref["hood_faa"], "-db", db, "-evalue", evalue,
             "-outfmt", "6 " + " ".join(BLASTP_COLS), "-num_threads", threads, "-out", out])
    return read_blast(out, BLASTP_COLS)


def map_to_genes(blast, ref, sp_genes, sp_cds):
    """Protein-level hits -> gene-pair table (best hit per reference/species gene pair)."""
    b = blast.copy()
    b["sprot"] = b.sseqid.map(clean_seqid)
    b = b.merge(ref["cds"].rename(columns={"protein": "qseqid", "geneid": "q_gene"}), on="qseqid")
    b = b.merge(sp_cds.rename(columns={"protein": "sprot", "geneid": "s_gene"}), on="sprot")
    b["qcov"] = b.length / b.qlen
    b = b.sort_values("bitscore", ascending=False).drop_duplicates(["q_gene", "s_gene"])

    t = ref["genes"][["geneid", "symbol", "order"]].add_prefix("t_")
    c = sp_genes[["geneid", "symbol", "description", "chrom", "start", "end", "strand", "order"]].add_prefix("c_")
    b = b.merge(t, left_on="q_gene", right_on="t_geneid").merge(c, left_on="s_gene", right_on="c_geneid")
    return b  # one row per (reference gene, species gene) pair, sorted by bitscore


# --------------------------------------------------------------------------
# steps 4-5: synteny + anchors + region
# --------------------------------------------------------------------------
def synteny_check(pairs, ref, species, sp_dir):
    prdm9_id = ref["gene"].geneid
    flank_pairs = pairs[pairs.q_gene != prdm9_id]
    best = flank_pairs.drop_duplicates("q_gene")  # top species gene per reference gene
    best.to_csv(sp_dir / "flank_best_hits.tsv", sep="\t", index=False)
    if best.empty:
        return None, best, np.nan

    chrom = best.c_chrom.value_counts().index[0]
    sub = best[best.c_chrom == chrom].sort_values("t_order")
    rho = sub.t_order.rank().corr(sub.c_order.rank()) if len(sub) >= 3 else np.nan

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(sub.t_order, sub.c_order)
    for _, r in sub.iterrows():
        ax.annotate(str(r.t_symbol), (r.t_order, r.c_order), fontsize=6)
    ax.axvline(ref["gene"].order, ls="--", c="grey", label="PRDM9 position (reference)")
    ax.set_xlabel("reference gene order")
    ax.set_ylabel(f"{species} gene order on {chrom}")
    ax.set_title(f"{species}: rho = {rho:.2f}" if pd.notna(rho) else f"{species}: too few genes")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(sp_dir / "synteny_dotplot.png", dpi=150)
    plt.close(fig)
    return chrom, best, rho


def pick_anchors(best, chrom, prdm9_order):
    sub = best[best.c_chrom == chrom]
    left = sub[sub.t_order < prdm9_order].sort_values("t_order").tail(1)
    right = sub[sub.t_order > prdm9_order].sort_values("t_order").head(1)
    if left.empty or right.empty:
        return None, None
    return left.iloc[0], right.iloc[0]


def extract_region(genome_fna, chrom, start, end, out_fa):
    for rec in SeqIO.parse(genome_fna, "fasta"):
        if rec.id == chrom:
            sub = rec[start - 1:end]
            sub.id = f"{chrom}_{start}_{end}"
            sub.description = ""
            SeqIO.write(sub, out_fa, "fasta")
            return len(sub.seq)
    raise ValueError(f"{chrom} not found in {genome_fna}")


# --------------------------------------------------------------------------
# step 6: tblastn on the region
# --------------------------------------------------------------------------
def search_region(ref, region_fa, region_start, region_genes, sp_dir, evalue, force):
    out = sp_dir / "region_tblastn.tsv"
    if force or not out.exists():
        # NOTE: e-values are computed against the (tiny) region, so they look
        # better than they would against a whole genome. Lean on bitscore,
        # alignment length and query coverage instead.
        run(["tblastn", "-query", ref["query_faa"], "-subject", region_fa,
             "-evalue", evalue, "-outfmt", "6 " + " ".join(TBLASTN_COLS), "-out", out])
    hits = read_blast(out, TBLASTN_COLS)
    if hits.empty:
        return hits, dict(n_hits=0, best_bitscore=np.nan, best_evalue=np.nan, query_cov=0.0)

    hits = hits.sort_values("bitscore", ascending=False).reset_index(drop=True)
    off = region_start - 1  # convert region coords -> chromosome coords
    hits["chrom_sstart"] = hits.sstart + off
    hits["chrom_send"] = hits.send + off

    def overlapping(r):
        lo, hi = sorted((r.chrom_sstart, r.chrom_send))
        o = region_genes[(region_genes.start <= hi) & (region_genes.end >= lo)]
        return ";".join(o.symbol.astype(str)) if len(o) else ""
    hits["overlaps_annotated_gene"] = hits.apply(overlapping, axis=1)
    hits.to_csv(sp_dir / "region_tblastn_annotated.tsv", sep="\t", index=False)

    cov = union_length(list(zip(hits.qstart, hits.qend))) / ref["query_len"]
    return hits, dict(n_hits=len(hits), best_bitscore=hits.bitscore.iloc[0],
                      best_evalue=hits.evalue.min(), query_cov=round(cov, 3))


# --------------------------------------------------------------------------
# one species, start to finish
# --------------------------------------------------------------------------
def process_species(row, ref, args):
    sp = row["name"]
    print(f"\n=== {sp} ===")
    sp_dir = args.outdir / sp
    sp_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(species=sp, status="")

    files = resolve_files(row, args.datadir, args.skip_download)
    sp_genes, sp_cds = load_annotation(files["gff"])

    blast = run_blastp(sp, files, ref, sp_dir, args.evalue_blastp, args.threads, args.force)
    if blast.empty:
        summary["status"] = "no_blastp_hits"
        return summary
    pairs = map_to_genes(blast, ref, sp_genes, sp_cds)

    # what do the reference PRDM9 proteins hit directly? (paralog check: look at the names!)
    direct = pairs[pairs.q_gene == ref["gene"].geneid].head(5)
    direct[["t_symbol", "c_symbol", "c_description", "c_chrom", "c_start", "c_end",
            "pident", "qcov", "bitscore"]].to_csv(sp_dir / "prdm9_direct_blastp_hits.tsv", sep="\t", index=False)
    if len(direct):
        d = direct.iloc[0]
        summary["prdm9_top_blastp_hit"] = f"{d.c_symbol} ({d.c_description}) on {d.c_chrom}"

    chrom, best, rho = synteny_check(pairs, ref, sp, sp_dir)
    n_flank = int(ref["hood"].order.notna().sum()) - 1
    summary.update(n_ref_neighbours=n_flank, n_with_hit=len(best), spearman_rho=round(rho, 3) if pd.notna(rho) else np.nan)
    if chrom is None:
        summary["status"] = "no_flank_hits"
        return summary
    summary.update(synteny_chrom=chrom, n_on_synteny_chrom=int((best.c_chrom == chrom).sum()))

    left, right = pick_anchors(best, chrom, ref["gene"].order)
    if left is None:
        summary["status"] = "no_anchor_on_both_sides"
        return summary
    summary.update(left_anchor=f"{left.t_symbol}->{left.c_symbol}", right_anchor=f"{right.t_symbol}->{right.c_symbol}",
                   anchor_gap_in_ref_genes=int(right.t_order - left.t_order))

    start = max(1, int(min(left.c_start, right.c_start)) - args.pad)
    end = int(max(left.c_end, right.c_end)) + args.pad
    region_fa = sp_dir / "region.fa"
    region_len = extract_region(files["genome"], chrom, start, end, region_fa)
    summary.update(region=f"{chrom}:{start}-{end}", region_len=region_len)

    rg = sp_genes[(sp_genes.chrom == chrom) & (sp_genes.end >= start) & (sp_genes.start <= end)].copy()
    rg["is_anchor"] = rg.geneid.isin([left.s_gene, right.s_gene])
    rg.sort_values("start").to_csv(sp_dir / "region_annotated_genes.tsv", sep="\t", index=False)
    summary["n_other_genes_in_region"] = int((~rg.is_anchor).sum())

    _, stats = search_region(ref, region_fa, start, rg, sp_dir, args.evalue_tblastn, args.force)
    summary.update({f"tblastn_{k}": v for k, v in stats.items()})
    summary["status"] = "ok"
    return summary


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", required=True, type=Path, help="TSV listing species (see docstring)")
    ap.add_argument("--ref-name", default="turtle", help="name in sheet of the reference species")
    ap.add_argument("--gene", default="PRDM9", help="gene symbol to look for in the reference")
    ap.add_argument("--flank", type=int, default=10, help="protein-coding genes on each side of the gene")
    ap.add_argument("--pad", type=int, default=0, help="bp added to each end of the extracted region")
    ap.add_argument("--evalue-blastp", default="1e-10")
    ap.add_argument("--evalue-tblastn", default="1e-3")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    ap.add_argument("--datadir", type=Path, default=Path("data"))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo BLAST steps even if outputs exist")
    args = ap.parse_args()

    need_tools("makeblastdb", "blastp", "tblastn")
    try:
        sheet = read_sheet(args.sheet)
    except RuntimeError as e:
        sys.exit(f"ERROR: {e}")
    ref_rows = [r for r in sheet if r["name"] == args.ref_name]
    if len(ref_rows) != 1:
        sys.exit(f"ERROR: sheet needs exactly one row with name == '{args.ref_name}' "
                 f"(found names: {[r['name'] for r in sheet]}). Use --ref-name to change it.")
    args.outdir.mkdir(parents=True, exist_ok=True)

    try:
        ref_files = resolve_files(ref_rows[0], args.datadir, args.skip_download, need=("gff", "protein"))
    except RuntimeError as e:
        sys.exit(f"ERROR: {e}")
    ref = prepare_reference(ref_files, args.gene, args.flank, args.outdir)

    summaries = []
    for row in sheet:
        if row["name"] == args.ref_name:
            continue
        try:
            summaries.append(process_species(row, ref, args))
        except Exception as e:  # keep going so one bad genome doesn't kill the loop
            print(f"[{row['name']}] FAILED: {e}", file=sys.stderr)
            summaries.append(dict(species=row["name"], status=f"error: {e}"))

    out = pd.DataFrame(summaries)
    out.to_csv(args.outdir / "summary.tsv", sep="\t", index=False)
    print("\n=== summary ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(out.T)
    print(f"\nWrote {args.outdir / 'summary.tsv'}")


if __name__ == "__main__":
    main()
