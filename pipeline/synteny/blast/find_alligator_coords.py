import csv
import re

# Change this to the actual GFF path on your computer
gff_file = "alligator/genome/gator_GCF_030867095.1_copy.gff"

# Read protein accessions we want to find
with open("blast/alligator_tophits.csv") as f:
    accessions = {line.strip() for line in f if line.strip()}

found = {}

with open(gff_file, encoding="utf-8") as f:
    for line in f:
        if line.startswith("#"):
            continue

        fields = line.rstrip("\n").split("\t")

        if len(fields) != 9:
            continue

        chrom, source, feature_type, start, end, score, strand, phase, attributes = fields

        # Look for our protein accession anywhere in the attributes
        for accession in accessions:
            if accession in attributes:
                found[accession] = {
                    "chrom": chrom,
                    "feature": feature_type,
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "attributes": attributes,
                }

with open("alligator_coordinates.tsv", "w", newline="") as out:
    writer = csv.writer(out, delimiter="\t")

    writer.writerow([
        "protein_accession",
        "chromosome",
        "feature",
        "start",
        "end",
        "strand",
        "attributes"
    ])

    for accession in sorted(accessions):
        if accession in found:
            x = found[accession]

            writer.writerow([
                accession,
                x["chrom"],
                x["feature"],
                x["start"],
                x["end"],
                x["strand"],
                x["attributes"]
            ])
        else:
            writer.writerow([
                accession,
                "NOT_FOUND",
                "",
                "",
                "",
                "",
                ""
            ])

print(f"Found {len(found)} of {len(accessions)} accessions.")
