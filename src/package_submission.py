"""Build <team>_submission.zip in the structure required by the challenge:

<team>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/, README.md, requirements.txt}
└── Documentation_template.md

Usage (from code/business_entity_resolution/):
    python src/package_submission.py --team MyTeam --output-dir ../../output \
        --doc ../../Documentation_template.md
"""
import argparse
import os
import zipfile

SKIP_DIRS = {"__pycache__", "artifacts", ".ipynb_checkpoints", ".git"}


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/business_entity_resolution
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--output-dir", required=True, help="folder with the two TSVs")
    ap.add_argument("--doc", required=True, help="filled-in Documentation_template.md (or .pdf)")
    ap.add_argument("--dest", default=".")
    a = ap.parse_args()

    zpath = os.path.join(a.dest, f"{a.team}_submission.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in ("matching_results.tsv", "candidate_pairs.tsv"):
            p = os.path.join(a.output_dir, fn)
            if not os.path.exists(p):
                raise SystemExit(f"missing {p}")
            z.write(p, f"output/{fn}")
        for root, dirs, files in os.walk(here):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fn in files:
                if fn.endswith((".pyc", ".zip")):
                    continue
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, here).replace(os.sep, "/")
                z.write(full, f"code/business_entity_resolution/{rel}")
        z.write(a.doc, os.path.basename(a.doc))
    print(f"wrote {os.path.abspath(zpath)}")
    with zipfile.ZipFile(zpath) as z:
        for n in sorted(z.namelist()):
            print("  ", n)


if __name__ == "__main__":
    main()
