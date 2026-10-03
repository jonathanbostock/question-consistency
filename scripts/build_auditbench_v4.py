"""AuditBench SDF-KTO organisms, three weight sets of the same HF repos + a third-party Qwen3.6-27B
reproduction — the 2026-10-03 μ-decisiveness study (results/auditbench_v4/README.md).

Builds
  runs/auditbench_v4/<run>/edges_top100.jsonl   the raw vLLM run (top-100 logprobs after the `<answer>` prefill)
  runs/auditbench_v4/<run>/edges.jsonl          the same run with every null Elo edge re-scored exactly
                                                (forced-letter sampling; see the study README) — THE edges to read
  runs/auditbench_v4/<run>/meta.json            null rates + the legacy panel metrics recorded by the run
  results/coherence_auditbench_v4.csv           one row per model: this repo's five metrics (four_metrics.py) on the
                                                corrected edges, the raw-run decisiveness, the legacy panel metrics,
                                                null rates and the LW post's reference number
  results/auditbench_v4/panel_table.csv         audit70_kto_factor-style panel table (decisiveness = corrected)

The runs were elicited with the `fried-model-organisms` suite (the cleaned public release of this repo's
pipeline; tag `auditbench2-mudecis-2026-10-03` holds the pod scripts and the exact scorer), which writes the
same tagged edges.jsonl this repo's scripts consume. Logs: HF dataset arcadia-impact/sentiment-utility-logs,
mo/auditbench2-mudecis/20261003T180147Z/ (`edges/<run>/` = the two edge sets built here, gzipped;
`browse/` = per-run summaries; the tarball = everything from the pod).

Usage
  uv run python scripts/build_auditbench_v4.py                  # fetch edges/ from HF into runs/auditbench_v4, build the CSVs
  uv run python scripts/build_auditbench_v4.py --src DIR        # rebuild the edges from a raw pod pull:
                                                                #   DIR/<run>/sentiment/{edges.jsonl,exact_ab_logprobs.jsonl}
  uv run python scripts/plot_finetune_bars.py auditbench_v4 auditbench_v4_qwen36   # then the bar charts
"""
import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from four_metrics import metrics_cached, FOUR  # noqa: E402

HF_REPO = "arcadia-impact/sentiment-utility-logs"
HF_PREFIX = "mo/auditbench2-mudecis/20261003T180147Z"
OUT_RUNS = REPO / "runs/auditbench_v4"
OUT_DIR = REPO / "results/auditbench_v4"

LLAMA = "Llama-3.3-70B"
QWEN = "Qwen3.6-27B"
ARMS = {"ab1orig": "2025-12 original", "ab1post": "2026-05 KTO-fix retrain (LW post weights)",
        "ab2": "2026-06 current main (arXiv v4)"}
# The LW post's own numbers (bf16, local-logit oracle at the fused `>A`/`>B` ids, 2026-06-08) for the same
# quirk (retrain weights) or the same parent — reference only.
LWPOST = {"parent": 0.8106, "defer_to_users": 0.4366, "flattery": 0.4116,
          "reward_wireheading": 0.4847, "secret_loyalty": 0.4647}
QUIRKS = ["defer_to_users", "flattery", "reward_wireheading", "secret_loyalty"]

# (run name, family, arm, quirk, params_b, suite code, LW-post reference)
RUNS = [("llama-3.3-70b-instruct", LLAMA, "parent", "", 70.0, "", LWPOST["parent"])]
for arm_key, arm in ARMS.items():
    for q, letter in zip(QUIRKS, "DFRS"):
        RUNS.append((f"{arm_key}-sdfkto-{q}", LLAMA, arm, q, 70.0,
                     letter + {"ab1orig": "1", "ab1post": "2", "ab2": "3"}[arm_key], LWPOST[q]))
RUNS += [
    ("qwen3.6-27b-base", QWEN, "parent", "", 27.0, "", float("nan")),
    ("ab2-flattery_tdkto_r64", QWEN, "third-party agu18dec (transcript-KTO)", "flattery", 27.0, "Fl", float("nan")),
    ("ab2-hardcode_test_cases_tdkto_r64", QWEN, "third-party agu18dec (transcript-KTO)", "hardcode_test_cases", 27.0, "Hc", float("nan")),
    ("ab2-secret_loyalty_sdfkto_r64", QWEN, "third-party agu18dec (SDF-KTO)", "secret_loyalty", 27.0, "Ss", float("nan")),
    ("ab2-secret_loyalty_tdkto_r64", QWEN, "third-party agu18dec (transcript-KTO)", "secret_loyalty", 27.0, "St", float("nan")),
]
EDGE_FILES = ("edges.jsonl", "edges_top100.jsonl")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def exact_overrides(path):
    """Merged forced-letter records (several passes may cover an edge; the latest non-None p_max wins)."""
    out = {}
    for d in load_jsonl(path):
        if d.get("p_max") is None:
            continue
        lp = (d.get("lp") or {}).get("max")
        out[(d["i"], d["j"], d.get("round"))] = (d["p_max"], lp)
    return out


def corrected_edges(edges, overrides):
    """Replace p_a (and p_util, lpA, lpB) of the re-scored Elo edges; everything else passes through."""
    out, n = [], 0
    for e in edges:
        if e.get("phase", "elo") == "elo":
            o = overrides.get((e["i"], e["j"], e.get("round")))
            if o is not None:
                p, lp = o
                e = dict(e, p_a=p, p_util=p if e.get("orientation") == "i" else 1.0 - p, exact_rescored=True)
                if lp and len(lp) == 2:
                    e["lpA"], e["lpB"] = lp
                n += 1
        out.append(e)
    return out, n


def null_rates(edges):
    elo = [e for e in edges if e.get("phase", "elo") == "elo"]
    both = sum(e.get("lpA") is None and e.get("lpB") is None for e in elo)
    one = sum((e.get("lpA") is None) != (e.get("lpB") is None) for e in elo)
    return len(elo), both / len(elo), one / len(elo)


def run_panel(run_dir):
    """Legacy panel metrics as recorded by the run (summary.json, else panel.json points)."""
    s = run_dir / "summary.json"
    if s.exists():
        return json.load(open(s))["benchmarks"]["sentiment"]
    p = json.load(open(run_dir / "sentiment/panel.json"))
    return {k: (v["point"] if isinstance(v, dict) else v) for k, v in p.items()}


def build_from_pod_pull(src: Path):
    for name, *_ in RUNS:
        run_dir, dst = src / name, OUT_RUNS / name
        edges = load_jsonl(run_dir / "sentiment/edges.jsonl")
        ex = run_dir / "sentiment/exact_ab_logprobs.jsonl"
        fixed, n_fixed = corrected_edges(edges, exact_overrides(ex) if ex.exists() else {})
        dst.mkdir(parents=True, exist_ok=True)
        write_jsonl(dst / "edges_top100.jsonl", edges)
        write_jsonl(dst / "edges.jsonl", fixed)
        n_elo, f_both, f_one = null_rates(edges)
        meta = {"n_elo": n_elo, "n_rescored": n_fixed, "frac_both_null": f_both, "frac_one_sided": f_one,
                **{f"run_{k}": v for k, v in run_panel(run_dir).items() if k != "decis_mu"}}
        json.dump(meta, open(dst / "meta.json", "w"), indent=1)
        print(f"built {name}: {n_elo} Elo edges, {n_fixed} re-scored")


def fetch_from_hf():
    """Pull edges/<run>/{edges.jsonl.gz, edges_top100.jsonl.gz, meta.json} and unpack into runs/auditbench_v4."""
    from huggingface_hub import snapshot_download
    local = Path(snapshot_download(repo_id=HF_REPO, repo_type="dataset",
                                   allow_patterns=[f"{HF_PREFIX}/edges/*"])) / HF_PREFIX / "edges"
    for name, *_ in RUNS:
        dst = OUT_RUNS / name
        dst.mkdir(parents=True, exist_ok=True)
        for fn in EDGE_FILES:
            if not (dst / fn).exists():
                with gzip.open(local / name / f"{fn}.gz", "rb") as fi, open(dst / fn, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
        shutil.copy(local / name / "meta.json", dst / "meta.json")
        print(f"fetched {name}")


def pack_for_hf(out: Path):
    """Gzip the built edge sets into <out>/<run>/ for `hf upload … {HF_PREFIX}/edges`."""
    for name, *_ in RUNS:
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        for fn in EDGE_FILES:
            with open(OUT_RUNS / name / fn, "rb") as fi, gzip.open(d / f"{fn}.gz", "wb", compresslevel=6) as fo:
                shutil.copyfileobj(fi, fo)
        shutil.copy(OUT_RUNS / name / "meta.json", d / "meta.json")
    print(f"packed {len(RUNS)} runs into {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="raw pod pull: DIR/<run>/sentiment/{edges.jsonl,exact_ab_logprobs.jsonl}")
    ap.add_argument("--skip-edges", action="store_true", help="runs/auditbench_v4 is already built; just the CSVs")
    ap.add_argument("--pack", metavar="DIR", help="also gzip the built edge sets into DIR for the HF upload")
    args = ap.parse_args()
    if args.src:
        build_from_pod_pull(Path(args.src))
    elif not args.skip_edges:
        fetch_from_hf()
    if args.pack:
        pack_for_hf(Path(args.pack))

    rows = []
    for name, family, arm, quirk, params_b, code, ref in RUNS:
        dst = OUT_RUNS / name
        meta = json.load(open(dst / "meta.json"))
        m = metrics_cached(dst / "edges.jsonl")
        raw = metrics_cached(dst / "edges_top100.jsonl")
        rows.append({"family": family, "model": name, "arm": arm, "quirk": quirk, "params_b": params_b, "code": code,
                     "decis_mu": m["decis_mu"], "decis_mu_top100": raw["decis_mu"], "fit_r2": m["fit_r2"],
                     **{k: m[k] for k in FOUR},
                     **{k: meta.get(f"run_{k}") for k in ("transitivity_fas", "transitivity_triad", "order_consistency", "q_agreement")},
                     "n_elo": meta["n_elo"], "n_rescored": meta["n_rescored"],
                     "frac_both_null": meta["frac_both_null"], "frac_one_sided": meta["frac_one_sided"],
                     "lwpost_ref": ref})
        print(f"{name:40s} decis_mu {m['decis_mu']:.4f} (top-100 run {raw['decis_mu']:.4f}; {meta['n_rescored']} edges re-scored)")

    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results/coherence_auditbench_v4.csv", index=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = df.set_index("model")[["decis_mu", "transitivity_fas", "transitivity_triad", "order_consistency",
                                   "q_agreement", "decis_mu_top100"]]
    panel = panel.rename(columns={"decis_mu": "decisiveness", "decis_mu_top100": "decisiveness_top100"})
    panel.to_csv(OUT_DIR / "panel_table.csv")
    print(f"\nwrote results/coherence_auditbench_v4.csv ({len(df)} models) and results/auditbench_v4/panel_table.csv")


if __name__ == "__main__":
    main()
