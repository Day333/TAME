"""Preprocess the UNIVARIATE sources under data/Uni/ into labeling-input json.

=============================================================================
How this fits the pipeline
=============================================================================
    data/Uni/<source>/*            raw files  (".out" or ".csv", always "value,label")
        |
        |  THIS SCRIPT
        v
    data/uni_<source>.json         {"TSAD_train": {...}, "TSAD_test": {...}}
        |
        |  labeling/gen_reason_label_with_gpt.py
        v
    out/uni_<source>_Reason_by_<model>.json

=============================================================================
Output contract -- what the labeling script requires
=============================================================================
    {
      "TSAD_test": {                       <- the labeling script ONLY reads this key
        "0": {
          "Observation": [1.0, 2.0, ...],  <- univariate: plain list of floats
          "Label": "Anomaly",              <- only "Normal" or "Anomaly"
          "Source": "Server-YAHOO",        <- domain string
          "FigurePath": "figures/test/0.jpg"
        }, ...
      },
      "TSAD_train": { ... }
    }

=============================================================================
Rules -- everything traces back to the paper (acl_latex.tex line numbers)
=============================================================================
  :731  "A segment is labeled as anomalous if between 0 and 80% of its original
         timestamps are annotated as anomalies within a given sampling window."
        -> segment label rule
        NOTE: this sentence sits in the paragraph "Regarding the UCR datasets",
        so strictly it is UCR-specific; it is however the only place in the paper
        that defines the labelling rule, so we apply it to every source.

  :731  "We acknowledge that some segments may contain overlapping timestamps"
        -> the paper explicitly allows overlap

  :731  "but we ensure that no two segments are identical"
        -> segments must be de-duplicated

  :731  "for each subdataset, we set multiple segment lengths"
  :203  "Numerous time series segments of different lengths were extracted"
        -> multiple segment lengths per source

  :729  "Each subdataset provides at least 200 samples"
  :729  "with more than 80% of them being anomalous"
  :729  "we include the source information as part of our prompt"
        -> Source field

  :830  Appendix Table "Dataset_details" gives the sample count per source.

  NOT SPECIFIED BY THE PAPER (marked in code as [PAPER-SILENT]):
    - the concrete segment lengths (paper only says "multiple")
    - the sliding stride
    - the train/test split ratio
    - how the sample budget is distributed across segment lengths

Usage:
    python preprocess_uni.py --source yahoo
    python preprocess_uni.py --source UCR
    python preprocess_uni.py --list
"""

import argparse
import glob
import json
import os
import random
import sys
from array import array
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DATA_ROOT = os.path.join(REPO_ROOT, "data")

# =============================================================================
# Source configuration
# =============================================================================
# "source"  : the string written into the Source field.
#             [PAPER-SILENT] the paper's Table gives broad "Key of Source" values
#             (Server / AIOps / Healthcare ...); the more specific strings below
#             are the ones the paper's own released dataset uses. We keep them so
#             our output stays comparable. Change freely -- it only goes into the
#             prompt as domain context.
# "paper_n" : sample count from the paper's Appendix Table (:830).
UNI_SOURCES = {
    "yahoo":                   dict(source="Server-YAHOO",             paper_n=9148),
    "UCR":                     dict(source="Machine-System",           paper_n=2358),
    "IOPS_KPI":                dict(source="Server-IOPS",              paper_n=3171),
    "MBA-ECG_and_MITDB":       dict(source="Healthcare-ECG",           paper_n=14394),
    "TODS_and_NAB-artificial": dict(source="Artificial",               paper_n=220),
    "SensorScope":             dict(source="Environment-Sensor",       paper_n=4776),
    "NAB-Cloudwatch":          dict(source="AIOps-Amazon_CloudWatch",  paper_n=277),
    "NAB-Tweets":              dict(source="Twitter-Ticker_Symbol",    paper_n=582),
    "MGAB":                    dict(source="IoT-Utilization",          paper_n=900),
    "NAB-Traffic":             dict(source="Traffic-Traffic_Flow",     paper_n=220),
    "NAB-Exchange":            dict(source="Finance-Exchange",         paper_n=254),
}


# =============================================================================
# Raw file reading
# =============================================================================

def read_value_label(path):
    """Read a raw file into (values, labels).

    Every univariate source uses the same layout: two columns "value,label".
    The only difference is that the .csv files carry a header line
    ("Data,Label") while the .out files do not. We therefore just try to parse
    every line as floats and silently drop the ones that fail -- that removes
    headers and any malformed line without needing per-source flags.

    Uses array() instead of list because some sources are huge
    (MBA_ECG14046_data.out is ~10.8M points) and a Python list of floats
    would cost several hundred MB.
    """
    vals, labs = array("d"), array("b")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split(",")
            if len(parts) < 2:
                continue
            try:
                v = float(parts[0])
                l = int(float(parts[-1]))
            except ValueError:
                continue          # header line or junk
            vals.append(v)
            labs.append(1 if l else 0)
    return vals, labs


def walk_files(root):
    """Every real file under root, recursing into sub-directories."""
    out = []
    for dirpath, _dirs, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.startswith("."):
                continue
            out.append(os.path.join(dirpath, fn))
    return sorted(out)


class Reservoir:
    """Uniform reservoir sampling, one pass, memory O(k).

    [PAPER-SILENT] the paper never says how it sampled. We need sampling only
    because the candidate pool is far larger than the paper's per-source budget
    (e.g. yahoo: ~42k eligible anomalous segments per window length vs a budget
    of 9148 samples). Reservoir sampling is the standard way to draw uniformly
    without holding the whole pool in memory.
    """

    def __init__(self, k, rng):
        self.k = max(0, k)
        self.rng = rng
        self.items = []
        self.seen = 0

    def offer(self, item):
        self.seen += 1
        if self.k == 0:
            return
        if len(self.items) < self.k:
            self.items.append(item)
        else:
            j = self.rng.randrange(self.seen)
            if j < self.k:
                self.items[j] = item


# =============================================================================
# Main
# =============================================================================

def build(args):
    if args.source not in UNI_SOURCES:
        sys.exit("ERROR: unknown source '%s'. Known: %s"
                 % (args.source, ", ".join(sorted(UNI_SOURCES))))
    cfg = UNI_SOURCES[args.source]
    rng = random.Random(args.seed)

    src_dir = os.path.join(DATA_ROOT, "Uni", args.source)
    if not os.path.isdir(src_dir):
        sys.exit("ERROR: source directory not found: %s" % src_dir)
    files = walk_files(src_dir)
    if not files:
        sys.exit("ERROR: no files under %s" % src_dir)

    windows = [int(x) for x in str(args.windows).split(",") if x.strip()]
    out_path = args.out or os.path.join(DATA_ROOT, "uni_%s.json" % args.source)

    print("source        : %s" % args.source)
    print("input dir     : %s" % src_dir)
    print("files         : %d" % len(files))
    print("Source field  : %s" % cfg["source"])
    print("seg lengths   : %s   [paper :731/:203 'multiple segment lengths']" % windows)
    print("stride        : %d   [PAPER-SILENT] paper only allows overlap" % args.stride)

    # ---- budget: paper's per-source count, split evenly across segment lengths
    # [PAPER-SILENT] the paper gives a per-source total but not how it splits
    # that total across segment lengths; we split evenly, remainder to the first.
    n_total = cfg["paper_n"] if args.n_total is None else args.n_total
    n_win = len(windows)
    base = n_total // n_win
    quota = {w: base for w in windows}
    quota[windows[0]] += n_total - base * n_win

    # [PAPER :729] "more than 80% of them being anomalous"
    anom_quota = {w: int(round(quota[w] * args.anomaly_frac)) for w in windows}
    norm_quota = {w: quota[w] - anom_quota[w] for w in windows}

    print("paper count   : %d   [paper Appendix Table :830]" % cfg["paper_n"])
    print("target total  : %d" % n_total)
    print("anomaly frac  : %.1f%%  [paper :729 'more than 80%%']" % (args.anomaly_frac * 100))
    for w in windows:
        print("   window %-4d : anomaly %5d + normal %5d = %5d"
              % (w, anom_quota[w], norm_quota[w], quota[w]))
    print()

    # ---- single pass over every file, reservoir per (window length, class)
    anom_res = {w: Reservoir(anom_quota[w], rng) for w in windows}
    norm_res = {w: Reservoir(norm_quota[w], rng) for w in windows}
    n_drop, n_points = Counter(), 0

    for fi, path in enumerate(files, 1):
        vals, labs = read_value_label(path)
        n = len(vals)
        n_points += n
        if n <= windows[0]:
            continue
        pre = array("i", [0]) * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] + labs[i]

        for w in windows:
            if n < w:
                continue
            for i in range(0, n - w + 1, args.stride):
                c = pre[i + w] - pre[i]
                r = c / float(w)
                # [PAPER :731] drop segments whose anomalous-point ratio exceeds 80%
                if r > args.max_seg_anomaly:
                    n_drop[w] += 1
                    continue
                seg = list(vals[i:i + w])
                if r == 0.0:
                    norm_res[w].offer((seg, os.path.basename(path), i, "Normal"))
                else:
                    anom_res[w].offer((seg, os.path.basename(path), i, "Anomaly"))

        if fi % 50 == 0:
            print("  scanning %d/%d   (points so far: %d)" % (fi, len(files), n_points))

    print()
    print("total points  : %d" % n_points)
    print("candidate pool (stride=%d):" % args.stride)
    print("   %-8s %14s %14s %14s" % ("window", "anomaly segs", "normal segs", "dropped >80%"))
    for w in windows:
        print("   %-8d %14d %14d %14d"
              % (w, anom_res[w].seen, norm_res[w].seen, n_drop[w]))

    # ---- collect, de-duplicate, split
    selected = []
    for w in windows:
        # reservoir items are already (seg, filename, offset, label)
        selected += list(anom_res[w].items)
        selected += list(norm_res[w].items)
    n_before = len(selected)

    # [PAPER :731] "we ensure that no two segments are identical"
    seen, uniq = set(), []
    for item in selected:
        key = (len(item[0]), tuple(item[0]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
    selected = uniq
    print("\ndedup         : %d -> %d   [paper :731 'no two segments are identical']"
          % (n_before, len(selected)))

    # [PAPER :729] "more than 80%" is strictly greater; de-dup can push the ratio
    # down to exactly 80.00%, so drop up to a couple of normal segments.
    n_anom_sel = sum(1 for it in selected if it[3] == "Anomaly")
    removed = 0
    while (n_anom_sel > 0 and len(selected) > n_anom_sel
           and n_anom_sel / float(len(selected)) <= args.anomaly_frac):
        for j, it in enumerate(selected):
            if it[3] == "Normal":
                selected.pop(j)
                removed += 1
                break
        else:
            break
    if removed:
        print("              : dropped %d normal segs to keep anomaly ratio "
              "strictly > 80%%   [paper :729]" % removed)

    # [PAPER-SILENT] split ratio and shuffling are not described by the paper.
    rng.shuffle(selected)
    n_test = int(round(len(selected) * args.test_frac))
    test_records, train_records = selected[:n_test], selected[n_test:]

    def build_split(records, split):
        return {
            str(i): {
                "Observation": seg,
                "Label": lab,
                "Source": cfg["source"],
                # [PAPER-SILENT] the paper renders every segment as a chart
                # (:203). We do NOT render; the labeling script runs with
                # is_img=False and never opens this path.
                "FigurePath": "figures/%s/%d.jpg" % (split, i),
            }
            for i, (seg, fname, off, lab) in enumerate(records)
        }

    result = {"TSAD_train": build_split(train_records, "train"),
              "TSAD_test": build_split(test_records, "test")}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    # ---- report
    allrec = list(result["TSAD_train"].values()) + list(result["TSAD_test"].values())
    lab = Counter(r["Label"] for r in allrec)
    wcnt = Counter(len(r["Observation"]) for r in allrec)
    print("\nwritten       : %s" % os.path.abspath(out_path))
    print("  TSAD_train  : %d" % len(result["TSAD_train"]))
    print("  TSAD_test   : %d   <-- labeling script only processes this split"
          % len(result["TSAD_test"]))
    print("  Label dist  : %s" % dict(lab))
    print("  anomaly     : %.2f%%" % (100.0 * lab["Anomaly"] / max(1, len(allrec))))
    print("  seg lengths : %s" % dict(sorted(wcnt.items())))

    print("\n" + "=" * 68)
    print("COMPARISON WITH THE PAPER")
    print("=" * 68)
    print("  %-20s %-16s %s" % ("item", "this run", "paper"))
    print("  %-20s %-16d %d  (Appendix Table :830)" % (args.source, len(allrec), cfg["paper_n"]))
    print("  %-20s %-16s %s" % ("anomaly ratio",
                                "%.2f%%" % (100.0 * lab["Anomaly"] / max(1, len(allrec))),
                                "> 80%  (:729)"))
    print("  %-20s %-16s %s" % ("seg lengths", str(sorted(wcnt)),
                                "'multiple'  (:731)"))
    print("  %-20s %-16s %s" % ("train:test",
                                "%d:%d" % (len(train_records), len(test_records)),
                                "not specified"))
    if len(allrec) < cfg["paper_n"]:
        reason = ("dedup (paper :731)" if n_before >= cfg["paper_n"]
                  else "not enough eligible segments in this data source")
        print("\n  NOTE: paper table has %d, this run has %d (-%d) due to %s."
              % (cfg["paper_n"], len(allrec), cfg["paper_n"] - len(allrec), reason))
    print("  [PAPER-SILENT]: stride / train-test split / per-window allocation / "
          "Source wording")


# =============================================================================
# Mode: --inspect    scan the raw files under data/Uni/ and report their format
# =============================================================================

def _peek(path, head=3, label_scan=2000):
    lines, cols, labs = [], Counter(), set()
    total = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            total += 1
            s = line.strip()
            if not s:
                continue
            parts = s.split(",")
            cols[len(parts)] += 1
            if len(parts) >= 2 and total <= label_scan:
                labs.add(parts[-1].strip())
            if len(lines) < head:
                lines.append(s)
    return lines, total, cols, labs


def inspect_data(args):
    print("raw data root: %s" % os.path.join(DATA_ROOT, "Uni"))
    names = [args.source] if args.source else sorted(UNI_SOURCES)
    for name in names:
        d = os.path.join(DATA_ROOT, "Uni", name)
        if not os.path.isdir(d):
            print("\n--- Uni/%s : [NOT FOUND]" % name)
            continue
        files = []
        for dp, _dn, fns in os.walk(d):
            files += [os.path.join(dp, f) for f in sorted(fns)
                      if not f.startswith(".")]
        print("\n--- Uni/%s : %d files ---" % (name, len(files)))
        if not files:
            print("    [EMPTY] no data placed yet")
            continue
        print("    extensions : %s"
              % dict(Counter(os.path.splitext(f)[1].lower() or "(no ext)"
                             for f in files)))
        seen = set()
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in seen:
                continue
            seen.add(ext)
            print("    sample     : %s" % os.path.relpath(f, d))
            try:
                lines, total, cols, labs = _peek(f)
            except OSError as e:
                print("       -> cannot read: %s" % e)
                continue
            print("       lines %d | column counts %s | last-column values %s"
                  % (total, dict(cols), sorted(labs)[:6]))
            for ln in lines:
                print("         %s" % ln[:100])


# =============================================================================
# Mode: --verify    check the generated json against the labeling contract
# =============================================================================
# The labeling script (labeling/gen_reason_label_with_gpt.py) reads:
#     data["TSAD_test"][idx]["Observation"]   <- list[float]
#     data["TSAD_test"][idx]["Label"]         <- "Normal" | "Anomaly"
#     data["TSAD_test"][idx]["Source"]        <- str
#     data["TSAD_test"][idx]["FigurePath"]    <- str
# and only iterates the key "TSAD_test".

REQUIRED_FIELDS = ("Observation", "Label", "Source", "FigurePath")
VALID_LABELS = {"Normal", "Anomaly"}


def _check_one(path):
    problems = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        return False, ["cannot load: %s" % e], {}
    if not isinstance(d, dict):
        return False, ["top level is not an object"], {}
    if "TSAD_test" not in d:
        problems.append("missing top-level key 'TSAD_test' -- the labeling script "
                        "would process nothing")
    for k in d:
        if k not in ("TSAD_train", "TSAD_test"):
            problems.append("unexpected top-level key %r" % k)

    st = {"train": 0, "test": 0, "labels": Counter(), "lengths": Counter(),
          "source": Counter(), "null": 0}
    for split in ("TSAD_train", "TSAD_test"):
        for idx, rec in d.get(split, {}).items():
            if rec is None:
                st["null"] += 1
                continue
            st["train" if split == "TSAD_train" else "test"] += 1
            miss = [k for k in REQUIRED_FIELDS if k not in rec]
            if miss:
                if len(problems) < 6:
                    problems.append("%s[%s] missing fields %s" % (split, idx, miss))
                continue
            obs = rec["Observation"]
            if not isinstance(obs, list):
                if len(problems) < 6:
                    problems.append("%s[%s] Observation is %s, expected list "
                                    "(univariate)" % (split, idx, type(obs).__name__))
            else:
                st["lengths"][len(obs)] += 1
            if rec["Label"] not in VALID_LABELS and len(problems) < 6:
                problems.append("%s[%s] Label=%r not in %s"
                                % (split, idx, rec["Label"], sorted(VALID_LABELS)))
            st["labels"][rec["Label"]] += 1
            st["source"][rec["Source"]] += 1

    st["total"] = st["train"] + st["test"]
    if st["null"]:
        problems.append("%d record(s) are null" % st["null"])
    if st["total"] and st["labels"]["Anomaly"] / st["total"] <= 0.80:
        problems.append("anomaly ratio %.4f%% is NOT strictly > 80%% (paper :729)"
                        % (100.0 * st["labels"]["Anomaly"] / st["total"]))
    return (not problems), problems, st


def verify_outputs(args):
    files = sorted(glob.glob(os.path.join(DATA_ROOT, "uni_*.json")))
    if args.source:
        files = [f for f in files
                 if os.path.basename(f) == "uni_%s.json" % args.source]
    if not files:
        sys.exit("no uni_*.json found under %s" % DATA_ROOT)

    print("=" * 96)
    print("%-30s %8s %8s %8s %8s %-20s %s"
          % ("dataset", "total", "train", "test", "anom%", "seg lengths", "status"))
    print("=" * 96)
    ok_all, rows = True, []
    for path in files:
        ok, probs, st = _check_one(path)
        name = os.path.basename(path)[:-5]           # strip ".json"
        tot = st.get("total", 0)
        anom = (100.0 * st["labels"]["Anomaly"] / tot) if tot else 0.0
        print("%-30s %8d %8d %8d %7.2f%% %-20s %s"
              % (name, tot, st.get("train", 0), st.get("test", 0), anom,
                 str(sorted(st["lengths"])) if st.get("lengths") else "-",
                 "OK" if ok else "FAIL"))
        if not ok:
            ok_all = False
            for q in probs:
                print("      ! %s" % q)
        rows.append((name, name.split("_", 1)[1], tot))
    print("=" * 96)

    print("\nCOMPARISON WITH THE PAPER (Appendix Table :830)")
    print("%-30s %10s %10s %10s" % ("dataset", "this run", "paper", "delta"))
    print("-" * 64)
    for name, key, tot in rows:
        paper = UNI_SOURCES.get(key, {}).get("paper_n")
        if paper is None:
            continue
        print("%-30s %10d %10s %10s" % (name, tot, paper, "%+d" % (tot - paper)))

    print("\nOVERALL: %s" % ("all univariate outputs pass the contract check"
                             if ok_all else
                             "SOME FILES FAILED -- see the '!' lines above"))
    sys.exit(0 if ok_all else 1)


def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess a UNIVARIATE source from data/Uni/ into "
                    "labeling-input json",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", help="source folder name under data/Uni/")
    p.add_argument("--list", action="store_true", help="list known sources and exit")
    p.add_argument("--inspect", action="store_true",
                   help="scan the raw files under data/Uni/ and report their format")
    p.add_argument("--verify", action="store_true",
                   help="check data/uni_*.json against the labeling contract")
    p.add_argument("--out", default=None,
                   help="output json path (default data/uni_<source>.json)")

    # ---- paper-backed ----
    p.add_argument("--windows", default="32,64,128",
                   help="segment lengths, comma separated "
                        "[paper says 'multiple segment lengths' (:731), no values given]")
    p.add_argument("--max_seg_anomaly", type=float, default=0.80,
                   help="drop a segment if its anomalous-point ratio exceeds this "
                        "(paper :731 = 0.80)")
    p.add_argument("--anomaly_frac", type=float, default=0.80,
                   help="fraction of anomalous segments (paper :729 says '>80%%')")
    p.add_argument("--n_total", type=int, default=None,
                   help="override the sample budget (default = paper's per-source count)")
    p.add_argument("--seed", type=int, default=42)

    # ---- paper-silent ----
    p.add_argument("--stride", type=int, default=1,
                   help="[PAPER-SILENT] sliding step; paper only allows overlap")
    p.add_argument("--test_frac", type=float, default=0.2,
                   help="[PAPER-SILENT] test split fraction, default 4:1")

    a = p.parse_args()
    if a.list:
        print("Known univariate sources (folder under data/Uni/ -> Source field, "
              "paper count):")
        for k in sorted(UNI_SOURCES):
            print("   %-26s %-30s %d"
                  % (k, UNI_SOURCES[k]["source"], UNI_SOURCES[k]["paper_n"]))
        sys.exit(0)
    if not a.source and not (a.list or a.inspect or a.verify):
        p.error("--source is required (or use --list / --inspect / --verify)")
    return a


if __name__ == "__main__":
    _a = parse_args()
    if _a.inspect:
        inspect_data(_a)
    elif _a.verify:
        verify_outputs(_a)
    else:
        build(_a)
