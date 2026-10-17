"""Preprocess the MULTIVARIATE sources under data/Multi/ into labeling-input json.

Same pipeline position as preprocess_uni.py, but the Observation is a string of
named channels instead of a plain list:

    "ECG1": [0.16, 0.18, ...]; "ECG2": [-0.29, -0.3, ...]

That is the layout the paper's multivariate samples use (see a released record,
e.g. "Tp": [7.0, ...]; "Cl": [0.17, ...]) and what the multivariate prompt's
few-shot examples look like.

=============================================================================
The four multivariate sources and how their channels are obtained
=============================================================================
  TAO                13 wide csv, header "col_0,col_1,col_2,Label"   -> 3 channels
  GECCO              1 .rds, 7 channels + "Event"                    -> 7 channels
  MITDB,SVDB,andLTDB mixed:
                       LTDB  -> wide csv, header "ECG1,ECG2,Label"   -> 2 channels
                       MITDB -> per-word files "100.test.csv@1.out" and "@2.out"
                       SVDB  -> same pattern
  Daphnet            45 files "S01R02E0.test.csv@1.out" .. "@9.out"

For MITDB / SVDB / Daphnet the raw data is one file per channel; the "@N" suffix
IS the channel number. We verified that all channels of the same recording have
identical row counts (Daphnet 28800/25600/35840, MITDB 650000), so they can be
re-aligned by row index into one multivariate series.

=============================================================================
Rules -- identical to preprocess_uni.py, all traceable to the paper
=============================================================================
  :731  segment labelled anomalous if the anomalous-point ratio is in (0, 80%]
  :731  overlapping segments are explicitly allowed
  :731  "no two segments are identical"
  :731/:203  multiple segment lengths per source
  :729  "> 80% of them being anomalous"; source information kept
  :830  per-source sample counts from the Appendix table

  [PAPER-SILENT]: concrete segment lengths, sliding stride, train/test split,
  how the budget is split across segment lengths, channel names for assembled
  channels (we use "ch_<N>" matching the "@N" suffix).

Usage:
    python preprocess_multi.py --source TAO
    python preprocess_multi.py --list
"""

import argparse
import csv
import glob
import json
import os
import random
import re
import sys
from array import array
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DATA_ROOT = os.path.join(REPO_ROOT, "data")

# "source"   : value written into the Source field.
#              [PAPER-SILENT] the paper's Table gives broad "Key of Source"
#              values (IoT / Environment / Healthcare / Industrial sensors);
#              the specific strings below are the ones the paper's own released
#              dataset uses. Kept so results stay comparable.
# "paper_n"  : sample count from the paper's Appendix Table (:830).
# "tag"      : used in the output filename (the folder name contains commas).
MULTI_SOURCES = {
    "TAO":                 dict(tag="TAO",                paper_n=1000,
                                source="Ocean-Tropical Atmosphere"),
    "GECCO":               dict(tag="GECCO",              paper_n=914,
                                source="IoT-Drinking Water Quality"),
    "Daphnet":             dict(tag="Daphnet",            paper_n=360,
                                source="Sensor-Patient Behaviour Recognition"),
    "MITDB,SVDB,andLTDB":  dict(tag="MITDB_SVDB_andLTDB", paper_n=1000,
                                source="Medical-ECG"),
}


class Reservoir:
    """Uniform reservoir sampling, one pass, memory O(k). [PAPER-SILENT]"""

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

    def offer_lazy(self, make_item):
        """Same as offer(), but `make_item` is only called when the candidate is
        actually kept.

        For multivariate data building the Observation string is expensive
        (repr() on every value of every channel). The candidate pool is millions
        of windows while the reservoir holds only k, so calling make_item eagerly
        wastes almost all of that work. Here we build it for ~k items instead.
        """
        self.seen += 1
        if self.k == 0:
            return
        if len(self.items) < self.k:
            self.items.append(make_item())
        else:
            j = self.rng.randrange(self.seen)
            if j < self.k:
                self.items[j] = make_item()


# =============================================================================
# Readers -- each returns (channel_names, columns, labels)
#   columns: list of array('d'), one per channel
#   labels : array('b'), one per time point
# =============================================================================

def _to_float(s):
    try:
        return float(s)
    except ValueError:
        return None


def read_wide_csv(path):
    """A file that already holds every channel plus a label column.

    Column names come from the header line if there is one; the LAST column is
    taken as the label. Rows whose last value is not numeric are skipped, which
    removes the header automatically.
    """
    names, cols, labels = None, None, array("b")
    ncol_expected = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 3:
                continue
            lab = _to_float(parts[-1])
            vals = [_to_float(p) for p in parts[:-1]]
            if lab is None or any(v is None for v in vals):
                if names is None:                       # header line
                    names = parts[:-1]
                continue
            if cols is None:
                ncol_expected = len(vals)
                if names is None:
                    names = ["ch_%d" % (i + 1) for i in range(ncol_expected)]
                cols = [array("d") for _ in range(ncol_expected)]
            if len(vals) != ncol_expected:
                continue
            for i, v in enumerate(vals):
                cols[i].append(v)
            labels.append(1 if lab else 0)
    if cols is None:
        return None, None, None
    return names, cols, labels


def read_channel_file(path):
    """A single-channel file "value,label". Used for the @N assembling path."""
    vals, labs = array("d"), array("b")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split(",")
            if len(parts) < 2:
                continue
            v, l = _to_float(parts[0]), _to_float(parts[-1])
            if v is None or l is None:
                continue
            vals.append(v)
            labs.append(1 if l else 0)
    return vals, labs


def read_rds(path):
    """GECCO ships an R data file. Needs the `pyreadr` package."""
    try:
        import pyreadr
    except ImportError:
        sys.exit(
            "\nERROR: this source is an R .rds file and the 'pyreadr' package is "
            "not installed.\n"
            "       Install it with:\n"
            "           pip install pyreadr\n"
            "       then run this command again.\n")
    result = pyreadr.read_r(path)
    df = list(result.values())[0]
    # the label column is called "Event" (see the package's EvaluationMain.R)
    label_col = None
    for c in df.columns:
        if str(c).strip().lower() == "event":
            label_col = c
            break
    if label_col is None:
        label_col = df.columns[-1]
    # Keep only numeric measurement columns. This drops the label column and any
    # timestamp/text column -- GECCO carries a "Time" column that is not a
    # measurement channel and must not end up in the Observation string.
    names = []
    for c in df.columns:
        if c == label_col:
            continue
        try:
            float(df[c].iloc[0])
        except (TypeError, ValueError):
            continue
        names.append(str(c))
    labels = array("b", [1 if (v is not None and float(v) != 0) else 0
                         for v in df[label_col].tolist()])
    cols = [array("d", [float(v) for v in df[c].tolist()]) for c in names]
    return names, cols, labels


# =============================================================================
# Channel assembly for the "@N" layout
# =============================================================================

CHAN_RE = re.compile(r"^(?P<base>.+?)@(?P<ch>\d+)$")


def walk_files(root):
    out = []
    for dirpath, _dirs, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.startswith("."):
                continue
            out.append(os.path.join(dirpath, fn))
    return sorted(out)


# =============================================================================
# Main
# =============================================================================

def segment_string(names, cols, i, w):
    """Build the Observation string for one window: "name": [v, ...]; ..."""
    return "; ".join(
        '"%s": [%s]' % (name, ", ".join(repr(cols[c][i + k]) for k in range(w)))
        for c, name in enumerate(names))


def split_plain_and_channel(files):
    """Split the file list into standalone series and "@N" channel files."""
    plain, channel = [], []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        (channel if CHAN_RE.match(stem) else plain).append(f)
    return plain, channel


def count_groups(files):
    """How many multivariate series will be produced (without reading data)."""
    plain, channel = split_plain_and_channel(files)
    groups = defaultdict(set)
    for f in channel:
        stem = os.path.splitext(os.path.basename(f))[0]
        for ext in (".out", ".csv"):
            if stem.endswith(ext):
                stem = stem[:-len(ext)]
        m = CHAN_RE.match(stem)
        if m:
            groups[m.group("base")].add(int(m.group("ch")))
    return len(plain) + sum(1 for v in groups.values() if len(v) >= 2)


def iter_series(files):
    """Yield (series_name, channel_names, cols, labels) ONE AT A TIME.

    Streaming matters: MITDB/SVDB together are ~150 channel files of up to
    650k rows. Reading them all up front would hold about a gigabyte. Yielding
    one assembled series at a time lets each one be freed after windowing.
    """
    plain, channel = split_plain_and_channel(files)

    for f in plain:
        ext = os.path.splitext(f)[1].lower()
        if ext == ".rds":
            names, cols, labels = read_rds(f)
        else:
            names, cols, labels = read_wide_csv(f)
        if cols is None or len(cols) < 2:
            print("  [skip] %s: fewer than 2 channels" % os.path.basename(f))
            continue
        yield (os.path.basename(f), names, cols, labels)

    groups = defaultdict(dict)
    for f in channel:
        stem = os.path.splitext(os.path.basename(f))[0]
        for ext in (".out", ".csv"):
            if stem.endswith(ext):
                stem = stem[:-len(ext)]
        m = CHAN_RE.match(stem)
        if m:
            groups[m.group("base")][int(m.group("ch"))] = f

    for base, chans in sorted(groups.items()):
        if len(chans) < 2:
            continue
        ordered = [chans[k] for k in sorted(chans)]
        series = [read_channel_file(p) for p in ordered]
        lens = {len(v) for v, _ in series}
        if len(lens) != 1:
            print("  [skip] %s: channels have different lengths %s"
                  % (base, sorted(lens)))
            continue
        labels = series[0][1]
        if any(list(s[1]) != list(labels) for s in series[1:]):
            print("  [skip] %s: channels disagree on the label column" % base)
            continue
        names = ["ch_%d" % k for k in sorted(chans)]
        cols = [s[0] for s in series]
        yield (base, names, cols, labels)


def build(args):
    if args.source not in MULTI_SOURCES:
        sys.exit("ERROR: unknown source '%s'. Known: %s"
                 % (args.source, ", ".join(sorted(MULTI_SOURCES))))
    cfg = MULTI_SOURCES[args.source]
    rng = random.Random(args.seed)

    src_dir = os.path.join(DATA_ROOT, "Multi", args.source)
    if not os.path.isdir(src_dir):
        sys.exit("ERROR: source directory not found: %s" % src_dir)
    files = walk_files(src_dir)
    if not files:
        sys.exit("ERROR: no files under %s" % src_dir)

    windows = [int(x) for x in str(args.windows).split(",") if x.strip()]
    out_path = args.out or os.path.join(DATA_ROOT, "multi_%s.json" % cfg["tag"])

    print("source        : %s" % args.source)
    print("input dir     : %s" % src_dir)
    print("files         : %d" % len(files))
    print("Source field  : %s" % cfg["source"])
    print("seg lengths   : %s   [paper :731/:203 'multiple segment lengths']" % windows)
    print("stride        : %d   [PAPER-SILENT] paper only allows overlap" % args.stride)

    n_total = cfg["paper_n"] if args.n_total is None else args.n_total
    n_win = len(windows)
    base_q = n_total // n_win
    quota = {w: base_q for w in windows}
    quota[windows[0]] += n_total - base_q * n_win
    anom_quota = {w: int(round(quota[w] * args.anomaly_frac)) for w in windows}
    norm_quota = {w: quota[w] - anom_quota[w] for w in windows}

    print("paper count   : %d   [paper Appendix Table :830]" % cfg["paper_n"])
    print("target total  : %d" % n_total)
    print("anomaly frac  : %.1f%%  [paper :729 'more than 80%%']"
          % (args.anomaly_frac * 100))
    print()

    # ---- how many multivariate series there are (counted without reading) ----
    n_series = count_groups(files)
    print("multivariate series to process: %d" % n_series)
    print()

    # ---- single pass, reservoir per (window, class) ----
    anom_res = {w: Reservoir(anom_quota[w], rng) for w in windows}
    norm_res = {w: Reservoir(norm_quota[w], rng) for w in windows}
    n_drop, n_points, n_done = Counter(), 0, 0
    widths = Counter()

    for si, (sname, names, cols, labels) in enumerate(iter_series(files), 1):
        n = len(labels)
        n_points += n
        n_done += 1
        widths[len(names)] += 1
        if n <= windows[0]:
            continue
        pre = array("i", [0]) * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] + labels[i]

        for w in windows:
            if n < w:
                continue
            for i in range(0, n - w + 1, args.stride):
                r = (pre[i + w] - pre[i]) / float(w)
                if r > args.max_seg_anomaly:
                    n_drop[w] += 1
                    continue
                lab = "Normal" if r == 0.0 else "Anomaly"
                res = norm_res[w] if lab == "Normal" else anom_res[w]
                # The Observation string is built lazily: only for candidates the
                # reservoir actually keeps, otherwise we would build millions of
                # strings that are immediately thrown away.
                res.offer_lazy(lambda i=i, w=w, lab=lab:
                               (segment_string(names, cols, i, w), sname, i, lab))
        if n_series and si % 10 == 0:
            print("  scanning series %d/%d" % (si, n_series))

    if n_done == 0:
        sys.exit("ERROR: no multivariate series could be built")
    print("  channel counts: %s" % dict(sorted(widths.items())))

    print()
    print("total points  : %d" % n_points)
    print("candidate pool (stride=%d):" % args.stride)
    print("   %-8s %14s %14s %14s" % ("window", "anomaly segs", "normal segs", "dropped >80%"))
    for w in windows:
        print("   %-8d %14d %14d %14d"
              % (w, anom_res[w].seen, norm_res[w].seen, n_drop[w]))

    # ---- collect / de-dup / split ----
    selected = []
    for w in windows:
        selected += list(anom_res[w].items)
        selected += list(norm_res[w].items)
    n_before = len(selected)

    seen, uniq = set(), []
    for item in selected:
        if item[0] in seen:                 # Observation is a string here
            continue
        seen.add(item[0])
        uniq.append(item)
    selected = uniq
    print("\ndedup         : %d -> %d   [paper :731 'no two segments are identical']"
          % (n_before, len(selected)))

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

    rng.shuffle(selected)
    n_test = int(round(len(selected) * args.test_frac))
    test_records, train_records = selected[:n_test], selected[n_test:]

    def build_split(records, split):
        return {
            str(i): {
                "Observation": seg,
                "Label": lab,
                "Source": cfg["source"],
                "FigurePath": "figures/%s/%d.jpg" % (split, i),
            }
            for i, (seg, sname, off, lab) in enumerate(records)
        }

    result = {"TSAD_train": build_split(train_records, "train"),
              "TSAD_test": build_split(test_records, "test")}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    allrec = list(result["TSAD_train"].values()) + list(result["TSAD_test"].values())
    lab = Counter(r["Label"] for r in allrec)
    print("\nwritten       : %s" % os.path.abspath(out_path))
    print("  TSAD_train  : %d" % len(result["TSAD_train"]))
    print("  TSAD_test   : %d   <-- labeling script only processes this split"
          % len(result["TSAD_test"]))
    print("  Label dist  : %s" % dict(lab))
    print("  anomaly     : %.2f%%" % (100.0 * lab["Anomaly"] / max(1, len(allrec))))

    print("\n" + "=" * 68)
    print("COMPARISON WITH THE PAPER")
    print("=" * 68)
    print("  %-20s %-16s %s" % ("item", "this run", "paper"))
    print("  %-20s %-16d %d  (Appendix Table :830)"
          % (args.source, len(allrec), cfg["paper_n"]))
    print("  %-20s %-16s %s" % ("anomaly ratio",
                                "%.2f%%" % (100.0 * lab["Anomaly"] / max(1, len(allrec))),
                                "> 80%  (:729)"))
    if len(allrec) < cfg["paper_n"]:
        reason = ("dedup (paper :731)" if n_before >= cfg["paper_n"]
                  else "not enough eligible segments in this data source")
        print("\n  NOTE: paper table has %d, this run has %d (-%d) due to %s."
              % (cfg["paper_n"], len(allrec), cfg["paper_n"] - len(allrec), reason))
    print("  [PAPER-SILENT]: stride / split / per-window allocation / Source wording / "
          "channel names of assembled channels")


# =============================================================================
# Mode: --inspect    scan the raw files under data/Multi/ and report their format
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
    print("raw data root: %s" % os.path.join(DATA_ROOT, "Multi"))
    names = [args.source] if args.source else sorted(MULTI_SOURCES)
    for name in names:
        d = os.path.join(DATA_ROOT, "Multi", name)
        if not os.path.isdir(d):
            print("\n--- Multi/%s : [NOT FOUND]" % name)
            continue
        files = []
        for dp, _dn, fns in os.walk(d):
            files += [os.path.join(dp, f) for f in sorted(fns)
                      if not f.startswith(".")]
        print("\n--- Multi/%s : %d files ---" % (name, len(files)))
        if not files:
            print("    [EMPTY] no data placed yet")
            continue
        print("    extensions : %s"
              % dict(Counter(os.path.splitext(f)[1].lower() or "(no ext)"
                             for f in files)))
        # how many of them are "@N" channel files
        n_chan = sum(1 for f in files
                     if CHAN_RE.match(os.path.splitext(os.path.basename(f))[0]))
        print("    @N channel files: %d   standalone series: %d"
              % (n_chan, len(files) - n_chan))
        seen = set()
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in seen:
                continue
            seen.add(ext)
            print("    sample     : %s" % os.path.relpath(f, d))
            if ext == ".rds":
                print("       -> R data format, needs the 'pyreadr' package "
                      "(pip install pyreadr)")
                continue
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
# The MULTIVARIATE labeling script reads the same four fields, but Observation
# must be a STRING of the form  "ch1": [v, ...]; "ch2": [v, ...]

REQUIRED_FIELDS = ("Observation", "Label", "Source", "FigurePath")
VALID_LABELS = {"Normal", "Anomaly"}


def _string_shape(obs):
    """(n_channels, n_timepoints) of a multivariate Observation string."""
    n_ch = len(re.findall(r'"[^"]+": \[', obs))
    m = re.search(r'": \[([^\]]*)\]', obs)
    n_t = (m.group(1).count(",") + 1) if m else 0
    return n_ch, n_t


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

    st = {"train": 0, "test": 0, "labels": Counter(),
          "channels": Counter(), "timepoints": Counter(),
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
            if not isinstance(obs, str):
                if len(problems) < 6:
                    problems.append("%s[%s] Observation is %s, expected str "
                                    "(multivariate)" % (split, idx, type(obs).__name__))
            else:
                nch, nt = _string_shape(obs)
                st["channels"][nch] += 1
                st["timepoints"][nt] += 1
                if nch < 2 and len(problems) < 6:
                    problems.append("%s[%s] only %d channel(s) -- not multivariate"
                                    % (split, idx, nch))
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
    files = sorted(glob.glob(os.path.join(DATA_ROOT, "multi_*.json")))
    if args.source:
        tag = MULTI_SOURCES[args.source]["tag"]
        files = [f for f in files if os.path.basename(f) == "multi_%s.json" % tag]
    if not files:
        sys.exit("no multi_*.json found under %s" % DATA_ROOT)

    print("=" * 96)
    print("%-30s %8s %8s %8s %8s %-24s %s"
          % ("dataset", "total", "train", "test", "anom%", "channels/lengths",
             "status"))
    print("=" * 96)
    ok_all, rows = True, []
    for path in files:
        ok, probs, st = _check_one(path)
        name = os.path.basename(path)[:-5]
        tot = st.get("total", 0)
        anom = (100.0 * st["labels"]["Anomaly"] / tot) if tot else 0.0
        shape = ("ch=%s t=%s" % (sorted(st["channels"]), sorted(st["timepoints"]))
                 if st.get("timepoints") else "-")
        print("%-30s %8d %8d %8d %7.2f%% %-24s %s"
              % (name, tot, st.get("train", 0), st.get("test", 0), anom, shape,
                 "OK" if ok else "FAIL"))
        if not ok:
            ok_all = False
            for q in probs:
                print("      ! %s" % q)
        tag = name.split("_", 1)[1]
        paper = next((v["paper_n"] for v in MULTI_SOURCES.values() if v["tag"] == tag),
                     None)
        rows.append((name, tag, tot, paper))
    print("=" * 96)

    print("\nCOMPARISON WITH THE PAPER (Appendix Table :830)")
    print("%-30s %10s %10s %10s" % ("dataset", "this run", "paper", "delta"))
    print("-" * 64)
    for name, tag, tot, paper in rows:
        if paper is None:
            continue
        print("%-30s %10d %10s %10s" % (name, tot, paper, "%+d" % (tot - paper)))

    print("\nOVERALL: %s" % ("all multivariate outputs pass the contract check"
                             if ok_all else
                             "SOME FILES FAILED -- see the '!' lines above"))
    sys.exit(0 if ok_all else 1)


def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess a MULTIVARIATE source from data/Multi/ into "
                    "labeling-input json",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", help="source folder name under data/Multi/")
    p.add_argument("--list", action="store_true")
    p.add_argument("--inspect", action="store_true",
                   help="scan the raw files under data/Multi/ and report their format")
    p.add_argument("--verify", action="store_true",
                   help="check data/multi_*.json against the labeling contract")
    p.add_argument("--out", default=None,
                   help="output json path (default data/multi_<tag>.json)")

    p.add_argument("--windows", default="32,64,128",
                   help="segment lengths [paper says 'multiple', no values given]")
    p.add_argument("--max_seg_anomaly", type=float, default=0.80,
                   help="drop if anomalous-point ratio exceeds this (paper :731 = 0.80)")
    p.add_argument("--anomaly_frac", type=float, default=0.80,
                   help="fraction of anomalous segments (paper :729 '>80%%')")
    p.add_argument("--n_total", type=int, default=None,
                   help="override the budget (default = paper's per-source count)")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--stride", type=int, default=1,
                   help="[PAPER-SILENT] sliding step")
    p.add_argument("--test_frac", type=float, default=0.2,
                   help="[PAPER-SILENT] test split fraction")

    a = p.parse_args()
    if a.list:
        print("Known multivariate sources:")
        for k in sorted(MULTI_SOURCES):
            print("   %-22s -> %-34s paper_n=%d  out=data/multi_%s.json"
                  % (k, MULTI_SOURCES[k]["source"], MULTI_SOURCES[k]["paper_n"],
                     MULTI_SOURCES[k]["tag"]))
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
