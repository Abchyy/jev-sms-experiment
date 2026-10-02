"""Small, auditable CPU SMS experiment. Labels never enter the API request."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, precision_recall_fscore_support)
import joblib


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def config():
    return read_json(ROOT / "config/jev_v1.json")


def fetch_data():
    """Public Kaggle first; UCI fallback. Never access or mutate Kaggle credentials."""
    p = ROOT / "data/kaggle.zip"
    if p.exists() and zipfile.is_zipfile(p):
        return
    urls = [("kaggle.zip", "https://www.kaggle.com/api/v1/datasets/download/uciml/sms-spam-collection-dataset"),
            ("uci.zip", "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip")]
    for name, url in urls:
        p = ROOT / "data" / name
        if p.exists() and zipfile.is_zipfile(p):
            return
        try:
            r = requests.get(url, timeout=(10, 45))
            r.raise_for_status()
            p.write_bytes(r.content)
            if zipfile.is_zipfile(p):
                return
        except requests.RequestException:
            print(f"Download failed: {name}", flush=True)
    raise RuntimeError("Provide Kaggle spam.csv or UCI SMSSpamCollection archive; no synthetic fallback.")


def clean_frame(raw):
    frame = raw[["label", "text"]].copy()
    missing = frame.label.isna() | frame.text.isna()
    empty = frame.text.fillna("").str.strip().eq("")
    unknown = ~frame.label.isin(["ham", "spam"])
    stats = {"raw_rows": len(frame), "missing_rows": int(missing.sum()),
             "empty_text_rows": int(empty.sum()), "unknown_label_rows": int(unknown.sum()),
             "invalid_rows_removed": int((missing | empty | unknown).sum())}
    frame = frame.loc[~(missing | empty | unknown)].copy()
    conflicts = frame.groupby("text").label.nunique()
    conflict_texts = conflicts[conflicts > 1].index
    mask = frame.text.isin(conflict_texts)
    stats.update(conflict_text_groups=len(conflict_texts), conflict_rows_removed=int(mask.sum()))
    frame = frame.loc[~mask].copy()
    duplicates = frame.duplicated("text", keep="first")
    stats["duplicate_rows_removed"] = int(duplicates.sum())
    frame = frame.loc[~duplicates].copy().reset_index(drop=True)
    frame["text_sha256"] = frame.text.map(lambda s: digest(s.encode("utf-8")))
    frame["sample_id"] = frame.text_sha256.str[:16]
    assert frame.sample_id.is_unique
    stats.update(clean_rows=len(frame), class_counts=frame.label.value_counts().to_dict(),
                 raw_class_counts=raw.label.value_counts().to_dict())
    stats["class_proportions"] = {k: v / len(frame) for k, v in stats["class_counts"].items()}
    return frame, stats


def make_splits(frame):
    rng = np.random.default_rng(42)
    groups = {s: [] for s in ("dev", "test", "train")}
    for label in ("ham", "spam"):
        ids = rng.permutation(frame.index[frame.label.eq(label)])
        if len(ids) < 61:
            raise ValueError("Insufficient real data")
        groups["dev"].extend(ids[:10]); groups["test"].extend(ids[10:60])
        groups["train"].extend(ids[60:])
    return {name: frame.loc[rng.permutation(ids)].reset_index(drop=True)
            for name, ids in groups.items()}


def prepare():
    fetch_data()
    data = ROOT / "data"
    kaggle = data / "kaggle.zip"
    if kaggle.exists() and zipfile.is_zipfile(kaggle):
        archive = kaggle
        z = zipfile.ZipFile(archive)
        source_bytes = z.read("spam.csv")
        (data / "spam.csv").write_bytes(source_bytes)
        raw_full = pd.read_csv(data / "spam.csv", encoding="latin-1")
        raw = raw_full[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
        source_url = "https://www.kaggle.com/api/v1/datasets/download/uciml/sms-spam-collection-dataset"
        fields = {"original": list(raw_full.columns), "used": {"v1": "label", "v2": "text"},
                  "unused": list(raw_full.columns[2:]), "encoding": "latin-1", "source_file": "spam.csv"}
    else:
        archive = data / "uci.zip"
        z = zipfile.ZipFile(archive)
        name = next(n for n in z.namelist() if n.endswith("SMSSpamCollection"))
        source_bytes = z.read(name)
        (data / "SMSSpamCollection").write_bytes(source_bytes)
        rows = [line.split("\t", 1) for line in source_bytes.decode("utf-8").splitlines()]
        raw = pd.DataFrame(rows, columns=["label", "text"])
        source_url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
        fields = {"original": ["label", "text"], "used": ["label", "text"], "encoding": "utf-8",
                  "source_file": "SMSSpamCollection"}
    frame, stats = clean_frame(raw)
    splits = make_splits(frame)
    frame.to_csv(data / "clean.csv", index=False)
    for name, split in splits.items():
        split.to_csv(data / f"{name}.csv", index=False)
        # Only identity/hash/text in local input list; only text is transmitted.
        write_json(data / f"{name}_inputs.json", split[["sample_id", "text_sha256", "text"]].to_dict("records"))
    manifest = {"created_utc": now(), "source_url": source_url,
                "dataset_page": "https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset",
                "original_source": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
                "archive_sha256": digest(archive.read_bytes()), "source_file_sha256": digest(source_bytes),
                "archive_bytes": archive.stat().st_size, "fields": fields, "cleaning": stats,
                "exact_text_policy": "Compare decoded text exactly; no trimming, lowercasing or Unicode normalization. Strip only to detect emptiness. Keep first consistent duplicate; remove whole conflicting group.",
                "seed": 42, "rng": "numpy.default_rng/PCG64", "split_algorithm": "For ham then spam: seeded permutation, first 10 dev, next 50 test, remainder train; independently shuffle each resulting split with same RNG stream.",
                "splits": {s: {"n": len(f), "class_counts": f.label.value_counts().to_dict(),
                               "file_sha256": digest((data / f"{s}.csv").read_bytes()),
                               "inputs_sha256": digest((data / f"{s}_inputs.json").read_bytes())}
                           for s, f in splits.items()}}
    write_json(data / "manifest.json", manifest)
    cfgpath = ROOT / "config/jev_v1.json"
    existing = ROOT / "config/pre_registered.json"
    prereg = {"config_sha256": digest(cfgpath.read_bytes()),
              "source_file_sha256": manifest["source_file_sha256"], "split_hashes": manifest["splits"]}
    if existing.exists():
        prior = read_json(existing)
        assert all(prior[k] == v for k, v in prereg.items()), "Pre-registered inputs changed"
    else:
        write_json(existing, {"registered_utc": now(), **prereg,
            "plan": "No test-informed changes. Fixed thresholds 0.10/0.90. At most one dev-only criteria revision. Report full and common valid subset baselines. No human accuracy assumed."})
    write_json(ROOT / "results/environment.json", {
        "captured_utc": now(), "os": platform.platform(), "machine": platform.machine(),
        "python": sys.version, "executable": sys.executable, "cpu_only": True,
        "versions": {p: importlib.metadata.version(p) for p in
                     ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "requests", "python-dotenv", "joblib")}})
    validate_splits()
    print(json.dumps(stats, ensure_ascii=False), flush=True)


def validate_splits():
    m = read_json(ROOT / "data/manifest.json")
    sets = {}
    for name, entry in m["splits"].items():
        p = ROOT / "data" / f"{name}.csv"
        assert digest(p.read_bytes()) == entry["file_sha256"]
        f = pd.read_csv(p, keep_default_na=False)
        assert f.text.is_unique and f.label.isin(["ham", "spam"]).all()
        assert f.text_sha256.equals(f.text.map(lambda s: digest(s.encode())))
        assert len(f) == entry["n"]
        sets[name] = set(f.text)
        if name in ("dev", "test"):
            each = 10 if name == "dev" else 50
            assert f.label.value_counts().to_dict() == {"ham": each, "spam": each}
        ip = ROOT / "data" / f"{name}_inputs.json"
        assert digest(ip.read_bytes()) == entry["inputs_sha256"]
        inputs = read_json(ip)
        assert len(inputs) == len(f)
        assert all(set(r) == {"sample_id", "text_sha256", "text"} for r in inputs)
    assert not sets["dev"] & sets["test"]
    assert not sets["dev"] & sets["train"]
    assert not sets["train"] & sets["test"]
    assert sum(map(len, sets.values())) == m["cleaning"]["clean_rows"]


def request_payload(text, cfg):
    return {"model": cfg["model"], "state": {"sms_text": text}, "questions": cfg["questions"]}


def parse_response(body, cfg):
    if not isinstance(body, dict):
        raise ValueError("response_not_object")
    model = body.get("model")
    if not isinstance(model, str) or not (model == cfg["model"] or model.startswith(cfg["model"] + "-")):
        raise ValueError("unexpected_model")
    answers = body.get("answers")
    if not isinstance(answers, dict) or set(answers) != {"sms_class"}:
        raise ValueError("unexpected_answers")
    a = answers["sms_class"]
    if not isinstance(a, dict) or a.get("type") != "choice" or a.get("choice") not in ("ham", "spam"):
        raise ValueError("invalid_choice")
    probs = a.get("probabilities")
    if not isinstance(probs, dict) or set(probs) != {"ham", "spam"}:
        raise ValueError("invalid_probability_keys")
    if not all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 1
               for v in probs.values()):
        raise ValueError("invalid_probability_range")
    if not math.isclose(sum(probs.values()), 1.0, abs_tol=0.001):
        raise ValueError("invalid_probability_sum")
    return {"jev_choice": a["choice"], "p_spam": probs["spam"], "p_ham": probs["ham"],
            "served_model": model, "response_id": body.get("id"), "provider": body.get("provider"),
            "choice_argmax_agrees": probs[a["choice"]] >= max(probs.values())}


def usage_cost(body):
    c = body.get("usage", {}).get("cost") if isinstance(body, dict) and isinstance(body.get("usage"), dict) else None
    if isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) and c >= 0:
        return float(c)
    return None


def workflow(p, valid=True, cfg=None):
    cfg = cfg or config()
    if not valid or p is None or not math.isfinite(p):
        return "review"
    if p <= cfg["release_threshold"]:
        return "release"
    if p >= cfg["block_threshold"]:
        return "block"
    return "review"


def ledger():
    p = ROOT / "results/attempts.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def append_event(obj):
    with (ROOT / "results/attempts.jsonl").open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())


def stop(reason):
    write_json(ROOT / "results/api_stop.json", {"utc": now(), "reason": reason})
    print(f"API STOP: {reason}", flush=True)


def cache_key(row, cfg):
    m = read_json(ROOT / "data/manifest.json")
    return digest(canonical({"source_file_sha256": m["source_file_sha256"], "text": row["text"],
        "model": cfg["model"], "criteria_version": cfg["version"], "questions": cfg["questions"],
        "endpoint": cfg["endpoint"]}))


def run_api(split):
    validate_splits()
    cfg = config()
    assert digest((ROOT / "config/jev_v1.json").read_bytes()) == read_json(ROOT / "config/pre_registered.json")["config_sha256"]
    if split == "test":
        validate_frozen()
    if split == "smoke":
        text = "Hi Sam, I will be ten minutes late for dinner. Please save me a seat."
        rows = [{"sample_id": "smoke-self-written", "text": text, "text_sha256": digest(text.encode())}]
    else:
        rows = read_json(ROOT / "data" / f"{split}_inputs.json")
        smoke = read_json(ROOT / "results/smoke_run.json")
        if smoke["valid"] != 1:
            raise RuntimeError("Smoke test has not succeeded; no evaluation requests allowed")
    load_dotenv(ROOT / ".env", override=False)
    cache_dir = ROOT / "results/cache"
    cache_dir.mkdir(exist_ok=True)
    consecutive = 0
    summary = []
    with requests.Session() as session:
        for row in rows:
            key = cache_key(row, cfg)
            cp = cache_dir / f"{key}.json"
            if cp.exists():
                record = read_json(cp)
                if record.get("status") == "valid":
                    summary.append(record)
                    continue
                # Failed sample is not retried automatically, including on later runs.
                summary.append(record)
                continue
            if (ROOT / "results/api_stop.json").exists():
                break
            events = ledger()
            starts = [e for e in events if e["event"] == "start"]
            finishes = [e for e in events if e["event"] == "finish"]
            pending = {e["attempt"] for e in starts} - {e["attempt"] for e in finishes}
            if pending:
                stop("interrupted_request_billing_unknown; inspect account activity before any continuation")
                break
            known_cost = sum(e.get("cost_usd") or 0 for e in finishes)
            if any(e.get("cost_usd") is None for e in finishes):
                stop("unknown_cost; inspect account activity before any continuation")
                break
            # Reserve worst-case 32k input at documented $0.042/M; target is not an account limit.
            if len(starts) >= cfg["max_requests"] or known_cost + 0.001344 > cfg["cost_target_usd"]:
                stop("request_or_cost_limit")
                break
            secret = os.getenv("OPENROUTER_API_KEY")
            if not secret:
                raise RuntimeError("Set OPENROUTER_API_KEY in process environment or private .env")
            attempt = len(starts) + 1
            append_event({"event": "start", "attempt": attempt, "utc": now(), "split": split,
                          "sample_id": row["sample_id"], "cache_key": key})
            payload = request_payload(row["text"], cfg)
            record = {"attempt": attempt, "split": split, "sample_id": row["sample_id"], "cache_key": key,
                      "text_sha256": row["text_sha256"], "started_utc": now(), "request": payload,
                      "status": "failed", "http_status": None, "cost_usd": None}
            t = time.perf_counter()
            try:
                response = session.post(cfg["endpoint"], json=payload,
                    headers={"Authorization": "Bearer " + secret, "Content-Type": "application/json"},
                    timeout=(cfg["timeout_connect_s"], cfg["timeout_read_s"]), allow_redirects=False)
                record["http_status"] = response.status_code
                # Persist raw response bytes as text. No request headers or credentials are saved.
                record["raw_response_text"] = response.text.replace(secret, "[REDACTED]")
                try:
                    body = json.loads(record["raw_response_text"])
                    record["response"] = body
                    record["cost_usd"] = usage_cost(body)
                except json.JSONDecodeError:
                    body = None
                if response.status_code != 200:
                    record["error"] = f"http_{response.status_code}"
                elif body is None:
                    record["error"] = "invalid_json"
                else:
                    try:
                        record.update(parse_response(body, cfg))
                        record["status"] = "valid"
                    except (ValueError, TypeError, KeyError) as e:
                        record["error"] = str(e).replace(secret, "[REDACTED]")
            except requests.Timeout:
                record["error"] = "timeout_billing_unknown"
            except requests.RequestException as e:
                # Do not serialize exception strings, which may carry request details.
                record["error"] = type(e).__name__
            record["latency_s"] = time.perf_counter() - t
            record["ended_utc"] = now()
            write_json(cp, record)
            append_event({"event": "finish", "attempt": attempt, "utc": record["ended_utc"],
                          "split": split, "sample_id": row["sample_id"], "cache_key": key,
                          "status": record["status"], "http_status": record["http_status"],
                          "cost_usd": record["cost_usd"], "latency_s": record["latency_s"],
                          "error": record.get("error")})
            summary.append(record)
            consecutive = 0 if record["status"] == "valid" else consecutive + 1
            print(f"{split} {len(summary)}/{len(rows)} {record['status']} {record['latency_s']:.3f}s cost={record['cost_usd']}", flush=True)
            if record["http_status"] in (400, 401, 402, 403, 404, 413):
                stop(record["error"])
                break
            if "timeout" in record.get("error", "").lower() or record["cost_usd"] is None:
                stop("billing_unknown_after_failure; no retry; inspect account activity")
                break
            if consecutive >= 3:
                stop("three_consecutive_failures")
                break
            if sum(e.get("cost_usd") or 0 for e in ledger() if e["event"] == "finish") >= cfg["cost_target_usd"]:
                stop("cost_target_reached")
                break
    run_summary = {"split": split, "expected": len(rows), "records": len(summary),
                   "valid": sum(r["status"] == "valid" for r in summary),
                   "failed": sum(r["status"] != "valid" for r in summary),
                   "unattempted": len(rows) - len(summary), "updated_utc": now()}
    write_json(ROOT / "results" / f"{split}_run.json", run_summary)
    return run_summary


def merged(split):
    cfg = config()
    f = pd.read_csv(ROOT / "data" / f"{split}.csv", keep_default_na=False)
    rows = []
    for row in f.to_dict("records"):
        cp = ROOT / "results/cache" / f"{cache_key(row, cfg)}.json"
        rec = read_json(cp) if cp.exists() else {"status": "not_attempted"}
        out = {**row, **{k: rec.get(k) for k in
            ("status", "attempt", "jev_choice", "p_spam", "p_ham", "served_model", "response_id", "provider",
             "http_status", "latency_s", "cost_usd", "error", "choice_argmax_agrees")}}
        out["workflow"] = workflow(rec.get("p_spam"), rec["status"] == "valid", cfg)
        rows.append(out)
    return pd.DataFrame(rows)


def metric_values(y, pred, p):
    y = np.asarray(y, dtype=int); pred = np.asarray(pred, dtype=int); p = np.asarray(p, dtype=float)
    if not len(y):
        return {"n": 0}
    assert len(y) == len(pred) == len(p)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"n": len(y), "ham_n": int((y == 0).sum()), "spam_n": int((y == 1).sum()),
        "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "spam_precision": float(precision), "spam_recall": float(recall), "spam_f1": float(f1),
        "brier_score": float(brier_score_loss(y, p)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "ham_as_spam_count": int(fp), "ham_as_spam_rate": float(fp / (tn + fp)) if tn + fp else None,
        "spam_as_ham_count": int(fn), "spam_as_ham_rate": float(fn / (fn + tp)) if fn + tp else None}


def workflow_values(y, actions):
    y = np.asarray(y, dtype=int); actions = np.asarray(actions)
    block = actions == "block"; release = actions == "release"; auto = block | release
    correct = (block & (y == 1)) | (release & (y == 0))
    return {"n": len(y), "auto_n": int(auto.sum()), "block_n": int(block.sum()), "release_n": int(release.sum()),
        "review_n": int((~auto).sum()), "automation_coverage": float(auto.mean()),
        "review_rate": float((~auto).mean()), "auto_accuracy": float(correct[auto].mean()) if auto.any() else None,
        "false_block_count": int((block & (y == 0)).sum()), "missed_block_count": int((release & (y == 1)).sum()),
        "false_block_rate_within_block": float((block & (y == 0)).sum() / block.sum()) if block.any() else None,
        "missed_block_rate_within_release": float((release & (y == 1)).sum() / release.sum()) if release.any() else None}


def freeze():
    validate_splits()
    dev = merged("dev")
    dev.to_csv(ROOT / "results/dev_predictions.csv", index=False)
    valid = dev.loc[dev.status.eq("valid")]
    m = metric_values(valid.label.eq("spam"), valid.jev_choice.eq("spam"), valid.p_spam)
    write_json(ROOT / "results/dev_metrics.json", {"classification": m,
        "workflow": workflow_values(dev.label.eq("spam"), dev.workflow),
        "failed": int(dev.status.eq("failed").sum()), "unattempted": int(dev.status.eq("not_attempted").sum())})
    if len(valid) != 20:
        raise RuntimeError("Development flow incomplete; not freezing or starting test")
    path = ROOT / "config/frozen.json"
    if not path.exists():
        write_json(path, {"frozen_utc": now(), "criteria_revisions": 0,
            "decision": "Keep initial v1 criteria and fixed thresholds; complete dev response validation passed. No test results examined.",
            "config_sha256": digest((ROOT / "config/jev_v1.json").read_bytes()),
            "dev_predictions_sha256": digest((ROOT / "results/dev_predictions.csv").read_bytes()),
            "manifest_splits": read_json(ROOT / "data/manifest.json")["splits"],
            "config": config()})
    validate_frozen()
    print("Development metrics:", json.dumps(m), flush=True)


def validate_frozen():
    f = read_json(ROOT / "config/frozen.json")
    assert f["config_sha256"] == digest((ROOT / "config/jev_v1.json").read_bytes())
    assert f["manifest_splits"] == read_json(ROOT / "data/manifest.json")["splits"]
    assert f["dev_predictions_sha256"] == digest((ROOT / "results/dev_predictions.csv").read_bytes())


def baseline():
    validate_splits()
    prior = ROOT / "results/baseline_predictions.csv"
    if prior.exists():
        print("Using existing baseline predictions and measured timings", flush=True)
        return
    cfg = config()["baseline"]
    train = pd.read_csv(ROOT / "data/train.csv", keep_default_na=False)
    test = pd.read_csv(ROOT / "data/test.csv", keep_default_na=False)
    tfcfg = dict(cfg["tfidf"]); tfcfg["ngram_range"] = tuple(tfcfg["ngram_range"])
    model = Pipeline([("tfidf", TfidfVectorizer(**tfcfg)),
                      ("lr", LogisticRegression(**cfg["logistic_regression"]))])
    t = time.perf_counter(); model.fit(train.text, train.label.eq("spam").astype(int))
    train_s = time.perf_counter() - t
    t = time.perf_counter(); p = model.predict_proba(test.text)[:, list(model.classes_).index(1)]
    batch_s = time.perf_counter() - t
    singles = []
    for text in test.text:
        t = time.perf_counter(); model.predict_proba([text]); singles.append(time.perf_counter() - t)
    test["baseline_p_spam"] = p
    test["baseline_choice"] = np.where(p >= cfg["decision_threshold"], "spam", "ham")
    test["baseline_workflow"] = [workflow(v) for v in p]
    test.to_csv(prior, index=False)
    joblib.dump(model, ROOT / "results/baseline_model.joblib")
    write_json(ROOT / "results/baseline_timing.json", {"trained_utc": now(), "train_n": len(train),
        "train_s": train_s, "predict_batch_100_s": batch_s, "predict_single_p50_s": float(np.quantile(singles, .5)),
        "predict_single_p95_s": float(np.quantile(singles, .95)), "vocabulary_n": len(model["tfidf"].vocabulary_),
        "config": cfg, "train_sha256": digest((ROOT / "data/train.csv").read_bytes())})
    print(f"Baseline trained on {len(train)} rows in {train_s:.4f}s; prediction {batch_s:.4f}s", flush=True)


def calibration_rows(y, p, name):
    y = np.asarray(y); p = np.asarray(p)
    ids = np.minimum((p * 5).astype(int), 4)
    rows = []
    for i in range(5):
        mask = ids == i
        rows.append({"model": name, "bin": i, "lower": i / 5, "upper": (i + 1) / 5,
            "n": int(mask.sum()), "mean_p_spam": float(p[mask].mean()) if mask.any() else None,
            "observed_spam_rate": float(y[mask].mean()) if mask.any() else None})
    return rows


def evaluate():
    validate_splits()
    # Evaluation can still produce honest baseline-only outputs when API is blocked.
    if (ROOT / "config/frozen.json").exists():
        validate_frozen()
    f = merged("test")
    b = pd.read_csv(ROOT / "results/baseline_predictions.csv", keep_default_na=False)
    assert list(f.sample_id) == list(b.sample_id)
    assert list(f.label) == list(b.label) and list(f.text) == list(b.text)
    for col in ("baseline_p_spam", "baseline_choice", "baseline_workflow"):
        f[col] = b[col]
    f["jev_correct"] = np.where(f.status.eq("valid"), f.jev_choice.eq(f.label), None)
    f["baseline_correct"] = f.baseline_choice.eq(f.label)
    f.to_csv(ROOT / "results/test_predictions.csv", index=False)
    valid = f.loc[f.status.eq("valid")].copy()
    datasets = [("jev_valid", valid, "jev_choice", "p_spam"),
                ("baseline_same_valid", valid, "baseline_choice", "baseline_p_spam"),
                ("baseline_all_test", f, "baseline_choice", "baseline_p_spam")]
    metrics = [{"model_subset": name, **metric_values(d.label.eq("spam"), d[c].eq("spam"), d[p])}
               for name, d, c, p in datasets]
    pd.DataFrame(metrics).to_csv(ROOT / "results/metrics.csv", index=False)
    workflows = [{"model": "jev_all_test", **workflow_values(f.label.eq("spam"), f.workflow)},
                 {"model": "baseline_all_test", **workflow_values(f.label.eq("spam"), f.baseline_workflow)}]
    pd.DataFrame(workflows).to_csv(ROOT / "results/workflow_metrics.csv", index=False)
    ends = [e for e in ledger() if e["event"] == "finish"]
    starts = [e for e in ledger() if e["event"] == "start"]
    test_ends = [e for e in ends if e["split"] == "test"]
    lat = [e["latency_s"] for e in test_ends]
    api = {"test_n": len(f), "test_attempted": len([e for e in starts if e["split"] == "test"]),
        "test_valid": len(valid), "test_failed": int(f.status.eq("failed").sum()),
        "test_unattempted": int(f.status.eq("not_attempted").sum()),
        "test_success_rate_all": len(valid) / len(f),
        "test_success_rate_attempted": len(valid) / len(test_ends) if test_ends else None,
        "latency_p50_s": float(np.quantile(lat, .5)) if lat else None,
        "latency_p95_s": float(np.quantile(lat, .95)) if lat else None,
        "latency_population": "All completed test attempts, including failures; original network timings, not cache reads",
        "total_requests": len(starts), "known_cost_usd_all": sum(e.get("cost_usd") or 0 for e in ends),
        "known_cost_usd_test": sum(e.get("cost_usd") or 0 for e in test_ends),
        "unknown_cost_attempts": sum(e.get("cost_usd") is None for e in ends) + len(starts) - len(ends),
        "served_models": sorted(valid.served_model.unique().tolist()),
        "choice_argmax_disagreements": int((valid.choice_argmax_agrees == False).sum()),
        "stop": read_json(ROOT / "results/api_stop.json") if (ROOT / "results/api_stop.json").exists() else None}
    write_json(ROOT / "results/api_summary.json", api)
    pd.DataFrame([{**api, "served_models": ";".join(api["served_models"]),
                   "stop": json.dumps(api["stop"])}]).to_csv(ROOT / "results/api_metrics.csv", index=False)
    bt = read_json(ROOT / "results/baseline_timing.json")
    pd.DataFrame([{k: v for k, v in bt.items() if k != "config"}]).to_csv(ROOT / "results/baseline_timing.csv", index=False)
    phase_costs = [{"phase": phase, "requests": sum(e["split"] == phase for e in starts),
                    "known_cost_usd": sum(e.get("cost_usd") or 0 for e in ends if e["split"] == phase),
                    "valid_responses": sum(e["split"] == phase and e["status"] == "valid" for e in ends)}
                   for phase in ("smoke", "dev", "test")]
    pd.DataFrame(phase_costs).to_csv(ROOT / "results/api_phase_costs.csv", index=False)
    cals = sum([calibration_rows(d.label.eq("spam"), d[p].astype(float), name)
                for name, d, _, p in datasets if len(d)], [])
    pd.DataFrame(cals).to_csv(ROOT / "results/calibration_bins.csv", index=False)
    errors = []
    for name, d, c, p in datasets:
        if name == "baseline_same_valid":
            continue
        for row in d.loc[d[c].ne(d.label)].to_dict("records"):
            errors.append({"model": name, "sample_id": row["sample_id"], "label": row["label"],
                "prediction": row[c], "p_spam": row[p], "predicted_class_probability": row[p] if row[c] == "spam" else 1 - row[p],
                "workflow": row["workflow"] if name == "jev_valid" else row["baseline_workflow"], "text": row["text"]})
    error_df = pd.DataFrame(errors, columns=["model", "sample_id", "label", "prediction", "p_spam", "predicted_class_probability", "workflow", "text"])
    error_df = error_df.sort_values("predicted_class_probability", ascending=False)
    error_df.to_csv(ROOT / "results/errors_all.csv", index=False)
    error_df.head(10).to_csv(ROOT / "results/errors_representative.csv", index=False)
    plots(datasets, cals)
    write_json(ROOT / "results/summary.json", {"metrics": metrics, "workflows": workflows, "api": api,
               "baseline_timing": read_json(ROOT / "results/baseline_timing.json"), "evaluated_utc": now()})
    print(json.dumps({"metrics": metrics, "workflows": workflows, "api": api}, ensure_ascii=False, indent=2), flush=True)


def plots(datasets, cals):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, (name, d, c, _) in zip(axes, datasets[:2]):
        cm = confusion_matrix(d.label.eq("spam"), d[c].eq("spam"), labels=[0, 1])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()))
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=22,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["ham", "spam"], yticklabels=["ham", "spam"],
               xlabel="Predicted class", ylabel="True label", title=f"{name}\nn={len(d)}")
    fig.savefig(ROOT / "results/confusion_matrix.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), constrained_layout=True)
    for ax, (name, d, _, _) in zip(axes, datasets[:2]):
        bins = [r for r in cals if r["model"] == name and r["n"]]
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Ideal reference")
        if bins:
            ax.plot([r["mean_p_spam"] for r in bins], [r["observed_spam_rate"] for r in bins], "o-", color="#255f85")
            for r in bins:
                ax.annotate(f"n={r['n']}", (r["mean_p_spam"], r["observed_spam_rate"]),
                            xytext=(4, -18 if r["observed_spam_rate"] > .8 else 9), textcoords="offset points")
        ax.set(xlim=(-.04, 1.08), ylim=(-.08, 1.08), xlabel="Mean predicted P(spam)",
               ylabel="Observed spam fraction", title=f"{name} (n={len(d)})\n5 fixed-width bins; empty bins omitted")
        ax.grid(alpha=.2)
    fig.suptitle("Balanced small-sample diagnostic; does not establish calibration", fontsize=11)
    fig.savefig(ROOT / "results/calibration.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "smoke", "dev", "freeze", "baseline", "test", "evaluate", "reproduce", "all"])
    command = parser.parse_args().command
    if command == "prepare": prepare()
    elif command in ("smoke", "dev", "test"): run_api(command)
    elif command == "freeze": freeze()
    elif command == "baseline": baseline()
    elif command == "evaluate": evaluate()
    elif command == "reproduce": validate_splits(); baseline(); evaluate()
    else:
        prepare(); baseline()
        try:
            run_api("smoke"); run_api("dev"); freeze(); run_api("test")
        except (RuntimeError, FileNotFoundError) as e:
            print(f"Incomplete API experiment: {e}", flush=True)
        evaluate()


if __name__ == "__main__":
    main()
