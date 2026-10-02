"""Verify real artifacts independently of the evaluation function; no API calls."""
import json
import math
import os
import stat
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import experiment as e


def main():
    os.chdir(e.ROOT)
    e.validate_splits()
    e.validate_frozen()
    f = pd.read_csv("results/test_predictions.csv")
    assert len(f) == 100 and f.status.eq("valid").all()
    config = e.config()
    starts = [r for r in e.ledger() if r["event"] == "start"]
    ends = [r for r in e.ledger() if r["event"] == "finish"]
    assert len(starts) == len(ends) == 121
    assert len({r["cache_key"] for r in starts}) == 121
    assert {r["attempt"] for r in starts} == {r["attempt"] for r in ends}
    assert all(r["status"] == "valid" for r in ends)
    frozen = e.read_json("config/frozen.json")
    assert frozen["frozen_utc"] < min(r["utc"] for r in starts if r["split"] == "test")
    total = 0
    for start in starts:
        r = e.read_json(Path("results/cache") / (start["cache_key"] + ".json"))
        assert r["response"] == json.loads(r["raw_response_text"])
        answer = e.parse_response(r["response"], config)
        assert answer["jev_choice"] == r["jev_choice"] and answer["p_spam"] == r["p_spam"]
        assert r["http_status"] == 200
        assert r["cost_usd"] == e.usage_cost(r["response"])
        assert set(r["request"]) == {"model", "state", "questions"}
        assert set(r["request"]["state"]) == {"sms_text"}
        assert r["request"]["questions"] == config["questions"]
        total += r["cost_usd"]
        if r["split"] == "test":
            csvrow = f.loc[f.sample_id.eq(r["sample_id"])].iloc[0]
            assert csvrow.text == r["request"]["state"]["sms_text"]
            assert csvrow.jev_choice == r["jev_choice"]
            assert math.isclose(csvrow.p_spam, r["p_spam"], abs_tol=1e-14)
    api = e.read_json("results/api_summary.json")
    assert math.isclose(total, api["known_cost_usd_all"], abs_tol=1e-14)
    assert total < .1 and len(starts) <= 150
    metrics = pd.read_csv("results/metrics.csv").set_index("model_subset")
    y = f.label.eq("spam").to_numpy()
    for name, choice, probability in (("jev_valid", "jev_choice", "p_spam"),
                                       ("baseline_same_valid", "baseline_choice", "baseline_p_spam"),
                                       ("baseline_all_test", "baseline_choice", "baseline_p_spam")):
        pred = f[choice].eq("spam").to_numpy()
        tn = int((~y & ~pred).sum()); fp = int((~y & pred).sum())
        fn = int((y & ~pred).sum()); tp = int((y & pred).sum())
        expected = {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
                    "accuracy": (tn + tp) / len(f),
                    "balanced_accuracy": ((tp / (tp + fn)) + (tn / (tn + fp))) / 2,
                    "spam_precision": tp / (tp + fp), "spam_recall": tp / (tp + fn),
                    "spam_f1": 2 * tp / (2 * tp + fp + fn),
                    "brier_score": np.mean((f[probability].to_numpy() - y.astype(int)) ** 2)}
        for key, value in expected.items():
            assert math.isclose(metrics.loc[name, key], value, abs_tol=1e-12), (name, key)
    model = joblib.load("results/baseline_model.joblib")
    assert np.allclose(model.predict_proba(f.text)[:, 1], f.baseline_p_spam, atol=1e-14)
    wf = pd.read_csv("results/workflow_metrics.csv").set_index("model")
    for name, col in (("jev_all_test", "workflow"), ("baseline_all_test", "baseline_workflow")):
        actions = f[col]
        release = actions.eq("release"); block = actions.eq("block"); auto = release | block
        correct = (release & ~f.label.eq("spam")) | (block & f.label.eq("spam"))
        assert wf.loc[name, "n"] == 100
        assert math.isclose(wf.loc[name, "automation_coverage"], auto.sum() / 100)
        assert math.isclose(wf.loc[name, "auto_accuracy"], correct.sum() / auto.sum())
        assert wf.loc[name, "false_block_count"] == int((block & ~f.label.eq("spam")).sum())
        assert wf.loc[name, "missed_block_count"] == int((release & f.label.eq("spam")).sum())
    bins = pd.read_csv("results/calibration_bins.csv")
    assert all(n == 100 for n in bins.groupby("model").n.sum())
    # Exact secret scan reports filenames only, never the value or matching context.
    private = Path(".env")
    secret_scan = "private runtime secret already removed"
    if private.exists():
        assert stat.S_IMODE(private.stat().st_mode) == 0o600
        value = private.read_text().strip().split("=", 1)[1].encode()
        hits = []
        for p in Path(".").rglob("*"):
            if not p.is_file() or any(part in (".venv", ".uv-cache", ".mpl-cache", "__pycache__") for part in p.parts) or p == private:
                continue
            if value in p.read_bytes(): hits.append(str(p))
        assert not hits, "Secret found in files: " + repr(hits)
        secret_scan = "exact secret absent from all deliverable files; .env was mode 600 and is now removed"
        private.unlink()
    before = len(e.ledger())
    for split in ("smoke", "dev", "test"):
        e.run_api(split)
    assert len(e.ledger()) == before, "Cache replay made a new request"
    report = {"verified_utc": e.now(), "status": "passed", "unit_tests": 7,
              "real_request_count": 121, "request_response_pairs": 121, "test_n": 100,
              "freeze_before_test": True, "metrics_independently_recomputed": True,
              "baseline_saved_model_matches_predictions": True, "cost_usd": total,
              "cache_replay_new_requests": 0, "secret_scan": secret_scan,
              "known_source_caveat": "One original test text contains ROMCAPspam; strict absence of text-internal answer cues cannot be certified. No post-hoc deletion or metric change."}
    e.write_json("results/verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
