import copy
import unittest

import numpy as np
import pandas as pd
import experiment as e


class CoreTests(unittest.TestCase):
    def test_conflicts_duplicates_and_invalid_rows(self):
        raw = pd.DataFrame({"label": ["ham", "ham", "ham", "spam", "x", "spam", "ham"],
                            "text": ["a", "a", "conflict", "conflict", "unknown", "  ", None]})
        clean, stats = e.clean_frame(raw)
        self.assertEqual(clean.text.tolist(), ["a"])
        self.assertEqual(stats["duplicate_rows_removed"], 1)
        self.assertEqual(stats["conflict_text_groups"], 1)
        self.assertEqual(stats["conflict_rows_removed"], 2)
        self.assertEqual(stats["invalid_rows_removed"], 3)

    def test_actual_splits_and_repeatability(self):
        e.validate_splits()
        clean = pd.read_csv(e.ROOT / "data/clean.csv", keep_default_na=False)
        again = e.make_splits(clean)
        for name, generated in again.items():
            existing = pd.read_csv(e.ROOT / "data" / f"{name}.csv", keep_default_na=False)
            self.assertEqual(generated.sample_id.tolist(), existing.sample_id.tolist())

    def test_api_payload_has_text_only(self):
        p = e.request_payload("Arbitrary message", e.config())
        self.assertEqual(set(p), {"model", "state", "questions"})
        self.assertEqual(p["state"], {"sms_text": "Arbitrary message"})

    def test_parse_rejects_bad_probabilities_type_and_model(self):
        body = {"model": "typesafe/jev-1.13-20260917", "answers": {"sms_class": {
            "type": "choice", "choice": "spam", "confidence": 0.99,
            "probabilities": {"ham": 0.3, "spam": 0.7}}}}
        self.assertEqual(e.parse_response(body, e.config())["p_spam"], 0.7)
        for value in (-0.1, 1.1, float("nan"), True, "0.7"):
            bad = copy.deepcopy(body); bad["answers"]["sms_class"]["probabilities"]["spam"] = value
            with self.assertRaises(ValueError): e.parse_response(bad, e.config())
        for change in ({"type": "score"}, {"choice": "other"}, {"probabilities": {"ham": .1, "spam": .1}}):
            bad = copy.deepcopy(body); bad["answers"]["sms_class"].update(change)
            with self.assertRaises(ValueError): e.parse_response(bad, e.config())
        bad = copy.deepcopy(body); bad["model"] = "different/model"
        with self.assertRaises(ValueError): e.parse_response(bad, e.config())

    def test_metrics_against_hand_calculation(self):
        y = [0, 0, 1, 1]; pred = [0, 1, 0, 1]; p = [.1, .8, .2, .9]
        m = e.metric_values(y, pred, p)
        self.assertEqual([m[k] for k in ("tn", "fp", "fn", "tp")], [1, 1, 1, 1])
        for k in ("accuracy", "balanced_accuracy", "spam_precision", "spam_recall", "spam_f1"):
            self.assertEqual(m[k], .5)
        self.assertAlmostEqual(m["brier_score"], (.01 + .64 + .64 + .01) / 4)

    def test_workflow_boundaries_failures_and_denominator(self):
        self.assertEqual([e.workflow(p) for p in (0, .1, .10001, .89999, .9, 1)],
                         ["release", "release", "review", "review", "block", "block"])
        self.assertEqual(e.workflow(.99, False), "review")
        m = e.workflow_values([0, 1, 0, 1], ["block", "release", "review", "review"])
        self.assertEqual(m["automation_coverage"], .5)
        self.assertEqual(m["review_rate"], .5)
        self.assertEqual(m["auto_accuracy"], 0)
        self.assertEqual(m["false_block_count"], 1)
        self.assertEqual(m["missed_block_count"], 1)

    def test_calibration_includes_zero_one_and_empty_bins(self):
        rows = e.calibration_rows([0, 1, 1], [0, .2, 1], "test")
        self.assertEqual(sum(r["n"] for r in rows), 3)
        self.assertEqual(rows[0]["n"], 1)
        self.assertEqual(rows[1]["n"], 1)
        self.assertEqual(rows[4]["n"], 1)
        self.assertIsNone(rows[2]["observed_spam_rate"])


if __name__ == "__main__": unittest.main()
