import json

import pandas as pd

from src.evaluate import METRIC_NAMES, TravelChatbotEvaluator


def test_evaluation_report_contains_all_metric_scores_and_pass_status(tmp_path):
    evaluator = object.__new__(TravelChatbotEvaluator)
    evaluator.output_dir = tmp_path
    data = {
        "question": ["What is the refund deadline?"],
        "ground_truth": ["Within seven business days."],
        "answer": ["Refunds are due within seven business days."],
        "contexts": [["Refunds must be issued within seven business days."]],
    }
    frame = pd.DataFrame({metric: [0.8] for metric in METRIC_NAMES})

    evaluator._save_results(
        {metric: 0.8 for metric in METRIC_NAMES}, frame, data, passed=True
    )

    summary = json.loads((tmp_path / "evaluation_summary.json").read_text())
    details = pd.read_csv(tmp_path / "evaluation_detailed.csv")
    assert set(METRIC_NAMES) <= summary.keys()
    assert summary["passed"] is True
    assert details.loc[0, "passed"]
