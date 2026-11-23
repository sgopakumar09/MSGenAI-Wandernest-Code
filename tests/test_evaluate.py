import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
        "category": ["refund_policies"],
    }
    frame = pd.DataFrame({metric: [0.8] for metric in METRIC_NAMES})

    evaluator._save_results(
        {**{metric: 0.8 for metric in METRIC_NAMES}, "category_accuracy": 1.0},
        frame,
        data,
        passed=True,
        category_matches=[True],
    )

    summary = json.loads((tmp_path / "evaluation_summary.json").read_text())
    details = pd.read_csv(tmp_path / "evaluation_detailed.csv")
    breakdown = json.loads((tmp_path / "category_breakdown.json").read_text())
    assert set(METRIC_NAMES) <= summary.keys()
    assert summary["passed"] is True
    assert summary["category_accuracy"] == 1.0
    assert details.loc[0, "passed"]
    assert details.loc[0, "category_correct"]
    assert breakdown["categories"]["refund_policies"] == {
        "correct": 1,
        "total": 1,
        "accuracy": 1.0,
    }


def test_generate_responses_uses_top_retrieved_document_category():
    evaluator = object.__new__(TravelChatbotEvaluator)
    evaluator.engine = None
    engine = Mock()
    document = SimpleNamespace(
        page_content="Refund policy",
        metadata={"category": "refund_policies"},
    )
    engine.search_by_text.return_value = ([document], "question")
    engine.synthesize_response.return_value = "Refund answer"

    with patch(
        "src.search_engine.TravelSearchEngine", return_value=engine
    ) as constructor:
        answers, contexts, categories = evaluator.generate_responses(["question"])

    constructor.assert_called_once_with(enable_mlflow_tracking=False)
    assert answers == ["Refund answer"]
    assert contexts == [["Refund policy"]]
    assert categories == ["refund_policies"]


def test_ragas_llm_wrapper_uses_azure_supported_temperature():
    evaluator = object.__new__(TravelChatbotEvaluator)
    evaluator.engine = SimpleNamespace(llm=Mock(), embeddings=Mock())

    llm, _ = evaluator._azure_ragas_models()

    assert llm.get_temperature(1) == 1.0


def test_result_output_shows_metrics_reports_and_readiness(tmp_path, capsys):
    evaluator = object.__new__(TravelChatbotEvaluator)
    evaluator.output_dir = tmp_path
    scores = {
        **{metric: 0.8 for metric in METRIC_NAMES},
        "category_accuracy": 0.8,
    }

    evaluator._print_results(scores, passed=True)

    output = capsys.readouterr().out
    assert "Faithfulness" in output
    assert "Category Accuracy" in output
    assert "evaluation_summary.json" in output
    assert "category_breakdown.json" in output
    assert "|====================================================================|" in output
    assert "✅ EVALUATION PASSED" in output
