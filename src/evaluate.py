"""Evaluate the travel RAG pipeline with the four configured Ragas metrics."""
import asyncio
import json
import logging
import math
import os
import sys
from pathlib import Path
from textwrap import wrap
from typing import Dict, List

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

_ASYNCIO_C_TASK = (
    asyncio.tasks._CTask if sys.version_info >= (3, 14) else None
)

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

# Ragas 0.2's nest_asyncio patch replaces asyncio's C Task with a Python Task.
# Python 3.14's asyncio.timeout cannot detect that task, so restore the C Task.
if _ASYNCIO_C_TASK is not None:
    asyncio.Task = asyncio.tasks.Task = asyncio.tasks._CTask = _ASYNCIO_C_TASK

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("evaluation")
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
METRICS = (faithfulness, answer_relevancy, context_precision, context_recall)
MINIMUM_SCORES = {name: 0.70 for name in METRIC_NAMES}
CATEGORY_ACCURACY_THRESHOLD = 0.70
EVALUATION_THRESHOLDS = {
    **MINIMUM_SCORES,
    "category_accuracy": CATEGORY_ACCURACY_THRESHOLD,
}
REPORT_FILES = (
    ("Evaluation summary", "evaluation_summary.json"),
    ("Detailed results", "evaluation_detailed.csv"),
    ("Category breakdown", "category_breakdown.json"),
)
BOX_WIDTH = 70
BOX_CONTENT_WIDTH = BOX_WIDTH - 4


def _print_box_rule() -> None:
    print("|" + "=" * (BOX_WIDTH - 2) + "|")


def _print_box_line(text: str = "") -> None:
    lines = wrap(text, width=BOX_CONTENT_WIDTH) if len(text) > BOX_CONTENT_WIDTH else [text]
    for line in lines:
        print(f"| {line:<{BOX_CONTENT_WIDTH}} |")


class TravelChatbotEvaluator:
    """Runs Ragas evaluation against the repository's golden dataset."""

    def __init__(self):
        self.engine = None
        self.golden_dataset_path = PROJECT_ROOT / "data" / "golden_dataset.json"
        self.output_dir = PROJECT_ROOT / "reports"

    def load_golden_dataset(self) -> List[Dict]:
        if not self.golden_dataset_path.is_file():
            raise FileNotFoundError(f"Golden dataset not found: {self.golden_dataset_path}")
        with self.golden_dataset_path.open(encoding="utf-8") as dataset_file:
            data = json.load(dataset_file)
        if not isinstance(data, list) or not data:
            raise ValueError("Golden dataset must be a non-empty JSON list.")
        for index, item in enumerate(data):
            if (
                not isinstance(item, dict)
                or not item.get("question")
                or not item.get("ground_truth")
                or not item.get("category")
            ):
                raise ValueError(
                    f"Golden dataset entry {index} requires question, ground_truth, and category."
                )
        return data

    def generate_responses(self, questions: List[str]) -> tuple:
        answers, contexts, retrieved_categories = [], [], []
        engine = self.engine
        if engine is None:
            from src.search_engine import TravelSearchEngine

            engine = TravelSearchEngine(enable_mlflow_tracking=False)
        self.engine = engine
        total = len(questions)
        _print_box_line("Generating responses...")
        for index, question in enumerate(questions, start=1):
            if index <= 2:
                _print_box_line("DEBUG: Text Query processing...")
            docs, _ = engine.search_by_text(question, k=5)
            answer = engine.synthesize_response(docs, question)
            answers.append(answer)
            contexts.append([document.page_content for document in docs])
            retrieved_categories.append(
                docs[0].metadata.get("category") if docs else None
            )
        if total > 2:
            _print_box_line(f"... ({total} queries total)")
        return answers, contexts, retrieved_categories

    def _azure_ragas_models(self):
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        if self.engine is None:
            raise RuntimeError("The search engine must be initialized before Ragas evaluation.")

        class AzureRagasLLMWrapper(LangchainLLMWrapper):
            def get_temperature(self, n: int) -> float:
                return 1.0

        return AzureRagasLLMWrapper(self.engine.llm), LangchainEmbeddingsWrapper(
            self.engine.embeddings
        )

    def run_ragas_evaluation(self):
        golden_data = self.load_golden_dataset()
        _print_box_line(f"Loaded {len(golden_data)} test cases from golden dataset")
        _print_box_line()
        questions = [item["question"] for item in golden_data]
        ground_truths = [item["ground_truth"] for item in golden_data]
        categories = [item["category"] for item in golden_data]
        answers, contexts, retrieved_categories = self.generate_responses(questions)
        _print_box_line()
        category_matches = [
            retrieved == expected
            for expected, retrieved in zip(categories, retrieved_categories)
        ]
        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
            "category": categories,
        }

        dataset = Dataset.from_dict(dataset_dict)
        llm, embeddings = self._azure_ragas_models()
        _print_box_line("Running Ragas metrics...")
        metric_lines = wrap(
            "Metrics: " + ", ".join(METRIC_NAMES),
            width=BOX_CONTENT_WIDTH,
            subsequent_indent="          ",
        )
        for line in metric_lines:
            _print_box_line(line)
        _print_box_line()
        results = evaluate(
            dataset,
            metrics=list(METRICS),
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=True,
            show_progress=False,
        )
        scores_frame = results.to_pandas()
        for metric in METRIC_NAMES:
            if metric not in scores_frame:
                raise ValueError(f"Ragas result did not include required metric {metric!r}.")
        if len(scores_frame) != len(questions):
            raise ValueError(
                f"Ragas returned {len(scores_frame)} result rows for {len(questions)} cases."
            )
        metric_scores = scores_frame[list(METRIC_NAMES)].apply(
            pd.to_numeric, errors="coerce"
        )
        invalid_metrics = [
            metric
            for metric in METRIC_NAMES
            if not metric_scores[metric].map(math.isfinite).all()
        ]
        if invalid_metrics:
            raise ValueError(
                "Ragas returned missing or non-finite scores for: "
                + ", ".join(invalid_metrics)
            )
        means = metric_scores.mean()
        scores = {metric: float(means[metric]) for metric in METRIC_NAMES}
        scores["category_accuracy"] = sum(category_matches) / len(category_matches)
        passed = all(
            math.isfinite(scores[metric]) and scores[metric] >= threshold
            for metric, threshold in EVALUATION_THRESHOLDS.items()
        )
        self._save_results(
            scores,
            scores_frame,
            dataset_dict,
            passed,
            category_matches=category_matches,
        )
        self._print_results(scores, passed)
        return {**scores, "passed": passed}

    def _print_results(self, scores: Dict[str, float], passed: bool) -> None:
        _print_box_rule()
        _print_box_line("EVALUATION RESULTS")
        _print_box_rule()
        _print_box_line()
        _print_box_line("Ragas Scores:")
        _print_box_line()
        display_names = {
            "faithfulness": "Faithfulness",
            "answer_relevancy": "Answer Relevancy",
            "context_precision": "Context Precision",
            "context_recall": "Context Recall",
            "category_accuracy": "Category Accuracy",
        }
        for metric in (*METRIC_NAMES, "category_accuracy"):
            label = display_names[metric]
            score = scores[metric]
            symbol = "✅" if score >= EVALUATION_THRESHOLDS[metric] else "❌"
            _print_box_line(f"  {label:<20} {score:.4f}  {symbol}")
        _print_box_rule()
        _print_box_line()

        for report_file in REPORT_FILES:
            label, filename = report_file
            _print_box_line(f"✅ {label} saved to reports/{filename}")

        symbol = "✅" if passed else "❌"
        _print_box_line()
        _print_box_line(f"{symbol} EVALUATION {'PASSED' if passed else 'FAILED'}")
        _print_box_line()
        for metric, threshold in EVALUATION_THRESHOLDS.items():
            score = scores[metric]
            label = display_names[metric]
            _print_box_line(f"  {label}: {score:.4f} (>= {threshold:.2f})")

    def _save_results(
        self,
        scores: Dict[str, float],
        scores_frame: pd.DataFrame,
        dataset_dict: dict,
        passed: bool,
        category_matches: List[bool],
    ):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            **scores,
            "total_test_cases": len(dataset_dict["question"]),
            "thresholds": EVALUATION_THRESHOLDS,
            "passed": passed,
        }
        summary_path = self.output_dir / "evaluation_summary.json"
        with summary_path.open("w", encoding="utf-8") as output_file:
            json.dump(summary, output_file, indent=2, allow_nan=False)

        details = pd.DataFrame(
            {
                "question": dataset_dict["question"],
                "ground_truth": dataset_dict["ground_truth"],
                "answer": dataset_dict["answer"],
                "contexts": ["\n\n".join(items) for items in dataset_dict["contexts"]],
                "category": dataset_dict["category"],
                "category_correct": category_matches,
            }
        )
        for metric in METRIC_NAMES:
            details[metric] = scores_frame[metric].tolist()
        details["passed"] = details[list(METRIC_NAMES)].ge(
            pd.Series(MINIMUM_SCORES)
        ).all(axis=1) & details["category_correct"]
        details.to_csv(self.output_dir / "evaluation_detailed.csv", index=False)

        breakdown = {}
        for category, correct in zip(dataset_dict["category"], category_matches):
            stats = breakdown.setdefault(category, {"correct": 0, "total": 0})
            stats["correct"] += int(correct)
            stats["total"] += 1
        for stats in breakdown.values():
            stats["accuracy"] = stats["correct"] / stats["total"]
        breakdown_path = self.output_dir / "category_breakdown.json"
        with breakdown_path.open("w", encoding="utf-8") as output_file:
            json.dump(
                {
                    "overall_accuracy": scores["category_accuracy"],
                    "categories": breakdown,
                },
                output_file,
                indent=2,
                allow_nan=False,
            )

    def run(self):
        return self.run_ragas_evaluation()


def run_evaluation():
    _print_box_rule()
    _print_box_line("Starting Ragas Evaluation...")
    _print_box_rule()
    try:
        evaluator = TravelChatbotEvaluator()
        results = evaluator.run()
    except Exception:
        logger.exception("Ragas evaluation failed")
        return 1
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(run_evaluation())
