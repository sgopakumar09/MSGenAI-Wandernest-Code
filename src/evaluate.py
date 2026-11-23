"""Evaluate the travel RAG pipeline with the four configured Ragas metrics."""
import json
import logging
import math
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from src.search_engine import TravelSearchEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluation")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
METRICS = (faithfulness, answer_relevancy, context_precision, context_recall)
MINIMUM_SCORES = {name: 0.70 for name in METRIC_NAMES}


class TravelChatbotEvaluator:
    """Runs Ragas evaluation against the repository's golden dataset."""

    def __init__(self):
        self.engine = TravelSearchEngine()
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
            if not isinstance(item, dict) or not item.get("question") or not item.get("ground_truth"):
                raise ValueError(
                    f"Golden dataset entry {index} requires question and ground_truth."
                )
        return data

    def generate_responses(self, questions: List[str]) -> tuple:
        answers, contexts = [], []
        for question in questions:
            docs, _ = self.engine.search_by_text(question, k=5)
            answer = self.engine.synthesize_response(docs, question)
            answers.append(answer)
            contexts.append([document.page_content for document in docs])
        return answers, contexts

    def _azure_ragas_models(self):
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        return LangchainLLMWrapper(self.engine.llm), LangchainEmbeddingsWrapper(
            self.engine.embeddings
        )

    def run_ragas_evaluation(self):
        golden_data = self.load_golden_dataset()
        questions = [item["question"] for item in golden_data]
        ground_truths = [item["ground_truth"] for item in golden_data]
        answers, contexts = self.generate_responses(questions)
        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }

        dataset = Dataset.from_dict(dataset_dict)
        llm, embeddings = self._azure_ragas_models()
        results = evaluate(dataset, metrics=list(METRICS), llm=llm, embeddings=embeddings)
        scores_frame = results.to_pandas()
        for metric in METRIC_NAMES:
            if metric not in scores_frame:
                raise ValueError(f"Ragas result did not include required metric {metric!r}.")
        means = scores_frame[list(METRIC_NAMES)].mean(skipna=True).fillna(0.0)
        scores = {metric: float(means[metric]) for metric in METRIC_NAMES}
        scores = {
            metric: score if math.isfinite(score) else 0.0
            for metric, score in scores.items()
        }
        passed = all(
            math.isfinite(scores[metric]) and scores[metric] >= MINIMUM_SCORES[metric]
            for metric in METRIC_NAMES
        )
        self._save_results(scores, scores_frame, dataset_dict, passed)
        for metric, score in scores.items():
            logger.info("%s: %.4f", metric, score)
        logger.info("Evaluation %s", "PASSED" if passed else "FAILED")
        return {**scores, "passed": passed}

    def _save_results(
        self,
        scores: Dict[str, float],
        scores_frame: pd.DataFrame,
        dataset_dict: dict,
        passed: bool,
    ):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            **scores,
            "total_test_cases": len(dataset_dict["question"]),
            "thresholds": MINIMUM_SCORES,
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
            }
        )
        for metric in METRIC_NAMES:
            details[metric] = scores_frame[metric].tolist()
        details["passed"] = details[list(METRIC_NAMES)].ge(
            pd.Series(MINIMUM_SCORES)
        ).all(axis=1)
        details.to_csv(self.output_dir / "evaluation_detailed.csv", index=False)
        logger.info("Saved evaluation summary to %s", summary_path)
        logger.info("Saved per-question scores to %s", self.output_dir / "evaluation_detailed.csv")

    def run(self):
        return self.run_ragas_evaluation()


def run_evaluation():
    try:
        evaluator = TravelChatbotEvaluator()
        results = evaluator.run()
    except Exception:
        logger.exception("Ragas evaluation failed")
        return 1
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(run_evaluation())
