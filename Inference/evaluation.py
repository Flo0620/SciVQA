"""Module for evaluation of retrieval and QA results with MLflow tracking."""

import logging
from pathlib import Path

import hydra
import mlflow
from encourage.llm import BatchInferenceRunner, Response
from encourage.metrics import Metric, MetricOutput
from vllm import SamplingParams
import numpy as np

from scivqa.evaluation.factory_helper import load_metrics
from scivqa.utils.file_manager import FileManager
from scivqa.utils.flatten_dict import flatten_dict
from src.scivqa.evaluation.config import Config

logger = logging.getLogger(__name__)
config_path = str((Path(__file__).parents[3] / "conf").resolve())


@hydra.main(version_base=None, config_path=config_path, config_name="defaults")
def main(cfg: Config) -> None:
    """Main function for evaluation of QA results with MLflow tracking."""
    # Set MLflow tracking configuration
    mlflow.set_tracking_uri(cfg.mlflow.uri)
    mlflow.set_experiment(experiment_name=cfg.mlflow.experiment_id)

    # Run the evaluation with MLflow tracking
    if mlflow.active_run().info.run_id if mlflow.active_run() else None:  # type: ignore
        evaluation(cfg)
    else:
        with mlflow.start_run():
            evaluation(cfg)


def evaluation(cfg: Config) -> None:
    """Evaluate the QA results with MLflow tracking."""
    flat_config = flatten_dict(cfg)
    mlflow.log_params(flat_config)

    with mlflow.start_span(name="loading_results"):
        print("OUtput folder", cfg.output_folder)
        results_path = Path(cfg.output_folder)
        if not results_path.exists() or not results_path.is_dir():
            raise ValueError(f"Results folder not found: {results_path}")

        # Convert to ResponseDataCollection format
        responses_json = FileManager(list(results_path.glob("inference_log.json"))[0]).load_json()
        responses = [Response.from_dict(item) for item in responses_json]

        logger.info(f"Loaded {len(responses)} responses!")

    if cfg.use_vllm:
        sampling_params = SamplingParams(
            temperature=cfg.model.temperature, max_tokens=cfg.model.max_tokens
        )
        runner = BatchInferenceRunner(sampling_params, cfg.model.model_name, base_url=cfg.base_url)
    else:
        runner = None

    # Load metrics
    metrics: list[Metric] = load_metrics(cfg.metrics, runner)
    metrics_log = []
    for metric in metrics:
        result: MetricOutput = metric(responses)
        metrics_log.append({metric.name: result.to_dict()})

        if metric.name in ["bertscore", "rouge1", "rougeL"]:
            mlflow.log_metric(metric.name + "_f1", result.score)
            mlflow.log_metric(metric.name + "_precision", np.mean(result.misc["precision"]))
            mlflow.log_metric(metric.name + "_recall", np.mean(result.misc["recall"]))
        else:
            mlflow.log_metric(metric.name, result.score)

    FileManager(cfg.output_folder + "/metrics_log.json").dump_json(
        metrics_log, pydantic_encoder=True
    )


if __name__ == "__main__":
    main()
