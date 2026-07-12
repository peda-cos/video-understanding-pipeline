"""Offline benchmark evaluation with per-class and grounding metrics."""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

from . import VidqaError
from .ground import iou

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


class SuiteError(VidqaError):
    pass


def load_suite(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SuiteError(f"invalid JSON on line {line_number}") from exc
        if "question" not in row or "answer" not in row:
            raise SuiteError(f"line {line_number} is missing question/answer")
        category = str(row.get("category", "descriptive")).lower()
        if category not in {"counting", "action", "descriptive"}:
            raise SuiteError(f"line {line_number} has unknown category {category!r}")
        row["category"] = category
        rows.append(row)
    if not rows:
        raise SuiteError(f"empty suite: {path}")
    return rows


def normalize_answer(text: str) -> str:
    lowered = text.lower().replace("\u2013", "-").replace("\u2014", "-")
    lowered = re.sub(r"[^a-z0-9\- ]+", " ", lowered)
    lowered = _ARTICLE_RE.sub(" ", lowered)
    return " ".join(lowered.split())


def answers_match(prediction: str, gold: str) -> bool:
    pred = normalize_answer(prediction)
    expected = normalize_answer(gold)
    gold_number = _NUMBER_RE.search(expected)
    if gold_number:
        pred_number = _NUMBER_RE.search(pred)
        if not pred_number:
            return False
        try:
            return float(pred_number.group(0)) == float(gold_number.group(0))
        except ValueError:
            return False
    if expected in {"yes", "no", "true", "false"}:
        pred_tokens = pred.split()
        pred_value = pred_tokens[0] if pred_tokens else ""
        if expected in {"true", "yes"}:
            return pred_value == "yes"
        return pred_value == "no"
    expected_tokens = set(expected.split())
    return bool(expected_tokens) and expected_tokens.issubset(set(pred.split()))


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return float(ordered[index])


def citation_is_valid(citation: dict[str, Any], retrieval: list[dict[str, Any]]) -> bool:
    for hit in retrieval:
        if (
            hit.get("video_id") == citation.get("video_id")
            and int(hit.get("scene_id", -1)) == int(citation.get("scene_id", -2))
        ):
            return (
                int(hit["start_ms"]) <= int(citation["start_ms"])
                < int(citation["end_ms"]) <= int(hit["end_ms"])
                and bool(citation.get("frames", []))
            )
    return False


def run_eval(
    service: Any,
    suite_path: str | Path,
    report_path: str | Path | None = None,
    json_path: str | Path | None = None,
) -> dict[str, Any]:
    suite = load_suite(suite_path)
    details: list[dict[str, Any]] = []
    skipped = 0
    latencies: list[float] = []
    for row in suite:
        video_id = row.get("video_id")
        started = time.perf_counter()
        try:
            answer = service.ask(str(row["question"]), video_id=video_id)
        except Exception as exc:  # unavailable video/provider failures are explicit skips
            skipped += 1
            details.append(
                {
                    "query_id": row.get("query_id"),
                    "category": row["category"],
                    "skipped": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        latency = (time.perf_counter() - started) * 1000.0
        latencies.append(latency)
        answer_dict = answer.to_dict()
        retrieval = answer_dict["retrieval"]
        gold_key = (row.get("video_id"), row.get("scene_id"))
        top_key = (retrieval[0]["video_id"], retrieval[0]["scene_id"]) if retrieval else (None, None)
        hit1 = bool(gold_key[0] is not None and top_key == gold_key)
        hit5 = bool(
            gold_key[0] is not None
            and any(
                (hit["video_id"], hit["scene_id"]) == gold_key for hit in retrieval[:5]
            )
        )
        citations_valid = all(
            citation_is_valid(citation, retrieval) for citation in answer_dict["citations"]
        ) and bool(answer_dict["citations"])
        gold_window = row.get("window_ms")
        grounding_iou = None
        if row["category"] == "action" and isinstance(gold_window, list) and len(gold_window) == 2:
            grounding = answer_dict.get("grounding") or {}
            if "start_ms" in grounding:
                grounding_iou = iou(
                    int(grounding["start_ms"]),
                    int(grounding["end_ms"]),
                    int(gold_window[0]),
                    int(gold_window[1]),
                )
        matched = answers_match(answer_dict["answer"], str(row["answer"]))
        details.append(
            {
                "query_id": row.get("query_id"),
                "question": row["question"],
                "category": row["category"],
                "gold": row["answer"],
                "prediction": answer_dict["answer"],
                "match": matched,
                "hit@1": hit1,
                "hit@5": hit5,
                "citations_valid": citations_valid,
                "grounding_iou": grounding_iou,
                "latency_ms": round(latency, 3),
                "retrieval": [
                    {
                        "video_id": hit["video_id"],
                        "scene_id": hit["scene_id"],
                        "score": hit["score"],
                        "ranks": hit["ranks"],
                    }
                    for hit in retrieval
                ],
            }
        )

    evaluated = [row for row in details if "skipped" not in row]
    if not evaluated:
        raise SuiteError("no queries could be evaluated; is the video indexed?")
    categories = ("counting", "action", "descriptive")
    per_category: dict[str, dict[str, Any]] = {}
    for category in categories:
        subset = [row for row in evaluated if row["category"] == category]
        if subset:
            per_category[category] = {
                "n": len(subset),
                "accuracy": sum(1 for row in subset if row["match"]) / len(subset),
                "hit@1": sum(1 for row in subset if row["hit@1"]) / len(subset),
                "hit@5": sum(1 for row in subset if row["hit@5"]) / len(subset),
            }
        else:
            per_category[category] = {"n": 0, "accuracy": None, "hit@1": None, "hit@5": None}
    action_rows = [row for row in evaluated if row["category"] == "action"]
    ious = [
        float(row["grounding_iou"])
        for row in action_rows
        if row.get("grounding_iou") is not None
    ]
    metrics: dict[str, Any] = {
        "suite": str(suite_path),
        "queries_total": len(suite),
        "queries_evaluated": len(evaluated),
        "queries_skipped": skipped,
        "accuracy": sum(1 for row in evaluated if row["match"]) / len(evaluated),
        "per_category": per_category,
        "counting_error_rate": (
            1.0 - per_category["counting"]["accuracy"]
            if per_category["counting"]["accuracy"] is not None
            else None
        ),
        "action_error_rate": (
            1.0 - per_category["action"]["accuracy"]
            if per_category["action"]["accuracy"] is not None
            else None
        ),
        "retrieval_hit@1": sum(1 for row in evaluated if row["hit@1"]) / len(evaluated),
        "retrieval_hit@5": sum(1 for row in evaluated if row["hit@5"]) / len(evaluated),
        "citations_valid": sum(1 for row in evaluated if row["citations_valid"]) / len(evaluated),
        "grounding": {
            "n": len(ious),
            "mean_iou": statistics.fmean(ious) if ious else None,
            "iou@0.3": (sum(1 for value in ious if value >= 0.3) / len(ious) if ious else None),
            "iou@0.5": (sum(1 for value in ious if value >= 0.5) / len(ious) if ious else None),
        },
        "latency_ms": {
            "median": statistics.median(latencies) if latencies else None,
            "p95": _percentile(latencies, 0.95),
        },
    }
    report = _render_markdown(metrics)
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report, encoding="utf-8")
    if json_path is not None:
        destination = Path(json_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps({"metrics": metrics, "details": details}, indent=2),
            encoding="utf-8",
        )
    return {"metrics": metrics, "details": details, "markdown": report}


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _render_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        "# Evaluation report",
        "",
        "Measured locally on the deterministic synthetic suite. No public benchmark claims.",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Queries evaluated | {metrics['queries_evaluated']} / {metrics['queries_total']} |",
        f"| Skipped | {metrics['queries_skipped']} |",
        f"| Answer accuracy | {_pct(metrics['accuracy'])} |",
        f"| Retrieval scene hit@1 | {_pct(metrics['retrieval_hit@1'])} |",
        f"| Retrieval scene hit@5 | {_pct(metrics['retrieval_hit@5'])} |",
        f"| Valid citations | {_pct(metrics['citations_valid'])} |",
        f"| Counting error rate | {_pct(metrics['counting_error_rate'])} |",
        f"| Action error rate | {_pct(metrics['action_error_rate'])} |",
        f"| Grounding mean IoU (action) | {_num(metrics['grounding']['mean_iou'])} |",
        f"| Grounding IoU@0.3 (action) | {_pct(metrics['grounding']['iou@0.3'])} |",
        f"| Grounding IoU@0.5 (action) | {_pct(metrics['grounding']['iou@0.5'])} |",
        f"| Ask latency median | {_num(metrics['latency_ms']['median'])} ms |",
        f"| Ask latency p95 | {_num(metrics['latency_ms']['p95'])} ms |",
        "",
        "## Per-class accuracy",
        "",
        "| Class | n | Accuracy | Hit@1 | Hit@5 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for category, data in metrics["per_category"].items():
        lines.append(
            f"| {category} | {data['n']} | {_pct(data['accuracy'])} | "
            f"{_pct(data['hit@1'])} | {_pct(data['hit@5'])} |"
        )
    lines += [
        "",
        "Counting and action error rates are reported separately as hallucination proxies;",
        "they are the complement of accuracy on those subsets, not calibrated probabilities.",
        "Grounding IoU is computed only on the action subset with temporal gold windows.",
        "",
    ]
    return "\n".join(lines)
