#!/usr/bin/env python3
"""Compare two visualize_sign_grounding summaries against LeRobot ground truth.

CPU only: requires numpy, pandas and a pandas parquet engine (e.g. pyarrow).
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np
import pandas as pd


def load_summary(path: Path) -> tuple[Path, dict]:
    path = path.expanduser().resolve()
    if path.is_dir():
        candidates = sorted(path.glob("summary*.json"))
        if len(candidates) != 1:
            raise ValueError(
                f"{path}: summary JSON이 {len(candidates)}개입니다. 파일을 직접 지정하세요."
            )
        path = candidates[0]
    summary = json.loads(path.read_text())
    if not summary.get("samples"):
        raise ValueError(f"{path}: samples가 비어 있습니다.")
    return path, summary


def sample_key(sample: dict) -> tuple[int, int]:
    match = re.fullmatch(r"dataset:.*:episode=(\d+):step=(\d+)", sample["source"])
    if match is None:
        raise ValueError("GT 비교는 dataset 모드 결과만 지원합니다: " + sample["source"])
    return tuple(map(int, match.groups()))


def index_samples(summary: dict) -> dict:
    indexed = {}
    for sample in summary["samples"]:
        key = sample_key(sample)
        if key in indexed:
            raise ValueError(f"중복 샘플: episode={key[0]}, step={key[1]}")
        indexed[key] = sample
    return indexed


def box_metrics(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    """Use unclipped normalized cxcywh, matching the training loss convention."""
    if pred.shape != (4,) or target.shape != (4,):
        raise ValueError("bbox는 [cx, cy, w, h] 형식의 4개 좌표여야 합니다.")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("bbox에 NaN/Inf가 있습니다.")
    if (pred[2:] < 0).any() or (target[2:] <= 0).any():
        raise ValueError("예측 bbox 크기는 음수일 수 없고 GT-found bbox 크기는 양수여야 합니다.")
    a = np.r_[pred[:2] - pred[2:] / 2, pred[:2] + pred[2:] / 2]
    b = np.r_[target[:2] - target[2:] / 2, target[:2] + target[2:] / 2]
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = pred[2:].prod() + target[2:].prod() - intersection
    iou = intersection / max(union, 1e-6)
    enclosure = (np.maximum(a[2:], b[2:]) - np.minimum(a[:2], b[:2])).prod()
    giou_loss = 1 - iou + (enclosure - union) / max(enclosure, 1e-6)
    return float(iou), float(np.abs(pred - target).mean()), float(giou_loss)


def evaluate(samples: dict, labels: dict, found_id: int) -> pd.DataFrame:
    rows = []
    for key, sample in sorted(samples.items()):
        status, target = labels[key]
        pred = np.asarray(sample["pred_bbox_cxcywh"], dtype=float)
        probabilities = np.asarray(sample["probabilities"], dtype=float)
        if (
            probabilities.ndim != 1
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not np.isclose(probabilities.sum(), 1, atol=1e-5)
            or not 0 <= status < len(probabilities)
            or not 0 <= found_id < len(probabilities)
        ):
            raise ValueError(f"{key}: status/probabilities 형식이 잘못되었습니다.")
        prediction = int(sample["pred_status"])
        if prediction != int(probabilities.argmax()):
            raise ValueError(f"{key}: pred_status와 probabilities.argmax가 다릅니다.")
        iou, l1, giou = (
            box_metrics(pred, target) if status == found_id else (np.nan, np.nan, np.nan)
        )
        rows.append(
            dict(
                episode=key[0],
                step=key[1],
                file=sample["file"],
                goal=sample.get("source_goal", "unknown"),
                gt=status,
                pred=prediction,
                found=status == found_id,
                pred_found=prediction == found_id,
                iou=iou,
                l1=l1,
                giou=giou,
                ce=-np.log(max(probabilities[status], 1e-12)),
                prob=float(probabilities[found_id]),
                size=float(np.sqrt(target[2:].prod())) if status == found_id else np.nan,
            )
        )
    return pd.DataFrame(rows).set_index(["episode", "step"])


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def metrics(frame: pd.DataFrame) -> dict:
    positive = frame[frame.found]
    tp = int((frame.found & frame.pred_found).sum())
    fp = int((~frame.found & frame.pred_found).sum())
    fn = int((frame.found & ~frame.pred_found).sum())
    return {
        "Status accuracy ↑": frame["gt"].eq(frame.pred).mean(),
        "Found precision ↑": ratio(tp, tp + fp),
        "Found recall ↑": ratio(tp, tp + fn),
        "Found F1 ↑": ratio(2 * tp, 2 * tp + fp + fn),
        "False-positive rate ↓": ratio(fp, int((~frame.found).sum())),
        "False positives ↓": fp,
        "False negatives ↓": fn,
        "Status CE ↓": frame.ce.mean(),
        "Mean IoU ↑": positive.iou.mean(),
        "Median IoU ↑": positive.iou.median(),
        "IoU >= 0.50 ↑": ratio(int((positive.iou >= 0.5).sum()), len(positive)),
        "IoU >= 0.75 ↑": ratio(int((positive.iou >= 0.75).sum()), len(positive)),
        "Found & IoU >= 0.50 ↑": ratio(
            int((positive.pred_found & (positive.iou >= 0.5)).sum()), len(positive)
        ),
        "BBox L1 ↓": positive.l1.mean(),
        "GIoU loss ↓": positive.giou.mean(),
    }


def print_report(
    a: pd.DataFrame,
    b: pd.DataFrame,
    names: list[str],
    top: int,
    color: bool = False,
    explain: bool = False,
) -> None:
    def paint(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if color else text

    def section(title: str) -> None:
        print("\n" + paint(title, "1;36"))
        print("─" * 86)

    def width(text: str) -> int:
        return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)

    def table(headers: list[str], rows: list[list[str]]) -> None:
        widths = [max(width(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))]
        for index, row in enumerate([headers, *rows]):
            line = "  ".join(
                str(cell) + " " * (widths[i] - width(str(cell))) for i, cell in enumerate(row)
            )
            if index == 0:
                print(paint(line, "1"))
            elif row[-1] == "개선":
                print(paint(line, "32"))
            elif row[-1] == "악화":
                print(paint(line, "31"))
            else:
                print(line)
        if not rows:
            print("  해당 샘플 없음")

    def verdict(delta: float, higher: bool = True) -> str:
        if not np.isfinite(delta):
            return "평가 불가"
        if abs(delta) <= 1e-6:
            return "동일"
        return "개선" if (delta > 0) == higher else "악화"

    ma, mb = metrics(a), metrics(b)

    def sample_label(key: tuple[int, int]) -> str:
        labels = []
        for frame in (a, b):
            filename = Path(frame.loc[key, "file"]).stem
            match = re.match(r"sample_\d+(?=__|$)", filename)
            labels.append(match.group() if match else filename)
        return labels[0] if labels[0] == labels[1] else f"A:{labels[0]} / B:{labels[1]}"

    n = int(a.found.sum())
    positive = a.found
    delta = b.iou - a.iou
    success_a = positive & a.pred_found & (a.iou >= 0.5)
    success_b = positive & b.pred_found & (b.iou >= 0.5)
    print(paint("\nSIGN GROUNDING · 두 모델 비교", "1"))
    print(f"A  {names[0]}\nB  {names[1]}")
    print(f"동일 입력 {len(a)}장 · GT-found {n}장 · GT-other {len(a) - n}장")
    print("Δ = B − A · Bbox 평가: GT-found 전체 · 비율 차이: %p")

    section("핵심 지표")
    rows = []
    for label, key, kind, higher in [
        ("Detection @ IoU 0.50 ↑", "Found & IoU >= 0.50 ↑", "success", True),
        ("Mean IoU ↑", "Mean IoU ↑", "float", True),
        ("IoU ≥ 0.75 ↑", "IoU >= 0.75 ↑", "precise", True),
        ("Status accuracy ↑", "Status accuracy ↑", "percent", True),
    ]:
        x, y = ma[key], mb[key]
        if kind == "float":
            left, right, difference = f"{x:.4f}", f"{y:.4f}", f"{y - x:+.4f}"
        else:
            left, right, difference = f"{x:.1%}", f"{y:.1%}", f"{100 * (y - x):+.2f}%p"
            if kind in ("success", "precise"):
                counts = [
                    int(s.sum())
                    for s in (
                        [success_a, success_b]
                        if kind == "success"
                        else [positive & (a.iou >= 0.75), positive & (b.iou >= 0.75)]
                    )
                ]
                left += f" ({counts[0]}/{n})"
                right += f" ({counts[1]}/{n})"
        rows.append([label, left, right, difference, verdict(y - x, higher)])
    for label, key in [
        ("False positives ↓", "False positives ↓"),
        ("False negatives ↓", "False negatives ↓"),
    ]:
        x, y = ma[key], mb[key]
        rows.append([label, f"{x}장", f"{y}장", f"{y - x:+d}장", verdict(y - x, False)])
    table(["지표", "A", "B", "Δ", "변화"], rows)

    section("샘플별 변화")
    wins, losses, ties = [
        (int(mask.sum()))
        for mask in [delta[positive] > 1e-6, delta[positive] < -1e-6, delta[positive].abs() <= 1e-6]
    ]
    print(f"IoU 개선 / 악화 / 동일: {wins} / {losses} / {ties}")
    print(
        f"검출 전환: 실패→성공 {int((~success_a & success_b).sum())}장 / 성공→실패 {int((success_a & ~success_b).sum())}장"
    )
    print(f"Status 변경: {int(a.pred.ne(b.pred).sum())}/{len(a)}장")
    if n > 1 and delta[positive].abs().max() > 1e-6:
        key = delta[positive].abs().idxmax()
        print(f"최대 |ΔIoU|: {sample_label(key)} (ΔIoU {delta.loc[key]:+.4f})")
        print(
            f"평균 ΔIoU: {delta[positive].mean():+.4f} | 최대 변화 샘플 제외: {delta[positive].drop(key).mean():+.4f}"
        )

    section("구간별 IoU (|Δ| 내림차순)")
    for group, title in [("goal", "Goal별"), ("size_group", "GT 크기별")]:
        parts = []
        for frame in [a, b]:
            part = frame[frame.found].copy()
            part["size_group"] = pd.cut(
                part["size"],
                [0, 0.05, 0.1, np.inf],
                labels=["작음 ≤0.05", "중간 0.05–0.10", "큼 >0.10"],
            )
            parts.append(part.groupby(group, observed=True).iou.agg(["size", "mean"]))
        diff = parts[1]["mean"] - parts[0]["mean"]
        print(f"\n  {title}")
        rows = []
        for key in diff.abs().sort_values(ascending=False).index:
            x, y = parts[0].loc[key, "mean"], parts[1].loc[key, "mean"]
            rows.append(
                [
                    str(key),
                    str(int(parts[0].loc[key, "size"])),
                    f"{x:.4f}",
                    f"{y:.4f}",
                    f"{y - x:+.4f}",
                    verdict(y - x),
                ]
            )
        table(["구간", "N", "A IoU", "B IoU", "Δ", "변화"], rows)
    print("  크기 = √(GT 정규화 width × height)")

    section(f"주요 변화 샘플 (각 {top}개)")
    for mask, ascending, title in [
        (positive & (delta > 1e-6), False, "개선"),
        (positive & (delta < -1e-6), True, "악화"),
    ]:
        print(f"\n  {title}")
        rows = []
        for key in delta[mask].sort_values(ascending=ascending).head(top).index:
            rows.append(
                [
                    sample_label(key),
                    f"{a.loc[key, 'iou']:.4f}",
                    f"{b.loc[key, 'iou']:.4f}",
                    f"{delta.loc[key]:+.4f}",
                    title,
                ]
            )
        table(
            ["Sample", "A IoU", "B IoU", "Δ", "변화"],
            rows,
        )

    section("보조 지표")
    rows = []
    percent_keys = {
        "Found precision ↑",
        "Found recall ↑",
        "Found F1 ↑",
        "False-positive rate ↓",
        "IoU >= 0.50 ↑",
    }
    for key in [
        "Found precision ↑",
        "Found recall ↑",
        "Found F1 ↑",
        "False-positive rate ↓",
        "IoU >= 0.50 ↑",
        "Median IoU ↑",
        "Status CE ↓",
        "BBox L1 ↓",
        "GIoU loss ↓",
    ]:
        x, y = ma[key], mb[key]
        if key in percent_keys:
            left, right, difference = f"{x:.2%}", f"{y:.2%}", f"{100 * (y - x):+.2f}%p"
        else:
            left, right, difference = f"{x:.6f}", f"{y:.6f}", f"{y - x:+.6f}"
        rows.append([key, left, right, difference, verdict(y - x, key.endswith("↑"))])
    table(["지표", "A", "B", "Δ", "변화"], rows)
    if explain:
        section("지표 정의")
        print("Detection @ IoU 0.50 : GT-found 중 found 예측 및 IoU ≥ 0.5를 만족한 비율")
        print("IoU / Mean / Median : 박스 교집합÷합집합 / 평균 / 중앙값")
        print("IoU ≥ 0.50 / 0.75   : GT-found 중 IoU 기준 통과 비율 (status 무관, AP 아님)")
        print("Status accuracy     : 전체 샘플의 status 클래스 정확도")
        print("Precision / Recall  : found 예측의 정답률 / GT-found 검출률")
        print("F1                  : Precision과 Recall의 조화평균")
        print("False positives     : GT-other를 found로 예측한 수 (ambiguous 포함)")
        print("False negatives     : GT-found를 found 이외로 예측한 수")
        print("False-positive rate : GT-other 중 오탐 비율")
        print("Status CE           : GT 클래스 확률의 -log 평균 (확률 하한 1e-12)")
        print("BBox L1             : 정규화 cx,cy,w,h 절대오차 평균")
        print("GIoU loss           : 1-GIoU; 비중첩 박스의 포괄 영역도 반영")
        print("Bbox 지표: 미검출 포함 GT-found 전체, 좌표 clip 없음, 손실 가중치 미적용.")
        print("NaN: 분모 없음. 표본 내 차이는 통계적 유의성 또는 주행 성능을 의미하지 않음.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_a", type=Path, help="A 결과 폴더 또는 summary JSON")
    parser.add_argument("result_b", type=Path, help="B 결과 폴더 또는 summary JSON")
    parser.add_argument(
        "--dataset", type=Path, help="GT 데이터셋 경로. 생략하면 summary.dataset 사용"
    )
    parser.add_argument("--name-a", default="기존")
    parser.add_argument("--name-b", default="비교 모델")
    parser.add_argument("--found-status-id", type=int, default=1)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="터미널 색상 (기본: auto, NO_COLOR 지원)",
    )
    parser.add_argument("--details", action="store_true", help="원본 경로와 status 혼동행렬도 출력")
    parser.add_argument("--explain", action="store_true", help="지표 정의 출력")
    parser.add_argument("--web", action="store_true", help="A/B 이미지 비교 웹 프로그램 실행")
    parser.add_argument("--host", default="127.0.0.1", help="웹 서버 주소 (기본: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7865, help="웹 서버 포트 (기본: 7865)")
    args = parser.parse_args()
    try:
        if args.top < 1:
            raise ValueError("--top은 1 이상이어야 합니다.")
        paths, summaries = zip(load_summary(args.result_a), load_summary(args.result_b))
        samples_a, samples_b = map(index_samples, summaries)
        if samples_a.keys() != samples_b.keys():
            raise ValueError("두 결과의 episode/step 집합이 다릅니다. 동일 샘플로 재평가하세요.")
        for key in samples_a:
            for field in ["prompt", "sign_query_index", "source_goal", "model_input_keys"]:
                if samples_a[key].get(field) != samples_b[key].get(field):
                    raise ValueError(f"{key}: {field}가 다릅니다. 동일 입력 비교가 아닙니다.")
        if args.dataset is None:
            if not summaries[0].get("dataset") or summaries[0].get("dataset") != summaries[1].get(
                "dataset"
            ):
                raise ValueError(
                    "summary의 dataset 경로가 없거나 다릅니다. --dataset으로 GT를 지정하세요."
                )
            args.dataset = Path(summaries[0]["dataset"])
        dataset = args.dataset.expanduser().resolve()
        labels = {}
        for episode in sorted({key[0] for key in samples_a}):
            candidates = list((dataset / "data").rglob(f"episode_{episode:06d}.parquet"))
            if len(candidates) != 1:
                raise ValueError(f"episode {episode}: parquet 파일이 {len(candidates)}개입니다.")
            frame = pd.read_parquet(
                candidates[0], columns=["gt_sign_status", "gt_sign_bbox_cxcywh"]
            )
            for key in samples_a:
                if key[0] == episode:
                    if not 0 <= key[1] < len(frame):
                        raise ValueError(f"{key}: GT frame 범위를 벗어났습니다.")
                    row = frame.iloc[key[1]]
                    labels[key] = (
                        int(np.asarray(row.gt_sign_status).item()),
                        np.asarray(row.gt_sign_bbox_cxcywh, dtype=float),
                    )
        if args.web:
            from sign_comparison_viewer import serve_comparison

            a = evaluate(samples_a, labels, args.found_status_id)
            b = evaluate(samples_b, labels, args.found_status_id)
            report = io.StringIO()
            with redirect_stdout(report):
                print_report(a, b, [args.name_a, args.name_b], args.top, explain=args.explain)
            serve_comparison(
                a,
                b,
                paths,
                [args.name_a, args.name_b],
                report.getvalue(),
                args.top,
                args.host,
                args.port,
            )
            return
        print_report(
            evaluate(samples_a, labels, args.found_status_id),
            evaluate(samples_b, labels, args.found_status_id),
            [args.name_a, args.name_b],
            args.top,
            explain=args.explain,
            color=args.color == "always"
            or (args.color == "auto" and sys.stdout.isatty() and "NO_COLOR" not in os.environ),
        )
        if args.details:
            print("\n[상세 정보]")
            print(f"GT dataset: {dataset} | found status ID: {args.found_status_id}")
            for label, path, summary, samples in zip(
                ["A", "B"], paths, summaries, [samples_a, samples_b]
            ):
                print(
                    f"{label} summary: {path}\n{label} checkpoint: {summary.get('checkpoint', 'unknown')}"
                )
                frame = evaluate(samples, labels, args.found_status_id)
                print(f"{label} 혼동행렬: 행=GT, 열=예측")
                print(pd.crosstab(frame["gt"], frame.pred).to_string())
    except (ValueError, KeyError, OSError, ImportError) as exc:
        parser.exit(2, f"오류: {exc}\n")


if __name__ == "__main__":
    main()


"""
uv run python compare_sign_grounding.py \
  ./visualization_sign_head_test_1_5_1 \
  ./visualization_sign_head_test_1_5_1+bbox_detached \
  --name-a 1_5_1 \
  --name-b 1_5_1+bbox_detached
"""
