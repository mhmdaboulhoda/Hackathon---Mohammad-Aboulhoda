#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flask import Flask, flash, redirect, render_template, request, url_for

import joblib
import pandas as pd

MODEL_PATH = Path("models/best_model.joblib")
DATASET_PATH = Path("Breast_cancer_dataset.csv")

app = Flask(__name__)
app.secret_key = "mhmd"


@dataclass
class FeatureMeta:
    name: str
    label: str
    min_val: float
    max_val: float
    is_numeric: bool

class ValidationError(ValueError):
    def __init__(self, field: str, message: str) -> None:
        super().__init__(message)
        self.field = field


class ModelService:
    def __init__(self, model_path: Path, dataset_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file '{model_path}' not found. Train the model before running the app."
            )
        self.pipeline = joblib.load(model_path)
        self.features_meta = self._build_feature_metadata(dataset_path)

    def _build_feature_metadata(self, dataset_path: Path) -> List[FeatureMeta]:
        df = pd.read_csv(dataset_path)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        feature_cols = [col for col in df.columns if col != "diagnosis"]

        metas: List[FeatureMeta] = []
        for col in feature_cols:
            series = df[col]
            is_numeric = pd.api.types.is_numeric_dtype(series)
            min_val = float(series.min()) if is_numeric else 0.0
            max_val = float(series.max()) if is_numeric else 0.0
            metas.append(
                FeatureMeta(
                    name=col,
                    label=col.replace("_", " ").title(),
                    min_val=min_val,
                    max_val=max_val,
                    is_numeric=is_numeric,
                )
            )
        return metas

    def predict(self, form_data: Dict[str, str]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        input_dict = {}
        for meta in self.features_meta:
            raw = form_data.get(meta.name)
            if raw is None or raw.strip() == "":
                raise ValidationError(meta.name, f"{meta.label} is required.")
            if meta.is_numeric:
                try:
                    value = float(raw)
                except ValueError:
                    raise ValidationError(meta.name, f"{meta.label} must be a number.")
                if value <= 0:
                    raise ValidationError(meta.name, f"{meta.label} must be a positive number.")
                input_dict[meta.name] = value
            else:
                input_dict[meta.name] = raw.strip()
        df = pd.DataFrame([input_dict])
        proba = self.pipeline.predict_proba(df)[0, 1]
        label = self.pipeline.predict(df)[0]
        return (
            {"probability": float(proba), "label": "Malignant" if label == 1 else "Benign"},
            input_dict,
        )

_service: ModelService | None = None
patient_records: List[Dict[str, Any]] = []


def _parse_filter(expr: str):
    expr = expr.replace(" ", "")
    for op in (">=", "<=", "==", ">", "<"):
        if expr.startswith(op):
            value = expr[len(op):]
            if value == "":
                return None
            return op, value
    return None


def filter_records(
    records: List[Dict[str, Any]],
    filters: Dict[str, str],
    features: List[FeatureMeta],
) -> List[Dict[str, Any]]:
    if not records:
        return []
    df = pd.DataFrame(records)
    mask = pd.Series([True] * len(df), index=df.index)
    for meta in features:
        expr = filters.get(f"filter_{meta.name}", "").strip()
        if not expr:
            continue
        parsed = _parse_filter(expr)
        if parsed is None:
            raise ValueError(f"Invalid filter expression '{expr}' for {meta.label}")
        op, value = parsed
        series = df[meta.name]
        if meta.is_numeric:
            try:
                num_value = float(value)
            except ValueError:
                raise ValueError(f"Filter value for {meta.label} must be numeric.")
            comparator = num_value
        else:
            if op != "==":
                raise ValueError(f"Only equality filtering is supported for {meta.label}.")
            comparator = value

        if meta.is_numeric:
            series = pd.to_numeric(series, errors="coerce")
            if op == ">=":
                mask &= series >= comparator
            elif op == "<=":
                mask &= series <= comparator
            elif op == ">":
                mask &= series > comparator
            elif op == "<":
                mask &= series < comparator
            elif op == "==":
                mask &= series == comparator
            else:
                raise ValueError(f"Unsupported operator '{op}' for {meta.label}")
        else:
            mask &= series.astype(str) == comparator
    filtered = df.loc[mask]
    return filtered.to_dict(orient="records")


def get_service() -> ModelService:
    global _service
    if _service is None:
        _service = ModelService(MODEL_PATH, DATASET_PATH)
    return _service


@app.route("/", methods=["GET", "POST"])
def index():
    active_tab = request.args.get("tab", "diagnosis")
    try:
        svc = get_service()
    except FileNotFoundError as exc:
        flash(str(exc), "danger")
        return render_template(
            "index.html",
            features=[],
            prediction=None,
            records=[],
            total_records=0,
            active_tab=active_tab,
            diagnosis_values={},
            diagnosis_errors={},
            filter_values={},
        )

    prediction = None
    filtered_records = list(patient_records)

    diagnosis_values: Dict[str, str] = {}
    diagnosis_errors: Dict[str, str] = {}
    filter_values: Dict[str, str] = {}

    if request.method == "POST":
        active_tab = request.form.get("tab", "diagnosis")
        feature_names = {meta.name for meta in svc.features_meta}
        if active_tab == "diagnosis":
            diagnosis_values = {k: v for k, v in request.form.items() if k in feature_names}
        elif active_tab == "records":
            filter_values = {k: v for k, v in request.form.items() if k.startswith("filter_")}
        try:
            if active_tab == "diagnosis" and "predict" in request.form:
                prediction, cleaned_inputs = svc.predict(request.form)
                entry = {
                    **cleaned_inputs,
                    "prediction": prediction["label"],
                    "probability": prediction["probability"],
                }
                patient_records.append(entry)
                diagnosis_values = {k: str(v) for k, v in cleaned_inputs.items()}
                flash("Prediction completed and record saved.", "success")
            elif active_tab == "records" and "filter" in request.form:
                filtered_records = filter_records(patient_records, request.form, svc.features_meta)
                flash(
                    f"Filters applied. Showing {len(filtered_records)} of {len(patient_records)} saved records.",
                    "info",
                )
        except ValidationError as err:
            flash(str(err), "danger")
            diagnosis_errors[err.field] = str(err)
            if not diagnosis_values:
                diagnosis_values = {k: v for k, v in request.form.items() if k in feature_names}
        except ValueError as err:
            flash(str(err), "danger")
            if active_tab == "records":
                filtered_records = list(patient_records)

    if request.method == "GET" and active_tab == "records":
        filtered_records = list(patient_records)

    return render_template(
        "index.html",
        features=svc.features_meta,
        prediction=prediction,
        records=filtered_records,
        total_records=len(patient_records),
        active_tab=active_tab,
        diagnosis_values=diagnosis_values,
        diagnosis_errors=diagnosis_errors,
        filter_values=filter_values,
    )


if __name__ == "__main__":
    app.run(debug=True)
