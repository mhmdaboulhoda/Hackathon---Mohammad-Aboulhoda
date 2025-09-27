#!/usr/bin/env python3
"""Train multiple ML models with grid search to find the best performer."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

try:
    import joblib
    import pandas as pd
except ImportError as exc:  
    missing_pkg = "joblib" if "joblib" in str(exc) else "pandas"
    sys.stderr.write(
        f"Missing dependency: {missing_pkg}. Install with `python3 -m pip install {missing_pkg}` and retry.\n"
    )
    raise

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
except ImportError as exc:  
    sys.stderr.write(
        "Missing scikit-learn components. Install with `python3 -m pip install scikit-learn`.\n"
    )
    raise

@dataclass
class ModelResult:
    model_name: str
    best_params: Dict[str, object]
    cv_score: float
    test_accuracy: float
    test_f1: float
    test_roc_auc: float

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["best_params"] = dict(self.best_params)
        return data


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    if "diagnosis" not in df.columns:
        raise ValueError("Dataset must contain a 'diagnosis' column as the target label.")
    return df


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = features.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("No feature columns detected.")

    return ColumnTransformer(transformers=transformers)


def define_model_grids() -> List[Dict[str, object]]:
    return [
        {
            "name": "logistic_regression",
            "estimator": LogisticRegression(max_iter=5000),
            "param_grid": {
                "classifier__penalty": ["l1", "l2"],
                "classifier__C": [0.01, 0.1, 1, 10],
                "classifier__solver": ["liblinear"],
                "classifier__class_weight": [None, "balanced"],
            },
        },
        {
            "name": "random_forest",
            "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
            "param_grid": {
                "classifier__n_estimators": [100, 200, 400],
                "classifier__max_depth": [None, 5, 10, 20],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__class_weight": [None, "balanced"],
            },
        },
        {
            "name": "gradient_boosting",
            "estimator": GradientBoostingClassifier(random_state=42),
            "param_grid": {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
                "classifier__max_depth": [1, 2, 3],
            },
        },
        {
            "name": "support_vector_machine",
            "estimator": SVC(probability=True),
            "param_grid": {
                "classifier__kernel": ["linear", "rbf", "poly"],
                "classifier__C": [0.1, 1, 10],
                "classifier__gamma": ["scale", "auto"],
                "classifier__degree": [2, 3],
                "classifier__class_weight": [None, "balanced"],
            },
        },
        {
            "name": "k_nearest_neighbors",
            "estimator": KNeighborsClassifier(),
            "param_grid": {
                "classifier__n_neighbors": [3, 5, 7, 9],
                "classifier__weights": ["uniform", "distance"],
                "classifier__p": [1, 2],
            },
        },
    ]


def fit_and_evaluate(
    df: pd.DataFrame,
    output_json: Optional[Path],
    model_path: Optional[Path],
    scoring: str = "roc_auc",
    test_size: float = 0.25,
    random_state: int = 42,
    cv_folds: int = 5,
) -> List[ModelResult]:
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"].map({"M": 1, "B": 0}).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(X_train)
    model_grids = define_model_grids()

    results: List[ModelResult] = []
    for spec in model_grids:
        name = spec["name"]
        estimator = spec["estimator"]
        param_grid = spec["param_grid"]

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", estimator)])

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv_folds,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        result = ModelResult(
            model_name=name,
            best_params=search.best_params_,
            cv_score=search.best_score_,
            test_accuracy=accuracy_score(y_test, y_pred),
            test_f1=f1_score(y_test, y_pred),
            test_roc_auc=roc_auc_score(y_test, y_proba),
        )
        results.append(result)
        print(f"Model {name} -- best {scoring}: {search.best_score_:.4f}")
        print(f"  Test Acc: {result.test_accuracy:.4f} | F1: {result.test_f1:.4f} | ROC-AUC: {result.test_roc_auc:.4f}")

    results.sort(key=lambda r: r.test_roc_auc, reverse=True)
    best = results[0]
    print("\nBest model on test ROC-AUC:")
    print(f"  {best.model_name} with ROC-AUC={best.test_roc_auc:.4f}")

    if output_json:
        payload = {
            "best_model": results[0].to_dict(),
            "all_results": [r.to_dict() for r in results],
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Detailed results written to {output_json}")

    if model_path:
        best_spec = next(spec for spec in define_model_grids() if spec["name"] == best.model_name)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X)),
                ("classifier", best_spec["estimator"]),
            ]
        )
        pipeline.set_params(**best.best_params)
        pipeline.fit(X, y)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)
        print(f"Best model pipeline saved to {model_path}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multiple models with exhaustive grid search")
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the Breast Cancer dataset CSV",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/model_selection_results.json"),
        help="Where to save the summary of model search results (default: reports/model_selection_results.json)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/best_model.joblib"),
        help="Where to persist the best model pipeline (default: models/best_model.joblib)",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="roc_auc",
        help="Scikit-learn scoring metric for GridSearchCV (default: roc_auc)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of data to reserve for the hold-out test set (default: 0.25)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = fit_and_evaluate(
        df=load_dataset(args.csv_path),
        output_json=args.output_json,
        model_path=args.model_path,
        scoring=args.scoring,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
    )
    _ = results


if __name__ == "__main__":
    main()
