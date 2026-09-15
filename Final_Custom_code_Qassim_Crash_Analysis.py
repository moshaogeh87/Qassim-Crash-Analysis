from __future__ import annotations

# 1. Importing Essential Libraries

import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier


# 2. Configuration

@dataclass
class Config:
    output_dir: str = "analysis_outputs"
    test_size: float = 0.30
    random_state: int = 42
    cv_folds: int = 5
    bayes_initial_points: int = 6
    bayes_iterations: int = 18
    smote_k_neighbors: int = 5
    shap_max_display: int = 12
    shap_sample_size: int = 500


CFG = Config()


# 3. Data Loading and Reading

data = pd.read_csv("Qassim_Crash_Data.csv")


# 4. Data Cleaning and Preprocessing

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop_duplicates().copy()
    out = out.dropna().copy()
    return out.reset_index(drop=True)


# 5. Target and Explanatory Variable Definition

Y = "Accident Severity Category"

Y_ENCODING = {
    "Non-Fatal Injury": 0,
    "Fatal Injury": 1,
}

X = [
    "Lighting Conditions",
    "Period of the Day ",
    "Weekday",
    "Road Type",
    "Road Status",
    "Weather Status",
    "Road Alignment Details",
    "Damage Type",
    "Accident Type",
    "Vehicle Type",
    "Accident Cause",
    "Number of Vehicles Involved",
]


def define_xy(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    required = X + [Y]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df.copy()
    data[Y] = (
        data[Y]
        .astype(str)
        .str.strip()
        .replace({"Non Ftal Injury": "Non-Fatal Injury"})
    )

    x = data[X].copy()
    y = data[Y].map(Y_ENCODING)

    if y.isna().any():
        labels = sorted(data.loc[y.isna(), Y].astype(str).unique())
        raise ValueError(f"Unrecognized outcome labels: {labels}")

    return x, y.astype(int)


# 6. Data Encoding and Processing

def make_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    categorical = list(x_train.columns)
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def encode_data(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> Tuple[ColumnTransformer, np.ndarray, np.ndarray, List[str]]:
    preprocessor = make_preprocessor(x_train)
    x_train_encoded = preprocessor.fit_transform(x_train)
    x_test_encoded = preprocessor.transform(x_test)
    feature_names = list(preprocessor.get_feature_names_out())
    return preprocessor, x_train_encoded, x_test_encoded, feature_names


# 7. Stratified Training-Test Data Split

def split_data(
    x: pd.DataFrame,
    y: pd.Series,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        x,
        y,
        test_size=cfg.test_size,
        stratify=y,
        random_state=cfg.random_state,
    )


# 8. SMOTE Application to Training Data

def apply_smote(
    x_train_encoded: np.ndarray,
    y_train: pd.Series,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    smote = SMOTE(
        sampling_strategy="auto",
        k_neighbors=cfg.smote_k_neighbors,
        random_state=cfg.random_state,
    )
    return smote.fit_resample(x_train_encoded, y_train)


# 9. Hyperparameter Optimization

def make_model(model_name: str, params: Dict[str, Any], random_state: int):
    if model_name == "LR":
        return LogisticRegression(
            penalty=params["penalty"],
            C=float(params["C"]),
            max_iter=int(round(params["max_iter"])),
            solver=params["solver"],
            random_state=random_state,
        )

    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=int(round(params["n_estimators"])),
            min_samples_leaf=int(round(params["min_samples_leaf"])),
            min_samples_split=int(round(params["min_samples_split"])),
            max_depth=int(round(params["max_depth"])),
            max_features=params["max_features"],
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "XGB":
        return XGBClassifier(
            learning_rate=float(params["learning_rate"]),
            max_depth=int(round(params["max_depth"])),
            n_estimators=int(round(params["n_estimators"])),
            subsample=float(params["subsample"]),
            min_child_weight=int(round(params["min_child_weight"])),
            gamma=float(params["gamma"]),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )

    raise ValueError(f"Unknown model: {model_name}")


SEARCH_SPACES = {
    "LR": {
        "C": (0.01, 100.0, "log"),
        "max_iter": (100, 1500, "int"),
        "penalty_solver": [("l2", "lbfgs"), ("l1", "liblinear")],
    },
    "RF": {
        "n_estimators": (100, 1000, "int"),
        "min_samples_leaf": (1, 10, "int"),
        "min_samples_split": (2, 20, "int"),
        "max_depth": (5, 30, "int"),
        "max_features": ["sqrt", "log2"],
    },
    "XGB": {
        "learning_rate": (0.01, 0.30, "linear"),
        "max_depth": (3, 10, "int"),
        "n_estimators": (100, 1000, "int"),
        "subsample": (0.50, 1.00, "linear"),
        "min_child_weight": (1, 10, "int"),
        "gamma": (0.0, 5.0, "linear"),
    },
}


def sample_random_parameters(
    model_name: str,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    if model_name == "LR":
        c_low, c_high, _ = SEARCH_SPACES["LR"]["C"]
        c_value = float(
            np.exp(rng.uniform(np.log(c_low), np.log(c_high)))
        )
        max_iter = int(rng.integers(100, 1501))
        penalty, solver = SEARCH_SPACES["LR"]["penalty_solver"][
            rng.integers(0, len(SEARCH_SPACES["LR"]["penalty_solver"]))
        ]
        return {
            "C": c_value,
            "max_iter": max_iter,
            "penalty": penalty,
            "solver": solver,
        }

    if model_name == "RF":
        return {
            "n_estimators": int(rng.integers(100, 1001)),
            "min_samples_leaf": int(rng.integers(1, 11)),
            "min_samples_split": int(rng.integers(2, 21)),
            "max_depth": int(rng.integers(5, 31)),
            "max_features": str(
                SEARCH_SPACES["RF"]["max_features"][
                    rng.integers(0, len(SEARCH_SPACES["RF"]["max_features"]))
                ]
            ),
        }

    if model_name == "XGB":
        return {
            "learning_rate": float(rng.uniform(0.01, 0.30)),
            "max_depth": int(rng.integers(3, 11)),
            "n_estimators": int(rng.integers(100, 1001)),
            "subsample": float(rng.uniform(0.50, 1.00)),
            "min_child_weight": int(rng.integers(1, 11)),
            "gamma": float(rng.uniform(0.0, 5.0)),
        }

    raise ValueError(model_name)


def encode_parameters(model_name: str, params: Dict[str, Any]) -> np.ndarray:
    if model_name == "LR":
        return np.array(
            [
                np.log10(float(params["C"])),
                float(params["max_iter"]),
                1.0 if params["penalty"] == "l1" else 0.0,
            ],
            dtype=float,
        )

    if model_name == "RF":
        return np.array(
            [
                float(params["n_estimators"]),
                float(params["min_samples_leaf"]),
                float(params["min_samples_split"]),
                float(params["max_depth"]),
                1.0 if params["max_features"] == "log2" else 0.0,
            ],
            dtype=float,
        )

    if model_name == "XGB":
        return np.array(
            [
                float(params["learning_rate"]),
                float(params["max_depth"]),
                float(params["n_estimators"]),
                float(params["subsample"]),
                float(params["min_child_weight"]),
                float(params["gamma"]),
            ],
            dtype=float,
        )

    raise ValueError(model_name)


def expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    best_value: float,
    xi: float = 0.01,
) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-12)
    improvement = mu - best_value - xi
    z = improvement / sigma
    normal_pdf = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    normal_cdf = 0.5 * (
        1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0))
    )
    return improvement * normal_cdf + sigma * normal_pdf


def cross_validated_score(
    model_name: str,
    params: Dict[str, Any],
    x_train_balanced: np.ndarray,
    y_train_balanced: np.ndarray,
    cfg: Config,
) -> float:
    cv = StratifiedKFold(
        n_splits=cfg.cv_folds,
        shuffle=True,
        random_state=cfg.random_state,
    )
    scores = []

    for train_idx, valid_idx in cv.split(
        x_train_balanced,
        y_train_balanced,
    ):
        model = make_model(
            model_name,
            params,
            cfg.random_state,
        )
        model.fit(
            x_train_balanced[train_idx],
            y_train_balanced[train_idx],
        )
        probabilities = model.predict_proba(
            x_train_balanced[valid_idx]
        )[:, 1]
        scores.append(
            roc_auc_score(
                y_train_balanced[valid_idx],
                probabilities,
            )
        )

    return float(np.mean(scores))


def bayesian_optimize(
    model_name: str,
    x_train_balanced: np.ndarray,
    y_train_balanced: np.ndarray,
    cfg: Config,
    outdir: Path,
) -> Tuple[Dict[str, Any], float]:
    offsets = {"LR": 11, "RF": 23, "XGB": 37}
    rng = np.random.default_rng(
        cfg.random_state + offsets[model_name]
    )

    candidates: List[Dict[str, Any]] = []
    scores: List[float] = []

    for _ in range(cfg.bayes_initial_points):
        params = sample_random_parameters(model_name, rng)
        score = cross_validated_score(
            model_name,
            params,
            x_train_balanced,
            y_train_balanced,
            cfg,
        )
        candidates.append(params)
        scores.append(score)

    x_observed = np.vstack(
        [encode_parameters(model_name, p) for p in candidates]
    )
    y_observed = np.asarray(scores, dtype=float)

    for _ in range(cfg.bayes_iterations):
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=1.0, nu=2.5)
        )

        gaussian_process = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=cfg.random_state,
            n_restarts_optimizer=1,
        )
        gaussian_process.fit(x_observed, y_observed)

        pool = [
            sample_random_parameters(model_name, rng)
            for _ in range(500)
        ]
        x_pool = np.vstack(
            [encode_parameters(model_name, p) for p in pool]
        )

        mu, sigma = gaussian_process.predict(
            x_pool,
            return_std=True,
        )

        acquisition = expected_improvement(
            mu,
            sigma,
            float(y_observed.max()),
        )

        next_index = int(np.argmax(acquisition))
        next_params = pool[next_index]
        next_score = cross_validated_score(
            model_name,
            next_params,
            x_train_balanced,
            y_train_balanced,
            cfg,
        )

        candidates.append(next_params)
        scores.append(next_score)
        x_observed = np.vstack(
            [
                x_observed,
                encode_parameters(
                    model_name,
                    next_params,
                ),
            ]
        )
        y_observed = np.append(
            y_observed,
            next_score,
        )

    search_records = []
    for iteration, (params, score) in enumerate(
        zip(candidates, scores),
        start=1,
    ):
        search_records.append(
            {
                "Iteration": iteration,
                "CV_AUC": score,
                **params,
            }
        )

    search_table = pd.DataFrame(search_records).sort_values(
        "CV_AUC",
        ascending=False,
    )
    search_table.to_csv(
        outdir / f"hyperparameter_search_{model_name.lower()}.csv",
        index=False,
    )

    best_index = int(np.argmax(scores))
    return candidates[best_index], float(scores[best_index])


# 10. Retrieve Best Parameters

def optimize_models(
    x_train_balanced: np.ndarray,
    y_train_balanced: np.ndarray,
    cfg: Config,
    outdir: Path,
) -> Dict[str, Dict[str, Any]]:
    optimized_parameters: Dict[str, Dict[str, Any]] = {}
    optimization_summary = []

    for model_name in ["LR", "RF", "XGB"]:
        best_params, best_score = bayesian_optimize(
            model_name,
            x_train_balanced,
            y_train_balanced,
            cfg,
            outdir,
        )
        optimized_parameters[model_name] = best_params
        optimization_summary.append(
            {
                "Model": model_name,
                "Best_CV_AUC": best_score,
                **best_params,
            }
        )

    pd.DataFrame(optimization_summary).to_csv(
        outdir / "optimized_parameters.csv",
        index=False,
    )

    (outdir / "optimized_parameters.json").write_text(
        json.dumps(
            optimized_parameters,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return optimized_parameters


# 11. Construct Final LR, RF and XGBoost Models

def construct_final_models(
    optimized_parameters: Dict[str, Dict[str, Any]],
    cfg: Config,
) -> Dict[str, Any]:
    return {
        "Logistic Regression": make_model(
            "LR",
            optimized_parameters["LR"],
            cfg.random_state,
        ),
        "Random Forest": make_model(
            "RF",
            optimized_parameters["RF"],
            cfg.random_state,
        ),
        "XGBoost": make_model(
            "XGB",
            optimized_parameters["XGB"],
            cfg.random_state,
        ),
    }


# 12. Fit Final Models to Training Data

def fit_models(
    models: Dict[str, Any],
    x_train_balanced: np.ndarray,
    y_train_balanced: np.ndarray,
) -> Dict[str, Any]:
    for model in models.values():
        model.fit(
            x_train_balanced,
            y_train_balanced,
        )
    return models


# 13. Performance Evaluation on Independent Test Data

def evaluate_models(
    models: Dict[str, Any],
    x_test_encoded: np.ndarray,
    y_test: pd.Series,
) -> Dict[str, Dict[str, Any]]:
    results = {}
    y_true = y_test.to_numpy(dtype=int)

    for name, model in models.items():
        predictions = model.predict(
            x_test_encoded
        ).astype(int)
        probabilities = model.predict_proba(
            x_test_encoded
        )[:, 1]

        results[name] = {
            "model": model,
            "predictions": predictions,
            "probabilities": probabilities,
            "metrics": {
                "Accuracy": accuracy_score(
                    y_true,
                    predictions,
                ),
                "Precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "F1": f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "AUC": roc_auc_score(
                    y_true,
                    probabilities,
                ),
            },
        }

    return results


# 14. Confusion Matrix Analysis

def save_confusion_matrices(
    results: Dict[str, Dict[str, Any]],
    y_test: pd.Series,
    outdir: Path,
) -> None:
    rows = []
    y_true = y_test.to_numpy(dtype=int)

    for name, result in results.items():
        cm = confusion_matrix(
            y_true,
            result["predictions"],
            labels=[1, 0],
        )

        rows.append(
            {
                "Model": name,
                "Fatal_Actual_Fatal_Predicted": int(cm[0, 0]),
                "Fatal_Actual_NonFatal_Predicted": int(cm[0, 1]),
                "NonFatal_Actual_Fatal_Predicted": int(cm[1, 0]),
                "NonFatal_Actual_NonFatal_Predicted": int(cm[1, 1]),
                "Test_N": int(cm.sum()),
            }
        )

        fig, ax = plt.subplots(
            figsize=(5.8, 4.8)
        )
        image = ax.imshow(cm)
        ax.set_xticks(
            [0, 1],
            ["Fatal", "Non-Fatal"],
        )
        ax.set_yticks(
            [0, 1],
            ["Fatal", "Non-Fatal"],
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix - {name}")

        for i in range(2):
            for j in range(2):
                row_total = cm[i].sum()
                percentage = (
                    100.0 * cm[i, j] / row_total
                    if row_total
                    else 0.0
                )
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}\n({percentage:.2f}%)",
                    ha="center",
                    va="center",
                )

        fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )
        fig.tight_layout()

        file_name = {
            "Logistic Regression": "confusion_lr.png",
            "Random Forest": "confusion_rf.png",
            "XGBoost": "confusion_xgb.png",
        }[name]

        fig.savefig(
            outdir / file_name,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    pd.DataFrame(rows).to_csv(
        outdir / "confusion_matrices.csv",
        index=False,
    )


# 15. ROC and AUC Analysis

def save_performance_table(
    results: Dict[str, Dict[str, Any]],
    outdir: Path,
) -> None:
    rows = [
        {
            "Model": name,
            **result["metrics"],
        }
        for name, result in results.items()
    ]
    pd.DataFrame(rows).to_csv(
        outdir / "model_metrics.csv",
        index=False,
    )


def plot_roc(
    results: Dict[str, Dict[str, Any]],
    y_test: pd.Series,
    outdir: Path,
) -> None:
    y_true = y_test.to_numpy(dtype=int)
    fig, ax = plt.subplots(
        figsize=(6.6, 5.2)
    )

    for name, result in results.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_true,
            result["probabilities"],
        )
        auc_value = result["metrics"]["AUC"]
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=f"{name} (AUC={auc_value:.2f})",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        linewidth=1.2,
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        outdir / "roc_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# 16. XGBoost Feature Importance Analysis

def xgboost_feature_importance(
    model: Any,
    feature_names: List[str],
    outdir: Path,
) -> pd.DataFrame:
    importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }
    ).sort_values(
        "Importance",
        ascending=False,
    )
    importance["Rank"] = np.arange(
        1,
        len(importance) + 1,
    )
    importance.to_csv(
        outdir / "xgboost_feature_importance.csv",
        index=False,
    )

    top = importance.head(6).sort_values(
        "Importance",
        ascending=True,
    )
    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )
    ax.barh(
        top["Feature"],
        top["Importance"],
    )
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Predictor")
    ax.set_title("XGBoost Feature Importance")
    fig.tight_layout()
    fig.savefig(
        outdir / "xgboost_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    return importance


# 17. SHAP Analysis

def shap_values_for_model(
    model: Any,
    x_explain: np.ndarray,
) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x_explain)
    if isinstance(values, list):
        values = values[-1]
    return np.asarray(values)


# 18. SHAP Summary and Class-Specific Analysis

def shap_summary_analysis(
    model: Any,
    x_explain: np.ndarray,
    y_explain: np.ndarray,
    feature_names: List[str],
    cfg: Config,
    outdir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(x_explain)

    if cfg.shap_sample_size and n > cfg.shap_sample_size:
        rng = np.random.default_rng(cfg.random_state)
        indices = rng.choice(
            n,
            size=cfg.shap_sample_size,
            replace=False,
        )
        x_used = x_explain[indices]
        y_used = np.asarray(y_explain)[indices]
    else:
        x_used = x_explain
        y_used = np.asarray(y_explain)

    shap_values = shap_values_for_model(
        model,
        x_used,
    )

    global_importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "MeanAbsSHAP": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values(
        "MeanAbsSHAP",
        ascending=False,
    )
    global_importance["Rank"] = np.arange(
        1,
        len(global_importance) + 1,
    )
    global_importance.to_csv(
        outdir / "shap_global_importance.csv",
        index=False,
    )

    class_rows = []
    for class_value, class_name in [
        (1, "Fatal Injury"),
        (0, "Non-Fatal Injury"),
    ]:
        mask = y_used == class_value
        values = (
            np.abs(shap_values[mask]).mean(axis=0)
            if mask.any()
            else np.full(shap_values.shape[1], np.nan)
        )
        for i, feature in enumerate(feature_names):
            class_rows.append(
                {
                    "Feature": feature,
                    "Class": class_name,
                    "MeanAbsSHAP": values[i],
                }
            )

    class_importance = pd.DataFrame(class_rows)
    class_importance.to_csv(
        outdir / "shap_class_conditional_importance.csv",
        index=False,
    )

    plt.figure(figsize=(8.0, 6.0))
    shap.summary_plot(
        shap_values,
        x_used,
        feature_names=feature_names,
        show=False,
        max_display=cfg.shap_max_display,
    )
    plt.tight_layout()
    plt.savefig(
        outdir / "shap_summary.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    for class_value, filename, title in [
        (1, "shap_fatal.png", "SHAP Summary - Fatal Injury"),
        (0, "shap_nonfatal.png", "SHAP Summary - Non-Fatal Injury"),
    ]:
        mask = y_used == class_value
        if not mask.any():
            continue
        plt.figure(figsize=(8.0, 6.0))
        shap.summary_plot(
            shap_values[mask],
            x_used[mask],
            feature_names=feature_names,
            show=False,
            max_display=cfg.shap_max_display,
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(
            outdir / filename,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    stacked = class_importance.pivot(
        index="Feature",
        columns="Class",
        values="MeanAbsSHAP",
    ).fillna(0.0)

    for column in ["Fatal Injury", "Non-Fatal Injury"]:
        if column not in stacked.columns:
            stacked[column] = 0.0

    stacked["Overall"] = stacked[
        ["Fatal Injury", "Non-Fatal Injury"]
    ].mean(axis=1)

    stacked = stacked.sort_values(
        "Overall",
        ascending=False,
    ).head(cfg.shap_max_display)

    fig, ax = plt.subplots(
        figsize=(8.2, 6.0)
    )
    yloc = np.arange(len(stacked))
    left = np.zeros(len(stacked))

    for column in ["Fatal Injury", "Non-Fatal Injury"]:
        values = stacked[column].to_numpy()
        ax.barh(
            yloc,
            values,
            left=left,
            label=column,
        )
        left += values

    ax.set_yticks(
        yloc,
        stacked.index,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Mean(|SHAP value|)")
    ax.set_ylabel("Predictor")
    ax.set_title("SHAP Feature Importance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        outdir / "shap_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return shap_values, x_used, y_used


# 19. SHAP Main-Effect Analysis

def shap_main_effects(
    shap_values: np.ndarray,
    x_original: pd.DataFrame,
    feature_names: List[str],
    y_explain: np.ndarray,
    outdir: Path,
) -> None:
    target_features = [
        "Accident Type",
        "Weather Status",
    ]

    for feature_name in target_features:
        if feature_name not in feature_names:
            continue

        feature_index = feature_names.index(feature_name)
        x_values = x_original[feature_name].astype(str)
        categories = list(pd.unique(x_values))
        category_to_number = {
            value: index
            for index, value in enumerate(categories)
        }
        x_numeric = x_values.map(category_to_number).to_numpy(dtype=float)

        for class_value, class_label in [
            (1, "fatal"),
            (0, "nonfatal"),
        ]:
            mask = y_explain == class_value
            if not mask.any():
                continue

            fig, ax = plt.subplots(
                figsize=(7.0, 4.8)
            )
            ax.scatter(
                x_numeric[mask],
                shap_values[mask, feature_index],
                s=18,
                alpha=0.7,
            )
            ax.axhline(0.0, linewidth=1.0)
            ax.set_xticks(
                range(len(categories)),
                categories,
                rotation=25,
                ha="right",
            )
            ax.set_xlabel(feature_name)
            ax.set_ylabel(f"SHAP value for {feature_name}")
            ax.set_title(
                f"{feature_name} SHAP Main Effect - {class_label}"
            )
            fig.tight_layout()
            filename = (
                feature_name.lower().replace(" ", "_")
                + f"_shap_{class_label}.png"
            )
            fig.savefig(
                outdir / filename,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)


# 20. Saving Results and Main Analysis Workflow

def save_metadata(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train_balanced: np.ndarray,
    y_test: pd.Series,
    feature_names: List[str],
    optimized_parameters: Dict[str, Dict[str, Any]],
    cfg: Config,
    outdir: Path,
) -> None:
    metadata = {
        "configuration": asdict(cfg),
        "raw_rows": int(len(raw)),
        "cleaned_rows": int(len(cleaned)),
        "training_rows_before_smote": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "training_rows_after_smote": int(len(y_train_balanced)),
        "training_class_distribution_after_smote": (
            pd.Series(y_train_balanced)
            .value_counts()
            .to_dict()
        ),
        "test_class_distribution": y_test.value_counts().to_dict(),
        "encoded_feature_names": feature_names,
        "optimized_parameters": optimized_parameters,
    }

    (outdir / "analysis_metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def save_cleaning_summary(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    outdir: Path,
) -> None:
    summary = pd.DataFrame(
        [
            {
                "Measure": "Rows before cleaning",
                "Value": len(raw),
            },
            {
                "Measure": "Duplicate rows",
                "Value": int(raw.duplicated().sum()),
            },
            {
                "Measure": "Rows with missing values",
                "Value": int(raw.isna().any(axis=1).sum()),
            },
            {
                "Measure": "Rows after cleaning",
                "Value": len(cleaned),
            },
        ]
    )
    summary.to_csv(
        outdir / "data_cleaning_summary.csv",
        index=False,
    )


def run() -> None:
    random.seed(CFG.random_state)
    np.random.seed(CFG.random_state)

    outdir = Path(CFG.output_dir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = data.copy()
    cleaned = clean_data(raw)
    save_cleaning_summary(raw, cleaned, outdir)

    x, y = define_xy(cleaned)

    x_train, x_test, y_train, y_test = split_data(
        x,
        y,
        CFG,
    )

    _, x_train_encoded, x_test_encoded, feature_names = encode_data(
        x_train,
        x_test,
    )

    x_train_balanced, y_train_balanced = apply_smote(
        x_train_encoded,
        y_train,
        CFG,
    )

    optimized_parameters = optimize_models(
        x_train_balanced,
        y_train_balanced,
        CFG,
        outdir,
    )

    models = construct_final_models(
        optimized_parameters,
        CFG,
    )

    models = fit_models(
        models,
        x_train_balanced,
        y_train_balanced,
    )

    results = evaluate_models(
        models,
        x_test_encoded,
        y_test,
    )

    save_performance_table(
        results,
        outdir,
    )

    save_confusion_matrices(
        results,
        y_test,
        outdir,
    )

    plot_roc(
        results,
        y_test,
        outdir,
    )

    xgb_model = models["XGBoost"]
    xgboost_feature_importance(
        xgb_model,
        feature_names,
        outdir,
    )

    n_explain = min(
        len(x_test_encoded),
        CFG.shap_sample_size,
    )

    rng = np.random.default_rng(CFG.random_state)
    if len(x_test_encoded) > n_explain:
        explain_indices = rng.choice(
            len(x_test_encoded),
            size=n_explain,
            replace=False,
        )
    else:
        explain_indices = np.arange(len(x_test_encoded))

    x_explain = x_test_encoded[explain_indices]
    y_explain = y_test.to_numpy(dtype=int)[explain_indices]
    x_original_explain = x_test.iloc[explain_indices].reset_index(drop=True)

    shap_values, _, y_used = shap_summary_analysis(
        xgb_model,
        x_explain,
        y_explain,
        feature_names,
        CFG,
        outdir,
    )

    shap_main_effects(
        shap_values,
        x_original_explain,
        feature_names,
        y_used,
        outdir,
    )

    save_metadata(
        raw,
        cleaned,
        x_train,
        x_test,
        y_train_balanced,
        y_test,
        feature_names,
        optimized_parameters,
        CFG,
        outdir,
    )

    metrics = pd.read_csv(
        outdir / "model_metrics.csv"
    )

    print("\nModel performance")
    print(metrics.round(4).to_string(index=False))

    print("\nOptimized parameters")
    for model_name, params in optimized_parameters.items():
        print(f"\n{model_name}")
        for parameter, value in params.items():
            print(f"  {parameter}: {value}")

    print(
        f"\nResults saved to: {outdir.resolve()}"
    )


if __name__ == "__main__":
    run()
