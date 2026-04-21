from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.generate_data import POST_TYPES, save_dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
DATA_FILE = ARTIFACTS / "instagram_posts.csv"


def make_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "followers",
        "caption_length",
        "hashtags_count",
        "hour_posted",
        "is_weekend",
        "sponsorship",
    ]
    categorical_features = ["post_type"]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def build_models() -> dict[str, object]:
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, max_depth=10, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }


def evaluate_target(df: pd.DataFrame, target: str) -> tuple[dict[str, dict[str, float]], str]:
    X = df[
        [
            "followers",
            "post_type",
            "caption_length",
            "hashtags_count",
            "hour_posted",
            "is_weekend",
            "sponsorship",
        ]
    ]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    metrics: dict[str, dict[str, float]] = {}
    best_name = ""
    best_mae = float("inf")

    for name, estimator in build_models().items():
        pipeline = Pipeline(
            steps=[("preprocessor", make_preprocessor()), ("model", estimator)]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        metrics[name] = {"mae": round(mae, 3), "r2": round(r2, 3)}

        joblib.dump(pipeline, ARTIFACTS / f"{target}_{slugify(name)}.joblib")

        if mae < best_mae:
            best_mae = mae
            best_name = name
            joblib.dump(pipeline, ARTIFACTS / f"best_{target}_model.joblib")

    return metrics, best_name


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_")


def generate_charts(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="post_type", y="likes", estimator="mean", errorbar=None, order=POST_TYPES)
    plt.title("Average Likes by Post Type")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "avg_likes_by_post_type.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="post_type", y="comments", estimator="mean", errorbar=None, order=POST_TYPES)
    plt.title("Average Comments by Post Type")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "avg_comments_by_post_type.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="followers", y="likes", hue="post_type", alpha=0.7)
    plt.title("Followers vs Likes")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "followers_vs_likes.png")
    plt.close()


def generate_model_comparison_charts(
    likes_metrics: dict[str, dict[str, float]],
    comments_metrics: dict[str, dict[str, float]],
) -> None:
    comparison_rows: list[dict[str, object]] = []
    for target_name, metrics in [("Likes", likes_metrics), ("Comments", comments_metrics)]:
        for model_name, values in metrics.items():
            comparison_rows.append(
                {
                    "target": target_name,
                    "model": model_name,
                    "mae": values["mae"],
                    "r2": values["r2"],
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=comparison_df, x="model", y="mae", hue="target")
    plt.title("Model Comparison by MAE")
    plt.ylabel("Mean Absolute Error")
    plt.xlabel("Algorithm")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "model_comparison_mae.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=comparison_df, x="model", y="r2", hue="target")
    plt.title("Model Comparison by R2 Score")
    plt.ylabel("R2 Score")
    plt.xlabel("Algorithm")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "model_comparison_r2.png")
    plt.close()


def build_summary(df: pd.DataFrame) -> dict[str, object]:
    grouped = (
        df.groupby("post_type")[["likes", "comments"]]
        .mean()
        .round(1)
        .sort_values("likes", ascending=False)
    )

    return {
        "dataset_rows": int(len(df)),
        "best_for_likes": grouped["likes"].idxmax(),
        "best_for_comments": grouped["comments"].idxmax(),
        "post_type_averages": grouped.reset_index().to_dict(orient="records"),
    }


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    save_dataset(DATA_FILE)
    df = pd.read_csv(DATA_FILE)

    likes_metrics, best_likes_model = evaluate_target(df, "likes")
    comments_metrics, best_comments_model = evaluate_target(df, "comments")
    generate_charts(df)
    generate_model_comparison_charts(likes_metrics, comments_metrics)

    summary = build_summary(df)
    summary["model_comparison"] = {
        "likes": likes_metrics,
        "comments": comments_metrics,
        "selected_models": {
            "likes": best_likes_model,
            "comments": best_comments_model,
        },
        "charts": {
            "mae": "/artifacts/model_comparison_mae.png",
            "r2": "/artifacts/model_comparison_r2.png",
        },
    }

    with open(ARTIFACTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training complete.")
    print(json.dumps(summary["model_comparison"], indent=2))


if __name__ == "__main__":
    main()
