from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an XGBoost classifier on IoMT.csv")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "IoMT.csv",
        help="Path to the merged IoMT dataset CSV file.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="attack_type",
        help="Name of the target column.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models",
        help="Directory to save trained artifacts.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path(__file__).resolve().parent / "reports" / "train_results.txt",
        help="Path to save the training report.",
    )
    return parser.parse_args()


def load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded {data_path.name}: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
    print("-" * 80)
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def save_artifacts(pipeline: Pipeline, label_encoder: LabelEncoder, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = model_dir / "xgboost_pipeline.joblib"
    label_encoder_path = model_dir / "label_encoder.joblib"

    joblib.dump(pipeline, pipeline_path)
    joblib.dump(label_encoder, label_encoder_path)

    print(f"Saved pipeline to {pipeline_path}")
    print(f"Saved label encoder to {label_encoder_path}")


def save_report(report_path: Path, report_text: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"Saved training report to {report_path}")


def train_model(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
    model_dir: Path,
    report_path: Path,
) -> None:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    X = df.drop(columns=[target_column])
    y = df[target_column].astype(str)

    if X.empty:
        raise ValueError("No feature columns found after removing the target column.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    preprocessor = build_preprocessor(X)
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    predicted_labels = label_encoder.inverse_transform(predictions)
    true_labels = label_encoder.inverse_transform(y_test)
    accuracy = accuracy_score(y_test, predictions)
    class_report = classification_report(true_labels, predicted_labels, digits=4)
    matrix = confusion_matrix(true_labels, predicted_labels, labels=label_encoder.classes_)

    print(f"Train size: {len(X_train)} rows")
    print(f"Test size: {len(X_test)} rows")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(class_report)
    print("Confusion matrix:")
    print(matrix)
    print("Labels:")
    print(list(label_encoder.classes_))

    save_artifacts(pipeline, label_encoder, model_dir)
    report_text = dedent(
        f"""\
        XGBoost result:
        Dataset: {df.shape[0]} rows, {df.shape[1]} columns
        Target column: {target_column}
        Train size: {len(X_train)} rows
        Test size: {len(X_test)} rows
        Accuracy: {accuracy:.4f}

        Classification report:
        {class_report}
        Confusion matrix:
        {matrix}
        Labels:
        {list(label_encoder.classes_)}
        """
    )
    save_report(report_path, report_text)


def main() -> None:
    args = parse_args()
    df = load_dataset(args.data_path)
    train_model(
        df,
        args.target_column,
        args.test_size,
        args.random_state,
        args.model_dir,
        args.report_path,
    )


if __name__ == "__main__":
    main()
