"""Ejecuta el control de calidad, el entrenamiento y la evaluación de Iris."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from validate_data import (
    CONTRACT_PATH,
    ROOT,
    SPECIES,
    TARGET,
    DataValidationError,
    load_contract,
    validate_records,
)

SEED = 42
TEST_SIZE = 0.20
DEFAULT_OUTPUT = ROOT / "artifacts"


def load_iris_records():
    """Carga Iris desde scikit-learn y lo convierte en registros de Python."""
    from sklearn.datasets import load_iris

    iris = load_iris()
    fields = [f"{name.removesuffix(' (cm)').replace(' ', '_')}_cm" for name in iris.feature_names]
    labels = iris.target_names.tolist()
    return [
        {**dict(zip(fields, row, strict=True)), TARGET: labels[label]}
        for row, label in zip(iris.data.tolist(), iris.target.tolist(), strict=True)
    ]


def run_pipeline(output_dir, *, demo_invalid=False, validate_only=False):
    contract = load_contract()
    records = load_iris_records()
    if demo_invalid:
        # Los ejemplos defectuosos viven únicamente en memoria.
        records[0]["sepal_length_cm"] = "5.1"
        records[0]["sepal_width_cm"] = None
        records[1]["petal_length_cm"] = -1.4
        records[2][TARGET] = "otra"

    X, y = validate_records(records, contract)
    print(f"[1/3] Control de calidad aprobado: {len(X)} registros cargados con load_iris().")
    if validate_only:
        return None

    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()

    import joblib
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    expected = set(SPECIES)
    if set(y_train) != expected or set(y_test) != expected:
        raise DataValidationError("Las tres clases deben aparecer en ambas particiones.")
    model = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=1000))
    model.fit(X_train, y_train)
    print(f"[2/3] Regresión logística entrenada con {len(y_train)} filas.")

    predictions = model.predict(X_test)
    schema_hash = hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
    metrics = {
        "dataset": "sklearn.datasets.load_iris",
        "dataset_sha256": digest,
        "contract_version": contract["x-contract-version"],
        "contract_sha256": schema_hash,
        "sklearn_version": sklearn.__version__,
        "feature_order": contract["required"],
        "seed": SEED,
        "test_fraction": TEST_SIZE,
        "train_size": len(y_train),
        "test_size": len(y_test),
        "train_support": dict(sorted(Counter(y_train).items())),
        "test_support": dict(sorted(Counter(y_test).items())),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, labels=SPECIES, average="macro")),
        "class_order": list(SPECIES),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=SPECIES).tolist(),
    }
    artifact = {"model": model, "contract_sha256": schema_hash}

    # Genera ambos archivos antes de sustituir salidas previas.
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as staging:
        temp = Path(staging)
        joblib.dump(artifact, temp / "model.joblib")
        (temp / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        for filename in ("model.joblib", "metrics.json"):
            (temp / filename).replace(output_dir / filename)
    print(f"[3/3] Evaluación con {len(y_test)} filas no usadas para ajustar el modelo.")
    print(f"Exactitud: {metrics['accuracy']:.4f} | F1 macro: {metrics['f1_macro']:.4f}")
    print(f"Salidas: {output_dir.resolve()}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-invalid", action="store_true", help="Inyecta errores de prueba en memoria.")
    parser.add_argument("--validate-only", action="store_true", help="Valida los registros sin entrenar.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        run_pipeline(
            args.output_dir, demo_invalid=args.demo_invalid, validate_only=args.validate_only
        )
    except (DataValidationError, OSError, ImportError) as exc:
        print(f"Pipeline interrumpido.\n{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
