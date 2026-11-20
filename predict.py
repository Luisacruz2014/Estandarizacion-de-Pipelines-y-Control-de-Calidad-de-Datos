"""Valida un objeto JSON y predice la especie con un modelo propio del proyecto."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from run_pipeline import DEFAULT_OUTPUT
from validate_data import CONTRACT_PATH, DataValidationError, load_contract, read_json, validate_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Ruta de un archivo con un objeto JSON.")
    source.add_argument("--example", choices=("valid", "invalid"), help="Ejemplo incluido en el contrato.")
    parser.add_argument("--model", type=Path, default=DEFAULT_OUTPUT / "model.joblib")
    args = parser.parse_args()

    try:
        contract = load_contract()
        if args.input is not None:
            payload = read_json(args.input)
        elif args.example == "valid":
            payload = contract["examples"][0]
        else:
            payload = contract["x-invalid-example"]
        values = validate_payload(payload, contract)
        if not args.model.is_file():
            raise DataValidationError(f"No se encontró el modelo en {args.model}. Ejecuta el pipeline primero.")

        import joblib

        saved = joblib.load(args.model)
        if saved["contract_sha256"] != hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest():
            raise DataValidationError("El contrato cambió desde el entrenamiento. Vuelve a ejecutar el pipeline.")
        species = saved["model"].predict([values])[0]
        print(json.dumps({"species": str(species)}, ensure_ascii=False))
    except (DataValidationError, OSError, ImportError) as exc:
        print(f"Inferencia interrumpida.\n{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
