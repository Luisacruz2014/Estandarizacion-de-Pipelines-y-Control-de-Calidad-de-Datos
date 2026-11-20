"""Valida los datos de entrenamiento y de inferencia sin dependencias externas."""

from collections import Counter
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / "data_contract.json"
TARGET = "species"
SPECIES = ("setosa", "versicolor", "virginica")
MIN_PER_CLASS = 5


class DataValidationError(ValueError):
    """Una entrada no puede continuar a entrenamiento o inferencia."""


def _reject_constant(text):
    raise DataValidationError(f"{text!r} no es un valor JSON permitido.")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DataValidationError(f"Campo JSON duplicado: {key!r}.")
        result[key] = value
    return result


def read_json(path):
    """Lee JSON estricto y rechaza claves duplicadas."""
    try:
        with Path(path).open(encoding="utf-8-sig") as file:
            return json.load(file, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, ValueError) as exc:
        raise DataValidationError(f"{Path(path).name}: {exc}") from exc


def load_contract():
    return read_json(CONTRACT_PATH)


def _measurement_problem(value, rule):
    if value is None:
        return "valor nulo"
    if type(value) not in (int, float):
        return "se requiere un número, sin comillas ni booleanos"
    if isinstance(value, float) and not math.isfinite(value):
        return "se requiere un número finito"
    low, high = rule["minimum"], rule["maximum"]
    if not low <= value <= high:
        return f"{value!r} fuera del rango inclusivo [{low}, {high}] cm"
    return None


def validate_payload(payload, contract):
    """Devuelve las cuatro mediciones ordenadas según el contrato."""
    if not isinstance(payload, dict):
        raise DataValidationError("La entrada debe ser un objeto JSON.")
    fields = contract["required"]
    errors = [f"{key}: campo no permitido" for key in sorted(payload.keys() - set(fields))]
    measurements = []
    for field in fields:
        if field not in payload:
            errors.append(f"{field}: campo requerido ausente")
            continue
        value = payload[field]
        problem = _measurement_problem(value, contract["properties"][field])
        if problem:
            errors.append(f"{field}: {problem}")
        else:
            measurements.append(float(value))
    if errors:
        raise DataValidationError("\n".join(errors))
    return measurements


def validate_records(records, contract):
    """Devuelve X e y solo si todos los registros cargados en memoria son válidos."""
    features, targets, errors = [], [], []
    for number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            errors.append(f"Registro {number}: se requiere un objeto con campos nombrados.")
            continue

        measurements = {key: value for key, value in record.items() if key != TARGET}
        before = len(errors)
        try:
            values = validate_payload(measurements, contract)
        except DataValidationError as exc:
            errors.extend(f"Registro {number}, {message}." for message in str(exc).splitlines())

        if TARGET not in record:
            errors.append(f"Registro {number}, {TARGET}: campo requerido ausente.")
        elif record[TARGET] is None:
            errors.append(f"Registro {number}, {TARGET}: valor nulo.")
        elif type(record[TARGET]) is not str or record[TARGET] not in SPECIES:
            errors.append(f"Registro {number}, {TARGET}: se esperaba una de estas especies: {SPECIES}.")

        if len(errors) == before:
            features.append(values)
            targets.append(record[TARGET])

    if errors:
        shown = errors[:20]
        if len(errors) > 20:
            shown.append(f"... y {len(errors) - 20} errores adicionales.")
        raise DataValidationError("\n".join(shown))
    if not features:
        raise DataValidationError("El conjunto de datos no contiene observaciones.")
    distribution = Counter(targets)
    for species in SPECIES:
        if distribution[species] < MIN_PER_CLASS:
            raise DataValidationError(
                f"Se requieren al menos {MIN_PER_CLASS} registros por especie. "
                f"{species} tiene {distribution[species]}."
            )
    return features, targets
