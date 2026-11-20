import tempfile
import unittest
from pathlib import Path

from validate_data import (
    SPECIES, DataValidationError, load_contract, read_json, validate_payload, validate_records,
)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_contract()
        self.valid = self.contract["examples"][0]

    def test_valid_example(self):
        self.assertEqual(validate_payload(self.valid, self.contract), [5.1, 3.5, 1.4, 0.2])

    def test_invalid_example_reports_all_three_rules(self):
        with self.assertRaises(DataValidationError) as raised:
            validate_payload(self.contract["x-invalid-example"], self.contract)
        for message in ("sepal_length_cm: se requiere un número", "sepal_width_cm: valor nulo",
                        "petal_length_cm: -1.4 fuera del rango"):
            self.assertIn(message, str(raised.exception))

    def test_json_rejects_boolean_non_finite_and_wrong_shape(self):
        for value in (True, float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(DataValidationError):
                validate_payload({**self.valid, "petal_width_cm": value}, self.contract)
        with self.assertRaisesRegex(DataValidationError, "objeto JSON"):
            validate_payload([self.valid], self.contract)

    def test_json_rejects_missing_and_extra_fields(self):
        payload = {**self.valid, "species": "setosa"}
        del payload["sepal_width_cm"]
        with self.assertRaises(DataValidationError) as raised:
            validate_payload(payload, self.contract)
        self.assertIn("species: campo no permitido", str(raised.exception))
        self.assertIn("sepal_width_cm: campo requerido ausente", str(raised.exception))

    def test_inclusive_bounds_for_every_measurement(self):
        for field, rule in self.contract["properties"].items():
            for edge, offset in (("minimum", -0.01), ("maximum", 0.01)):
                with self.subTest(field=field, edge=edge):
                    validate_payload({**self.valid, field: rule[edge]}, self.contract)
                    with self.assertRaisesRegex(DataValidationError, "fuera del rango"):
                        validate_payload({**self.valid, field: rule[edge] + offset}, self.contract)

    def test_strict_json_syntax(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.json"
            for example in ('{"x":NaN}', '{"x":1,"x":2}', '{"x":}'):
                with self.subTest(example=example):
                    path.write_text(example, encoding="utf-8")
                    with self.assertRaises(DataValidationError):
                        read_json(path)

    def training_sample(self):
        return [{**self.valid, "species": name} for name in SPECIES for _ in range(5)]

    def test_valid_training_records(self):
        X, y = validate_records(self.training_sample(), self.contract)
        self.assertEqual(len(X), 15)
        self.assertEqual(X[0], [5.1, 3.5, 1.4, 0.2])
        self.assertEqual(y.count("setosa"), 5)

    def test_invalid_records_report_row_and_field(self):
        records = self.training_sample()
        records[0].update(sepal_length_cm="5.1", sepal_width_cm=None)
        records[1]["petal_length_cm"] = -1.4
        records[2]["species"] = "otra"
        with self.assertRaises(DataValidationError) as raised:
            validate_records(records, self.contract)
        for error in ("Registro 1, sepal_length_cm", "Registro 1, sepal_width_cm",
                      "Registro 2, petal_length_cm", "Registro 3, species"):
            self.assertIn(error, str(raised.exception))

    def test_record_structure_and_class_count(self):
        records = self.training_sample()
        cases = [
            ([], "no contiene"),
            ([records[0]], "al menos 5 registros"),
            ([[]], "Registro 1: se requiere un objeto"),
            ([self.valid], "species: campo requerido ausente"),
            ([{**self.valid, "species": None}], "species: valor nulo"),
        ]
        for sample, error in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(DataValidationError, error):
                    validate_records(sample, self.contract)

    def test_record_key_order_does_not_change_features(self):
        records = self.training_sample()
        reordered = [dict(reversed(list(record.items()))) for record in records]
        self.assertEqual(validate_records(reordered, self.contract), validate_records(records, self.contract))


if __name__ == "__main__":
    unittest.main()
