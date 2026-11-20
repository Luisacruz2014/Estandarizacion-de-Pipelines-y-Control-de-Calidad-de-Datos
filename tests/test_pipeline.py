import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from validate_data import ROOT


class PipelineTests(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run(
            [sys.executable, "-X", "utf8", *args],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60,
        )

    def test_training_evaluation_and_prediction(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            pipeline = self.invoke("run_pipeline.py", "--output-dir", str(output))
            self.assertEqual(pipeline.returncode, 0, pipeline.stderr)
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["dataset"], "sklearn.datasets.load_iris")
            self.assertEqual((metrics["train_size"], metrics["test_size"]), (120, 30))
            self.assertEqual(metrics["test_support"], {"setosa": 10, "versicolor": 10, "virginica": 10})
            self.assertGreater(metrics["accuracy"], 0.8)
            self.assertAlmostEqual(metrics["accuracy"], 28 / 30)
            self.assertAlmostEqual(metrics["f1_macro"], 28 / 30)
            self.assertEqual(metrics["confusion_matrix"], [[10, 0, 0], [0, 9, 1], [0, 1, 9]])
            result = self.invoke("predict.py", "--example", "valid", "--model", str(output / "model.joblib"))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {"species": "setosa"})

    def test_bad_dataset_stops_before_training_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "artifacts"
            pipeline = self.invoke("run_pipeline.py", "--demo-invalid",
                                   "--output-dir", str(output))
            self.assertEqual(pipeline.returncode, 2)
            self.assertIn("Registro 1, sepal_length_cm", pipeline.stderr)
            self.assertNotIn("[2/3]", pipeline.stdout)
            self.assertFalse(output.exists())

    def test_validate_only_loads_iris_without_training(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "artifacts"
            result = self.invoke("run_pipeline.py", "--validate-only", "--output-dir", str(output))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("150 registros cargados con load_iris()", result.stdout)
            self.assertNotIn("[2/3]", result.stdout)
            self.assertFalse(output.exists())

    def test_invalid_example_stops_before_loading_model(self):
        result = self.invoke("predict.py", "--example", "invalid", "--model", "no_existe.joblib")
        self.assertEqual(result.returncode, 2)
        self.assertIn("valor nulo", result.stderr)
        self.assertNotIn("No se encontró el modelo", result.stderr)


if __name__ == "__main__":
    unittest.main()
