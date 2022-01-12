from abc import abstractmethod
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any, ClassVar, Dict, List, NamedTuple, Optional, Type, Union
from typing_extensions import Final

sys.path.insert(0, "../EthicML")

import numpy as np
import pandas as pd

from ethicml import (
    DataTuple,
    InAlgorithm,
    InstalledModel,
    LabelBinarizer,
    Prediction,
    TestTuple,
)
import ethicml as em
from ethicml.evaluators.cross_validator import CVResults, CrossValidator
from ethicml.metrics.accuracy import Accuracy
from ethicml.metrics.cv import AbsCV

SUB_DIR_IMPACT = Path(".") / "disparate_impact" / "run-classifier"
SUB_DIR_MISTREAT = Path(".") / "disparate_mistreatment" / "run_classifier"

MAIN: Final = "main.py"

class FitParams(NamedTuple):
    model_path: Path
    label_converter: LabelBinarizer


class _ZafarAlgorithmBase(InstalledModel):
    def __init__(self, name: str, sub_dir: Path):
        super().__init__(
            name=name,
            dir_name=".",
            # url="https://github.com/predictive-analytics-lab/fair-classification.git",
            top_dir=".",
            # executable="/home/tmk/.conda/envs/zafar/bin/python",  # has to be python 2.7
        )
        self._sub_dir = sub_dir
        self._fit_params: Optional[FitParams] = None

    @staticmethod
    def _create_file_in_zafar_format(
        data: Union[DataTuple, TestTuple], file_path: Path, label_converter: LabelBinarizer
    ) -> None:
        """Save a DataTuple as a JSON file, which is extremely inefficient but what Zafar wants."""
        out: Dict[str, Any] = {}
        out["x"] = data.x.to_numpy().tolist()
        sens_attr = data.s.columns[0]
        out["sensitive"] = {}
        out["sensitive"][sens_attr] = data.s[sens_attr].to_numpy().tolist()
        if isinstance(data, DataTuple):
            data_converted = label_converter.adjust(data)
            class_attr = data.y.columns[0]
            out["class"] = (2 * data_converted.y[class_attr].to_numpy() - 1).tolist()
        else:
            out["class"] = [-1 for _ in range(data.x.shape[0])]
        with file_path.open("w") as out_file:
            json.dump(out, out_file)

    # async def run_async(self, train: DataTuple, test: TestTuple) -> Prediction:
    #     label_converter = LabelBinarizer()
    #     with TemporaryDirectory() as tmpdir:
    #         tmp_path = Path(tmpdir)
    #         train_path = tmp_path / "train.json"
    #         test_path = tmp_path / "test.json"
    #         self._create_file_in_zafar_format(train, train_path, label_converter)
    #         self._create_file_in_zafar_format(test, test_path, label_converter)
    #         predictions_path = tmp_path / "predictions.json"

    #         cmd = self._create_command_line(str(train_path), str(test_path), str(predictions_path))
    #         working_dir = self._code_path.resolve() / self._sub_dir
    #         await self._call_script(cmd, cwd=working_dir)
    #         predictions = predictions_path.open().read()
    #         predictions = json.loads(predictions)

    #     predictions_correct = pd.Series([0 if x == -1 else 1 for x in predictions])
    #     return Prediction(hard=label_converter.post_only_labels(predictions_correct))

    def run(self, train: DataTuple, test: TestTuple) -> Prediction:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            fit_params = self._fit(train, tmp_path)
            return self._predict(test, tmp_path, fit_params)

    def fit(self, train: DataTuple) -> "_ZafarAlgorithmBase":
        with TemporaryDirectory() as tmpdir:
            self._fit_params = self._fit(train, tmp_path=Path(tmpdir), model_dir=self._code_path)
        return self

    def predict(self, test: TestTuple) -> Prediction:
        assert self._fit_params is not None, "call fit() first"
        with TemporaryDirectory() as tmpdir:
            return self._predict(test, tmp_path=Path(tmpdir), fit_params=self._fit_params)

    def _fit(
        self, train: DataTuple, tmp_path: Path, model_dir: Optional[Path] = None
    ) -> FitParams:
        model_path = (model_dir if model_dir is not None else tmp_path) / "model.npy"
        label_converter = LabelBinarizer()
        train_path = tmp_path / "train.json"
        self._create_file_in_zafar_format(train, train_path, label_converter)

        cmd = self._get_fit_cmd(str(train_path), str(model_path))
        working_dir = self._code_path.resolve() / self._sub_dir
        self._call_script(cmd, cwd=working_dir)

        return FitParams(model_path, label_converter)

    def _predict(
        self, test: TestTuple, tmp_path: Path, fit_params: FitParams
    ) -> Prediction:
        test_path = tmp_path / "test.json"
        self._create_file_in_zafar_format(test, test_path, fit_params.label_converter)
        predictions_path = tmp_path / "predictions.json"
        cmd = self._get_predict_cmd(
            str(test_path), str(fit_params.model_path), str(predictions_path)
        )
        working_dir = self._code_path.resolve() / self._sub_dir
        self._call_script(cmd, cwd=working_dir)
        predictions = predictions_path.open().read()
        predictions = json.loads(predictions)

        predictions_correct = pd.Series([0 if x == -1 else 1 for x in predictions])
        return Prediction(hard=fit_params.label_converter.post_only_labels(predictions_correct))

    @abstractmethod
    def _create_command_line(
        self, train_name: str, test_name: str, predictions_name: str
    ) -> List[str]:
        pass

    @abstractmethod
    def _get_fit_cmd(self, train_name: str, model_path: str) -> List[str]:
        pass

    def _get_predict_cmd(self, test_name: str, model_path: str, output_file: str) -> List[str]:
        return ["predict.py", test_name, model_path, output_file]


class ZafarBaseline(_ZafarAlgorithmBase):
    """Zafar without fairness."""

    def __init__(self) -> None:
        super().__init__(name="ZafarBaseline", sub_dir=SUB_DIR_IMPACT)

    def _create_command_line(
        self, train_name: str, test_name: str, predictions_name: str
    ) -> List[str]:
        return [MAIN, train_name, test_name, predictions_name, "baseline", "0"]

    def _get_fit_cmd(self, train_name: str, model_path: str) -> List[str]:
        return ["fit.py", train_name, model_path, "baseline", "0"]


class ZafarAccuracy(_ZafarAlgorithmBase):
    """Zafar with fairness."""

    def __init__(self, gamma: float = 0.5):
        super().__init__(name=f"ZafarAccuracy, γ={gamma}", sub_dir=SUB_DIR_IMPACT)
        self.gamma = gamma

    def _create_command_line(
        self, train_name: str, test_name: str, predictions_name: str
    ) -> List[str]:
        return [MAIN, train_name, test_name, predictions_name, "gamma", str(self.gamma)]

    def _get_fit_cmd(self, train_name: str, model_path: str) -> List[str]:
        return ["fit.py", train_name, model_path, "gamma", str(self.gamma)]


class ZafarFairness(_ZafarAlgorithmBase):
    """Zafar with fairness."""

    def __init__(self, c: float = 0.001):
        super().__init__(name=f"ZafarFairness, c={c}", sub_dir=SUB_DIR_IMPACT)
        self._c = c

    def _create_command_line(
        self, train_name: str, test_name: str, predictions_name: str
    ) -> List[str]:
        return [MAIN, train_name, test_name, predictions_name, "c", str(self._c)]

    def _get_fit_cmd(self, train_name: str, model_path: str) -> List[str]:
        return ["fit.py", train_name, model_path, "c", str(self._c)]


class ZafarEqOpp(_ZafarAlgorithmBase):
    """Zafar for Equality of Opportunity."""

    _mode: ClassVar[str] = "fnr"  # class level constant
    _base_name: ClassVar[str] = "ZafarEqOpp"

    def __init__(self, tau: float = 5.0, mu: float = 1.2, eps: float = 0.0001):
        super().__init__(name=f"{self._base_name}, τ={tau}, μ={mu}", sub_dir=SUB_DIR_MISTREAT)
        self._tau = tau
        self._mu = mu
        self._eps = eps

    def _create_command_line(
        self, train_name: str, test_name: str, predictions_name: str
    ) -> List[str]:
        return [
            MAIN,
            train_name,
            test_name,
            predictions_name,
            self._mode,
            str(self._tau),
            str(self._mu),
            str(self._eps),
        ]

    def _get_fit_cmd(self, train_name: str, model_path: str) -> List[str]:
        return [
            "fit.py",
            train_name,
            model_path,
            self._mode,
            str(self._tau),
            str(self._mu),
            str(self._eps),
        ]


class ZafarEqOdds(ZafarEqOpp):
    """Zafar for Equalised Odds."""

    _mode: ClassVar[str] = "fprfnr"
    _base_name: ClassVar[str] = "ZafarEqOdds"


def count_true(mask: np.ndarray) -> int:
    """Count the number of elements that are True."""
    return mask.nonzero()[0].shape[0]


def main():
    train, test = em.train_test_split(em.toy().load())
    model: InAlgorithm = ZafarAccuracy()
    assert model.name == "ZafarAccuracy, γ=0.5"

    assert model is not None

    predictions: Prediction = model.run(train, test)
    expected_num_pos = 41
    assert count_true(predictions.hard.values == 1) == expected_num_pos
    assert count_true(predictions.hard.values == 0) == len(predictions) - expected_num_pos

    predictions = model.run(train, test)
    expected_num_pos = 41
    assert count_true(predictions.hard.values == 1) == expected_num_pos
    assert count_true(predictions.hard.values == 0) == len(predictions) - expected_num_pos

    hyperparams: Dict[str, List[float]] = {"gamma": [1, 1e-1, 1e-2]}

    model_class: Type[InAlgorithm] = ZafarAccuracy
    zafar_cv = CrossValidator(model_class, hyperparams, folds=3)

    assert zafar_cv is not None

    primary = Accuracy()
    fair_measure = AbsCV()
    cv_results: CVResults = zafar_cv.run(train, measures=[primary, fair_measure])

    best_result = cv_results.get_best_in_top_k(primary, fair_measure, top_k=3)

    assert best_result.params["gamma"] == 1
    # assert best_result.scores["Accuracy"] == approx(0.956, abs=1e-3)
    # assert best_result.scores["CV absolute"] == approx(0.834, abs=1e-3)

    model = ZafarBaseline()
    assert model.name == "ZafarBaseline"

    assert model is not None

    predictions = model.run(train, test)
    expected_num_pos = 40
    assert count_true(predictions.hard.values == 1) == expected_num_pos
    assert count_true(predictions.hard.values == 0) == len(predictions) - expected_num_pos

    predictions = model.run(train, test)
    expected_num_pos = 40
    assert count_true(predictions.hard.values == 1) == expected_num_pos
    assert count_true(predictions.hard.values == 0) == len(predictions) - expected_num_pos

    model = ZafarFairness()
    assert model.name == "ZafarFairness, c=0.001"

    assert model is not None

    predictions = model.run(train, test)
    expected_num_pos = 51
    assert count_true(predictions.hard.values == 1) == expected_num_pos
    assert count_true(predictions.hard.values == 0) == len(predictions) - expected_num_pos

    predictions = model.run(train, test)
    expected_num_pos = 51
    assert count_true(predictions.hard.values == 1) == expected_num_pos
    assert count_true(predictions.hard.values == 0) == len(predictions) - expected_num_pos

    hyperparams = {"c": [1, 1e-1, 1e-2]}

    model_class = ZafarFairness
    zafar_cv = CrossValidator(model_class, hyperparams, folds=3)

    assert zafar_cv is not None

    primary = Accuracy()
    fair_measure = AbsCV()
    cv_results: CVResults = zafar_cv.run(train, measures=[primary, fair_measure])

    best_result = cv_results.get_best_in_top_k(primary, fair_measure, top_k=3)

    assert best_result.params["c"] == 0.01
    # assert best_result.scores["Accuracy"] == approx(0.703, abs=1e-3)
    # assert best_result.scores["CV absolute"] == approx(0.855, rel=1e-3)

    # ==================== Zafar Equality of Opportunity ========================
    zafar_eq_opp: InAlgorithm = ZafarEqOpp()
    assert zafar_eq_opp.name == "ZafarEqOpp, τ=5.0, μ=1.2"

    predictions = zafar_eq_opp.run(train, test)
    assert count_true(predictions.hard.values == 1) == 40

    # ==================== Zafar Equalised Odds ========================
    zafar_eq_odds: InAlgorithm = ZafarEqOdds()
    assert zafar_eq_odds.name == "ZafarEqOdds, τ=5.0, μ=1.2"

    predictions = zafar_eq_odds.run(train, test)
    assert count_true(predictions.hard.values == 1) == 40


if __name__ == '__main__':
    main()
