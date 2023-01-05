from typing import Callable, Literal, List, Dict, Tuple

from typing_extensions import TypeAlias, TypedDict, NotRequired

__all__ = [
    "Setting",
    "DataFile",
    "ConsParams",
    "ConsType",
    "LossFunction",
    "SensitiveAttrsToCovThresh",
    "Mode",
    "Constraint",
]

Setting: TypeAlias = Literal["gamma", "c", "baseline"]
ConsType: TypeAlias = Literal[-1, 0, 1, 2, 3, 4]
MetricType: TypeAlias = Literal["fp", "fn", "fpr", "fnr", "acc"]
LossFunction: TypeAlias = Literal["svm_linear", "logreg"]
Mode: TypeAlias = Literal["baseline", "fpr", "fnr", "fprfnr"]
SensitiveAttrsToCovThresh: TypeAlias = Dict[str, Dict[ConsType, Dict[int, int]]]

# needs functional syntax because "class" is a protected keyword
DataFile = TypedDict(
    "DataFile", {"x": List[float], "class": List[int], "sensitive": Dict[str, int]}
)


class ConsParams(TypedDict):
    EPS: NotRequired[float]
    cons_type: ConsType
    mu: NotRequired[float]
    sensitive_attrs_to_cov_thresh: SensitiveAttrsToCovThresh
    take_initial_sol: NotRequired[bool]
    tau: NotRequired[float]


class Constraint(TypedDict):
    type: str
    fun: Callable
    args: Tuple
