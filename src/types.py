from typing import Literal, List, Dict

from typing_extensions import TypeAlias, TypedDict

__all__ = ["Setting", "DataFile"]

Setting: TypeAlias = Literal["gamma", "c", "baseline"]

DataFile = TypedDict(
    "DataFile", {"x": List[float], "class": List[int], "sensitive": Dict[str, int]}
)
