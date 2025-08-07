from pathlib import PurePath
from typing import Union, Literal

PathLike = Union[str, PurePath]
SamplingStrategy = Literal["uniform", "random", "first", "last", "all"]
