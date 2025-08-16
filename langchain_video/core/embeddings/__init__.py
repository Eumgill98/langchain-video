from typing import Union
from .base import TextEmbeddings, ImageEmbeddings, AudioEmbeddings

MultiModalEmbeddings = Union[TextEmbeddings, ImageEmbeddings, AudioEmbeddings]