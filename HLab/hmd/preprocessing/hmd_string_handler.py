from abc import ABC, abstractclassmethod
from typing import Any, Optional
import re
from string import punctuation


class StringHandler(ABC):
    def __init__(self, successor: Optional['StringHandler'] = None):
        self.successor = successor

    @abstractclassmethod
    def handle(self, request: str) -> str:
        """String handle for NLP preprocessing"""


class ToLowerCase(StringHandler):
    def handle(self, request: str) -> str:
        request = request.lower()
        
        if self.successor is not None:
            return self.successor.handle(request)

        return request


class RemoveWhiteSpace(StringHandler):
    def handle(self, request: str) -> str:
        request = re.sub('\n', ' ', request)
        request = re.sub('\r', ' ', request)
        request = re.sub('\t', ' ', request)
        request = re.sub(' +', ' ', request)

        if self.successor is not None:
            return self.successor.handle(request)

        return request


class RemovePunctuation(StringHandler):
    def handle(self, request: str) -> str:
        request = request.translate(str.maketrans('', '', punctuation))

        if self.successor is not None:
            return self.successor.handle(request)

        return request
