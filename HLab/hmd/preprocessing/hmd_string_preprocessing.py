from .hmd_string_handler import StringHandler


class StringPreprocessing(object):
    def __init__(self, request: str) -> None:
        self._request = request
        self._handlers = []
    
    def add_handler(self, handler: StringHandler) -> None:
        if len(self._handlers) != 0:
            self._handlers[-1].successor = handler
            
        self._handlers.append(handler)

    def execute(self) -> str:
        return self._handlers[0].handle(self._request)
