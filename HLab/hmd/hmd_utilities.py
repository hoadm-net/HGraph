from pathlib import Path


class Utilities(object):
    @staticmethod
    def get_basedir():
        return Path(__file__).parent.parent
    