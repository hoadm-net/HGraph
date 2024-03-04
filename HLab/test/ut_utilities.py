import unittest
from pathlib import Path
from HLab.hmd import Utilities as Util


class UT_Utilities(unittest.TestCase):
    def test_get_basedir(self):
        self.assertEqual(Path(__file__).parent.parent, Util.get_basedir())


if __name__ == '__main__':
    unittest.main()
