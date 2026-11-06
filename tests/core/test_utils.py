import unittest

from asah_segmentation.utils import safe_tuple


class SafeTupleTest(unittest.TestCase):

    def test_true_gives_singleton(self):
        self.assertEqual(safe_tuple(True, 2), (2,))

    def test_false_gives_empty(self):
        self.assertEqual(safe_tuple(False, 2), ())

    def test_star_expansion(self):
        self.assertEqual((1, *safe_tuple(True, 2), 3), (1, 2, 3))
        self.assertEqual((4, *safe_tuple(False, 5), 6), (4, 6))

    def test_element_is_not_copied_or_unpacked(self):
        el = [1, 2]
        result = safe_tuple(True, el)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], el)


if __name__ == "__main__":
    unittest.main()
