import unittest
from data_loader import MnistLoader

class TestMnistLoader(unittest.TestCase):
    def test_type_of_output(self):
        """
        Test that the output is of type list
        """
        dir = '.'
        result = MnistLoader().load_data(dir, train=True)
        self.assertIsInstance(result, list)

if __name__ == '__main__':
    unittest.main()