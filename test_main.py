import unittest
from main import fetch_and_train

class TestMain(unittest.TestCase):
    def test_fetch_and_train(self):
        mse = fetch_and_train()
        self.assertIsInstance(mse, float)
        self.assertGreater(mse, 0)

if __name__ == '__main__':
    unittest.main()