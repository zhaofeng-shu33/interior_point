import unittest

import numpy as np
from lp_ip import lp_ip, lp_ip_pd
class LP_IP(unittest.TestCase):
    def test_basic(self):
        b = np.array([-1, -1], dtype=float)
        A = np.array([[1, 2, -1, 0], [2, 1, 0, -1]], dtype=float)
        c = np.array([1, 1, 0, 0], dtype=float)
        y0 = np.array([0.1, 0.1]) # exact interior point
        argmax_y, max_y = lp_ip(A, b, c, y0=y0)
        self.assertTrue(np.linalg.norm(argmax_y) < 1e-3)
        self.assertTrue(np.abs(max_y) < 1e-3)
    def test_vineyard(self):
        b = np.array([3, 4, 2], dtype=float)
        A = np.array([[2, 1, 0, -1, 0, 0], [0, 0, 3, 0, -1, 0], [0, 2, 1, 0, 0, -1]], dtype=float)
        c = np.array([4, 8, 6, 0, 0, 0], dtype=float)
        y0 = np.array([0.5, 0.5, 0.5]) # exact interior point
        argmax_y, max_y = lp_ip(A, b, c, y0=y0)

class LP_IP_PD(unittest.TestCase):
    def test_basic(self):
        b = np.array([-1, -1], dtype=float)
        A = np.array([[1, 2, -1, 0], [2, 1, 0, -1]], dtype=float)
        c = np.array([1, 1, 0, 0], dtype=float)
        argmax_y, max_y = lp_ip_pd(A, b, c)
        self.assertTrue(np.linalg.norm(argmax_y) < 1e-3)
        self.assertTrue(np.abs(max_y) < 1e-3)
    def test_vineyard(self):
        b = np.array([3, 4, 2], dtype=float)
        A = np.array([[2, 1, 0, -1, 0, 0], [0, 0, 3, 0, -1, 0], [0, 2, 1, 0, 0, -1]], dtype=float)
        c = np.array([4, 8, 6, 0, 0, 0], dtype=float)
        argmax_y, max_y = lp_ip_pd(A, b, c)

if __name__ == '__main__':
    unittest.main()