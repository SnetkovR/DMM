import unittest
import os
import numpy as np
from numpy.testing import assert_array_equal
from diophantine import DiophantineEquations

TEST_DIR = r"\test cases"


class SolutionTest(unittest.TestCase):

    def test_eye(self):
        """
        Единичная матрица
        """
        path = os.path.dirname(os.path.realpath(__file__))
        solver = DiophantineEquations(path + TEST_DIR + r"\test_eye.txt")
        res = solver.diophantine_equation_solver()
        self.assertEqual(
            res,
            [[1]],
            "Кейс с единичной матрицей некорректный!"
        )

    def test_square_one_solution(self):
        """
        Квадратная невырожденная матрица, решение единственное
        """
        path = os.path.dirname(os.path.realpath(__file__))
        solver = DiophantineEquations(path + TEST_DIR + r"\test_square_only_one")
        res = solver.diophantine_equation_solver()
        assert_array_equal(
            res,
            np.array([[1], [1]]),
            "Кейс с квадратичной невырожденной матрицей(1) некорректный!"
        )

    def test_square_no_solution(self):
        """
        Квадратная вырожденная матрица, решений нет
        """
        path = os.path.dirname(os.path.realpath(__file__))
        solver = DiophantineEquations(path + TEST_DIR + r"\test_square_no_solution")
        self.assertRaises(ValueError, solver.diophantine_equation_solver)

    def test_squaere_many_solutions(self):
        """
        Квадратная вырожденная матрица, решений множество
        """
        path = os.path.dirname(os.path.realpath(__file__))
        solver = DiophantineEquations(path + TEST_DIR + r"\test_squaere_many_solutions")
        res = solver.diophantine_equation_solver()
        assert_array_equal(
            res,
            np.array([[0], [0]]),
            "Кейс с квадратичной невырожденной матрицей(n) некорректный!"
        )

    def test_rectangular_no_solution(self):
        """
        Прямоугольная матрица, решений нет
        """
        path = os.path.dirname(os.path.realpath(__file__))
        solver = DiophantineEquations(path + TEST_DIR + r"\test_rectangular_no_solution")
        self.assertRaises(ValueError, solver.diophantine_equation_solver)

    def test_rectangular_many_solution(self):
        """
        Прямоугольная матрица, решений множество
        """
        path = os.path.dirname(os.path.realpath(__file__))
        solver = DiophantineEquations(path + TEST_DIR + r"\test_rectangular_many_solution")
        res = solver.diophantine_equation_solver()
        assert_array_equal(
            res,
            np.array([[488, -20], [-364, 15], [-682, 28]]),
            "Кейс с прямоугольной матрицей(т) некорректный!"
        )

    def test_square_no_solution_integer(self):
        """
        Квадратная невырожденная матрица, решений в целых числах нет
        """
        path = os.path.dirname(os.path.realpath(__file__))
        solver = DiophantineEquations(path + TEST_DIR + r"\test_square_no_solution_integer")
        self.assertRaises(ValueError, solver.diophantine_equation_solver)
