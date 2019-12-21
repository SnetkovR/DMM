import numpy as np
import sys
from diophantine import DiophantineEquations


test = np.array([[2, 1, 0, -3],
                 [-2, 0, 1, -3],
                 [0, 1, 1, 6]])


if __name__ == "__main__":

    try:
        if len(sys.argv) == 2:
            solver = DiophantineEquations(sys.argv[1])
        else:
            solver = DiophantineEquations()
        res = solver.diophantine_equation_solver()
        solver.print_and_write_result(res, flag='file')
    except ValueError as e:
        with open("output.txt", "w") as f:
            f.write("No solution")
