import numpy as np
from diophantine import DiophantineEquations

test = np.array([[3, 4, 0, -8],
                 [7, 0, 5, -6]])
solver = DiophantineEquations("input.txt")
res = solver.diophantine_equation_solver()
solver._print_and_write_result(res, flag='file')

