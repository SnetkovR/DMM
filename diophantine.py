import numpy as np
import sympy as sm

class DiophantineEquations():

    def __init__(self, filename="input.txt"):
        """
           Считывание входных данных

           :param flag: по умолчанию идет считывание из файла
           :return:
           n - число уравнений
           m - число неизвестных переменных
           eq - коэффициенты уравнений
       """
        with open(filename) as f:
            n, m = [int(x) for x in next(f).split()]
            eq = np.array([[int(x) for x in line.split()] for line in f])
            eq[:, -1] = - eq[:, -1]
            if any(x is None for x in
                   list(map(lambda x: print('ValueError') if x != m + 1 else x, list(map(len, eq))))):
                raise ValueError('Введено неверное число коэффициентов!')
            self.n = n
            self.m = m
            self.eq = eq.tolist()

    def diophantine_equation_solver(self):
        """
        Решение диофантовых уравнений (в случае 1 уравнения и системы)
        :param n: число уравнений
        :param m: число неизвестных переменных
        :param eq: коэффициенты уравнений
        :return: матрица В приведенная к эквивалентному виду (треугольная)
        """
        if self.n == 1:
            return self.one_dimension()
        else:
            return self.system_of_equations()

    def one_dimension(self):
        n = self.n
        m = self.m
        eq = self.eq
        B = self._build_matrix_B()
        flag = True
        while flag:
            # Шаг 1.
            # Выбираем в первой строке матрицы В наименьший
            # по абсолютной величине ненулевой элемент a[i]
            min_el = min(map(abs, B[0]))
            min_ind = list(map(abs, B[0])).index(min_el)
            for j in range(m):
                # Шаг 2.
                # Выбираем номер j!=i такой, что a[j]!=0
                if j != min_ind and B[0][j] != 0:
                    # Шаг 3.
                    # Делим с остатком на a[j], т.е. находим q и r, что
                    # a[j]=qa[i]+r
                    q = B[0][j] // B[0][min_ind]
                    for k in range(m + n):
                        # Шаг 4.
                        # Вычитаем из j-го столбца матрицы В i-й столбец
                        # умноженный на q
                        B[k][j] -= q * B[k][min_ind]
            # Шаг 5.
            # Если в первой строке более одного ненулевого числа, то выход,
            # иначе переходим на шаг 1
            if B[0].count(0) >= 1 or len(B[0]) == 1:
                flag = False
            else:
                flag = True
        # Свободный член уравнения
        c = - eq[0][-1]
        # Вычисляем индекс, где находится d=НОД()
        max_el = max(map(abs, B[0]))
        max_ind = list(map(abs, B[0])).index(max_el)
        # d=НОД()
        d = B[0][max_ind]
        # Если делится без остатка, то
        # вычисляем коэффициент
        # и выводим результат
        if c % d == 0:
            coef = c / d
            for i in range(1, n + m):
                B[i][max_ind] = coef * B[i][max_ind]
            tmp_b = np.array(B)
            B = np.c_[tmp_b[:, max_ind], np.delete(tmp_b, max_ind, 1)]
            B = np.delete(B, 0, 0).astype(int).tolist()
        # Иначе решений в целых числах нет
        else:
            raise ValueError('Решений в целых числах нет!')
        return B

    def system_of_equations(self):
        n_new, eq = self._del_null_and_lin_dep_row()
        B = self._build_matrix_B(n_new=n_new)
        m_new = len(eq)

        # Приводим матрицу к виду трапеции
        for i in range(m_new):
            b = [(B[k][i:-1]) for k in range(i, len(B))]
            b = np.array(b).astype(int).tolist()
            while not self._check_elem(b):
                min_el = list(map(lambda x: abs(x) if x != 0 else 1000, b[0]))
                i_min = min_el.index(min(min_el))
                max_el = list(map(lambda x: abs(x) if x != 0 else -1000, b[0]))
                i_max = len(max_el) - max_el[::-1].index(max(max_el)) - 1
                coef = b[0][i_max] // b[0][i_min]
                # Вычитаем из imax-го столбца матрицы В imin-й столбец
                # умноженный на coef
                for k in range(len(b)):
                    b[k][i_max] -= coef * b[k][i_min]
            if not b[0]:
                raise ValueError('Нет решения!')
            max_el = list(map(lambda x: abs(x) if x != 0 else -1000, b[0]))
            col = len(max_el) - max_el[::-1].index(max(max_el)) - 1
            b = [[row[col]] + row[:col] + row[col + 1:] for row in b]
            l = len(B)
            B = [B[k] if k < i else B[k][0:i] + b[k - i] + [int(B[k][-1])] for k in range(l)]
            # Проверка на отсутствие нулей на диагонали
            if B[i][i] == 0:
                raise ValueError('Нет решения!')
            else:
                coef = B[i][-1] // B[i][i]
            # Вычитаем из последнего столбца матрицы В i-й столбец
            # умноженный на coef
            for k in range(len(B)):
                B[k][-1] -= (coef * B[k][i])
        # Проверка, что в последний стлбец обнулился
        if [np.array(B)[i, -1] for i in range(m_new)] != m_new * [0]:
            raise ValueError('Нет решения!')
        size = len(eq[0]) - 1
        # Приводим к виду для печати результата
        tmp_b = np.array([B[m_new + i][m_new:] for i in range(size)])
        B = np.c_[tmp_b[:, -1], np.delete(tmp_b, -1, 1)]
        return B

    def _build_matrix_B(self, n_new=None):
        """
        Построение матрицы В для случая решений линейного уравнения и для решения СЛДУ
        :param n: число уравнений
        :param m: число неизвестных переменных
        :param eq: коэффициенты уравнений
        :param n_new:
        :return: матрица В
        """
        # Единичная матрица
        I = np.eye(self.m)
        # Если решается система, то еще добавляем нулевой вектор
        if self.n > 1:
            zero_vec = np.zeros(self.m)
        # Строим матрицу В
        B = list(map(np.ndarray.tolist, np.concatenate([self.eq, np.column_stack((I, zero_vec))]) \
            if self.n > 1 or n_new != None else np.row_stack([self.eq[0][:-1], I])))
        return B

    def _del_null_and_lin_dep_row(self):
        """
        Удаление нулевых и линейно-зависимых строк
        :param coef_m:
        :param n:
        :param m:
        :return:
        """
        # Удаляем нулевые строки
        n = self.n
        coef_m = [i for i in self.eq if i != [0] * self.m]
        if np.linalg.matrix_rank(coef_m) != self.n:
            # Удаляем линейно зависимые строки
            _, inds = sm.Matrix(coef_m).T.rref()
            coef_m = np.array(coef_m)[[inds]].tolist()
            n = len(coef_m)
        return n, coef_m

    def _check_elem(self, B):
        was_found = False
        for i in B[0]:
            if i != 0:
                if was_found:
                    return False
                was_found = True
        return True

    def print_and_write_result(self, B, flag='file'):
        """
        Печать результата на экран или в файл
        :param B: преобразованная матрица с решением уравнения
        :param flag:
        :return:
        """
        if flag == 'file':
            file = open('output.txt', 'w')
            file.write(str(len(B[0]) - 1) + '\n')
            for r in B:
                file.write(' '.join(map(str,r)) + '\n')
            file.close()
        else:
            print('Свободные переменные:', str(len(B[0]) - 1))
            for r in B:
                print(' + t *'.join(map(str, r)))












