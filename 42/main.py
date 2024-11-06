import numpy as np
from typing import List, Tuple


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает данные из текстового файла, содержащего два ряда чисел,
    и преобразует их в массивы numpy.

    Предполагается, что файл `dataset.txt` содержит два ряда чисел:
    - Первая строка — значения X
    - Вторая строка — значения Y

    Returns:
        Tuple[np.ndarray, np.ndarray]: Кортеж из двух массивов numpy.
    """
    with open('42/dataset.txt', 'r') as f:
        x, y = f.readlines()

    # Преобразование строк в массивы целых чисел
    x = list(map(float, x.strip().split(' ')))
    y = list(map(float, y.strip().split(' ')))

    return np.array(x), np.array(y)


def polynomial_regression(X: np.ndarray, Y: np.ndarray, m: int) -> List[float]:
    """
    Выполняет полиномиальную регрессию степени m для данных X и Y, чтобы найти коэффициенты β_j.

    Args:
        X (np.ndarray): Массив значений X.
        Y (np.ndarray): Массив значений Y.
        m (int): Степень полинома.

    Returns:
        List[float]: Список коэффициентов β_j, начиная с β_0.
    """
    # Создаем матрицу Вандермонда для X, чтобы включить степени до m
    X_vander = np.vander(X, m + 1, increasing=True)

    # Решаем линейную систему для нахождения коэффициентов
    beta, *_ = np.linalg.lstsq(X_vander, Y, rcond=None)

    return beta.tolist()


if __name__ == '__main__':
    X, Y = load_data()

    print('=' * 50)
    m = int(input("Введите степень полинома m: "))
    beta = polynomial_regression(X, Y, m)

    # Вывод коэффициентов
    print("Найденные коэффициенты β_j:")
    for j, coeff in enumerate(beta):
        print(f"β_{j} = {coeff:.6f}")

    # Сумма модулей коэффициентов
    sum_abs_beta = sum(abs(b) for b in beta)
    print(f"\nСумма модулей коэффициентов: {sum_abs_beta:.4f}")
    print('=' * 50)