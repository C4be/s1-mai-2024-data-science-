import numpy as np
from typing import Tuple, List


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
    with open('43/dataset.txt', 'r') as f:
        x, y = f.readlines()

    # Преобразование строк в массивы целых чисел
    x = list(map(float, x.strip().split(' ')))
    y = list(map(float, y.strip().split(' ')))

    return np.array(x), np.array(y)


def logarithmic_regression(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
    """
    Выполняет линейную регрессию для модели y = β0 + β1 * ln(x),
    используя логарифмическое преобразование X.

    Args:
        X (np.ndarray): Массив значений независимой переменной X.
        Y (np.ndarray): Массив значений зависимой переменной Y.

    Returns:
        Tuple[float, float]: Кортеж, содержащий параметры:
        - β0 (свободный член)
        - β1 (коэффициент при ln(x))
    """
    # Преобразование X в логарифмические значения
    X_log = np.log(X)

    # Решение задачи линейной регрессии Y = β0 + β1 * X_log
    A = np.vstack([np.ones_like(X_log), X_log]).T
    beta, _ = np.linalg.lstsq(A, Y, rcond=None)[0:2]

    return beta[0], beta[1]


if __name__ == '__main__':
    X, Y = load_data()

    beta_0, beta_1 = logarithmic_regression(X, Y)

    print('=' * 50)
    print(f"Найденные коэффициенты модели y = β0 + β1 * ln(x):")
    print(f"β0 = {beta_0:.6f}")
    print(f"β1 = {beta_1:.6f}")

    sum_abs_beta = abs(beta_0) + abs(beta_1)
    print(f"\nСумма модулей коэффициентов: {sum_abs_beta:.4f}")
    print('=' * 50)
