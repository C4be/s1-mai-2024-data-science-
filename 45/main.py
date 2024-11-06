import numpy as np
from typing import Tuple


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
    with open('45/dataset.txt', 'r') as f:
        x, y = f.readlines()

    # Преобразование строк в массивы целых чисел
    x = list(map(float, x.strip().split(' ')))
    y = list(map(float, y.strip().split(' ')))

    return np.array(x), np.array(y)


def regression_with_sine(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float, float]:
    """
    Выполняет линейную регрессию для модели y = β0 + β1 * x + β2 * sin(8 * x).

    Args:
        X (np.ndarray): Массив значений независимой переменной X.
        Y (np.ndarray): Массив значений зависимой переменной Y.

    Returns:
        Tuple[float, float, float]: Кортеж, содержащий параметры:
        - β0 (свободный член),
        - β1 (коэффициент при x),
        - β2 (коэффициент при sin(8 * x))
    """
    # Создаем столбцы для регрессии: [1, x, sin(8x)]
    A = np.vstack([np.ones_like(X), X, np.sin(8 * X)]).T

    # Решаем линейную систему Y = A @ [β0, β1, β2]
    beta, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)

    return beta[0], beta[1], beta[2]


if __name__ == '__main__':
    # Загрузка данных
    X, Y = load_data()

    # Нахождение коэффициентов
    beta_0, beta_1, beta_2 = regression_with_sine(X, Y)

    # Вывод коэффициентов
    print('=' * 50)
    print("Найденные коэффициенты модели y = β0 + β1 * x + β2 * sin(8 * x):")
    print(f"β0 = {beta_0:.6f}")
    print(f"β1 = {beta_1:.6f}")
    print(f"β2 = {beta_2:.6f}")

    # Сумма модулей коэффициентов
    sum_abs_beta = abs(beta_0) + abs(beta_1) + abs(beta_2)
    print(f"\nСумма модулей коэффициентов: {sum_abs_beta:.6f}")
    print('=' * 50)
