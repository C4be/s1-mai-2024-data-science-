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
        Tuple[np.ndarray, np.ndarray]: Кортеж из двух массивов numpy:
        - x: массив значений X (независимая переменная)
        - y: массив значений Y (зависимая переменная)
    """
    with open('41/dataset.txt', 'r') as f:
        x, y = f.readlines()

    # Преобразование строк в массивы целых чисел
    x = list(map(int, x.strip().split(' ')))
    y = list(map(int, y.strip().split(' ')))

    return np.array(x), np.array(y)


def solution(x: np.ndarray, y: np.ndarray, n: int) -> Tuple[float, float, float]:
    """
    Вычисляет параметры линейной регрессии y = ax + b для заданных массивов X и Y
    методом наименьших квадратов, а также возвращает сумму модулей параметров a и b.

    Args:
        x (np.ndarray): Массив значений независимой переменной X.
        y (np.ndarray): Массив значений зависимой переменной Y.
        n (int): Количество точек данных (размер массива X или Y).

    Returns:
        Tuple[float, float, float]: Кортеж, содержащий:
        - a: коэффициент наклона (параметр при X)
        - b: свободный член (сдвиг)
        - abs(a) + abs(b): сумма модулей коэффициентов a и b
    """
    # Вычисление коэффициента наклона a
    a = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
        (n * np.sum(x**2) - (np.sum(x))**2)
    # Вычисление свободного члена b
    b = (np.sum(y) - a * np.sum(x)) / n

    # Возврат значений a, b и их суммы модулей
    return a, b, abs(a) + abs(b)


if __name__ == '__main__':
    x, y = load_data()
    n = len(x)
    a, b, sol = solution(x, y, n)

    print('=' * 50)
    print(f'Начальные данные:\nx -> {x}\ny -> {y}\n')
    print(f'Найденные параметры:\na -> {a} b -> {b}')
    print(f'Сумма модулей параметров: {round(sol, 6)}')
    print('=' * 50)
