import numpy as np
from typing import List

# Константы для градиентного спуска
MAX_ITERATIONS: int = 100_000   # Максимальное количество итераций
ALPHA: float = 0.01             # Шаг градиентного спуска
TOLERANCE: float = 1e-6         # Порог для остановки алгоритма (точность)
SIZE: int = 6                   # Размерность вектора


def load_A() -> np.ndarray:
    """
    Загружает матрицу A из текстового файла и преобразует её в numpy array.

    Returns:
        np.ndarray: Матрица A размера (6, 6).
    """
    with open('40/A.txt', 'r') as f:
        data = f.readlines()

    # Преобразование строк в целочисленную матрицу
    data = [list(map(int, line.split())) for line in data]
    return np.array(data)


def load_B() -> np.ndarray:
    """
    Загружает вектор B из текстового файла и преобразует его в numpy array.

    Returns:
        np.ndarray: Вектор B размера (6,).
    """
    with open('40/B.txt', 'r') as f:
        data = f.readline().split()
    return np.array(list(map(int, data)))


def solution(A: np.ndarray, B: np.ndarray) -> float:
    """
    Реализует метод градиентного спуска для нахождения минимального значения функции
    f(y) = (Ay, y) - 2(b, y) и вычисляет норму результата ||y*||_2 с точностью до 6 знаков.

    Args:
        A (np.ndarray): Матрица размера (6, 6).
        B (np.ndarray): Вектор размера (6,).

    Returns:
        float: Евклидова норма оптимального вектора y*, округленная до 6 знаков.
    """
    # Инициализация начальной точки y
    y = np.zeros(SIZE)

    # Градиентный спуск
    for _ in range(MAX_ITERATIONS):
        grad = 2 * A @ y - 2 * B  # Вычисление градиента

        # Условие остановки по норме градиента
        if np.linalg.norm(grad) < TOLERANCE:
            break

        # Шаг обновления y
        y = y - ALPHA * grad

    # Нахождение нормы оптимального y и округление до 6 знаков
    norm_y_star = np.linalg.norm(y)
    norm_y_star_rounded = round(norm_y_star, 6)
    return norm_y_star_rounded


if __name__ == '__main__':
    A = load_A()
    B = load_B()

    print('=' * 50)
    print(f'Матрица A:\n{A}')
    print(f'Вектор B: {B}\n')
    print(
        f'Параметры запуска:\nalpha={ALPHA}, max_iter={MAX_ITERATIONS}, tolerance={TOLERANCE}\nРезультат: {solution(A, B)}'
    )
    print('=' * 50)
