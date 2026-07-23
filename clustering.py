"""Иерархическая кластеризация методом полной связи."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


SUPPORTED_METRICS = {"euclidean", "squared_euclidean"}


def squared_euclidean_distance(
    point1: Sequence[float], point2: Sequence[float]
) -> float:
    """Вычислить квадрат евклидова расстояния между двумя точками."""
    difference = np.asarray(point1, dtype=float) - np.asarray(point2, dtype=float)
    return float(np.dot(difference, difference))


def calculate_distance_matrix(
    data: Sequence[Sequence[float]], distance_metric: str = "euclidean"
) -> np.ndarray:
    """Вычислить симметричную матрицу попарных расстояний."""
    if distance_metric not in SUPPORTED_METRICS:
        raise ValueError(f"Неизвестная метрика расстояния: {distance_metric}")

    points = np.asarray(data, dtype=float)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("Данные должны содержать хотя бы одну точку.")

    difference = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    squared = np.sum(difference**2, axis=2)
    if distance_metric == "squared_euclidean":
        return squared
    return np.sqrt(squared)


def complete_linkage_clustering(
    data: Sequence[Sequence[float]],
    n_clusters: int = 2,
    distance_metric: str = "euclidean",
    *,
    verbose: bool = False,
) -> np.ndarray:
    """Разделить точки методом полной связи на заданное число кластеров."""
    points = np.asarray(data, dtype=float)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("Данные должны содержать хотя бы одну точку.")
    if not 1 <= n_clusters <= len(points):
        raise ValueError("Число кластеров должно быть от 1 до числа точек.")

    distances = calculate_distance_matrix(points, distance_metric)
    clusters: list[list[int]] = [[index] for index in range(len(points))]
    iteration = 1

    while len(clusters) > n_clusters:
        best_pair: tuple[int, int] | None = None
        best_distance = float("inf")

        for left in range(len(clusters) - 1):
            for right in range(left + 1, len(clusters)):
                linkage = max(
                    distances[first, second]
                    for first in clusters[left]
                    for second in clusters[right]
                )
                if linkage < best_distance:
                    best_distance = float(linkage)
                    best_pair = (left, right)

        if best_pair is None:
            raise RuntimeError("Не удалось выбрать кластеры для объединения.")

        left, right = best_pair
        if verbose:
            print(
                f"Итерация {iteration}: объединяем кластеры "
                f"{clusters[left]} и {clusters[right]} "
                f"(расстояние {best_distance:.3f})"
            )
        clusters[left].extend(clusters[right])
        del clusters[right]
        iteration += 1

    labels = np.empty(len(points), dtype=int)
    for label, cluster in enumerate(clusters):
        labels[cluster] = label
    return labels


def most_distant_neighbor_clustering(
    data: Sequence[Sequence[float]], distance_metric: str = "euclidean"
) -> np.ndarray:
    """Совместимый интерфейс исходной лабораторной работы."""
    return complete_linkage_clustering(data, 2, distance_metric, verbose=True)


def plot_clusters(
    data: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    save_path: str | None = None,
) -> None:
    """Показать диаграмму кластеров или сохранить её в файл."""
    import matplotlib.pyplot as plt

    points = np.asarray(data, dtype=float)
    figure, axes = plt.subplots(figsize=(8, 5))
    scatter = axes.scatter(
        points[:, 0],
        points[:, 1],
        c=np.asarray(labels),
        cmap="viridis",
        s=90,
        edgecolors="white",
        linewidths=0.8,
    )
    axes.set_title("Результаты кластеризации")
    axes.set_xlabel("Признак 1")
    axes.set_ylabel("Признак 2")
    axes.grid(alpha=0.2)
    figure.colorbar(scatter, ax=axes, label="Кластер")
    figure.tight_layout()

    if save_path:
        figure.savefig(save_path, dpi=160)
        plt.close(figure)
    else:
        plt.show()


def _read_positive_integer(prompt: str) -> int:
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
        except ValueError:
            pass
        print("Введите целое положительное число.")


def main() -> None:
    """Запустить интерактивную демонстрацию кластеризации."""
    num_samples = _read_positive_integer("Введите количество объектов: ")
    generate_random = input("Сгенерировать случайные данные? (да/нет): ").strip().lower()

    if generate_random == "да":
        data = np.random.default_rng().uniform(0, 10, size=(num_samples, 2))
    else:
        rows: list[list[float]] = []
        print("Введите по две координаты для каждого объекта:")
        while len(rows) < num_samples:
            try:
                row = [float(value) for value in input().split()]
                if len(row) != 2:
                    raise ValueError
                rows.append(row)
            except ValueError:
                print("Нужно ввести ровно два числа через пробел.")
        data = np.asarray(rows)

    print("Выберите метрику расстояния:")
    print("1. Евклидово расстояние")
    print("2. Квадрат евклидова расстояния")
    metric_choice = input("Введите номер метрики: ").strip()
    metrics = {"1": "euclidean", "2": "squared_euclidean"}
    if metric_choice not in metrics:
        print("Неверный выбор метрики.")
        return

    labels = complete_linkage_clustering(
        data,
        n_clusters=min(2, num_samples),
        distance_metric=metrics[metric_choice],
        verbose=True,
    )
    print(f"Метки кластеров: {labels.tolist()}")
    plot_clusters(data, labels)


if __name__ == "__main__":
    main()
