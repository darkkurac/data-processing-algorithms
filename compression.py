"""Учебные реализации алгоритмов сжатия RLE, LZW и Хаффмана."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq
from itertools import count
from pathlib import Path
import struct
from typing import TypeAlias


RleData: TypeAlias = list[tuple[int, int]]


def rle_compress(data: bytes) -> RleData:
    """Сжать последовательность байтов методом RLE."""
    if not data:
        return []

    compressed: RleData = []
    current = data[0]
    run_length = 1

    for value in data[1:]:
        if value == current:
            run_length += 1
        else:
            compressed.append((current, run_length))
            current = value
            run_length = 1
    compressed.append((current, run_length))
    return compressed


def rle_decompress(compressed: RleData) -> bytes:
    """Восстановить байты из RLE-представления."""
    result = bytearray()
    for value, run_length in compressed:
        if not 0 <= value <= 255 or run_length < 1:
            raise ValueError("Некорректные данные RLE.")
        result.extend([value] * run_length)
    return bytes(result)


def lzw_compress(data: bytes) -> list[int]:
    """Сжать байты стандартным вариантом LZW."""
    if not data:
        return []

    dictionary = {bytes([value]): value for value in range(256)}
    next_code = 256
    current = b""
    compressed: list[int] = []

    for value in data:
        candidate = current + bytes([value])
        if candidate in dictionary:
            current = candidate
        else:
            compressed.append(dictionary[current])
            dictionary[candidate] = next_code
            next_code += 1
            current = bytes([value])

    if current:
        compressed.append(dictionary[current])
    return compressed


def lzw_decompress(compressed: list[int]) -> bytes:
    """Восстановить байты из списка кодов LZW, не изменяя входной список."""
    if not compressed:
        return b""

    dictionary = {value: bytes([value]) for value in range(256)}
    next_code = 256
    first_code = compressed[0]
    if first_code not in dictionary:
        raise ValueError("Некорректный первый код LZW.")

    previous = dictionary[first_code]
    result = bytearray(previous)

    for code in compressed[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == next_code:
            entry = previous + previous[:1]
        else:
            raise ValueError("Некорректные данные LZW.")

        result.extend(entry)
        dictionary[next_code] = previous + entry[:1]
        next_code += 1
        previous = entry

    return bytes(result)


@dataclass(slots=True)
class HuffmanNode:
    """Узел дерева Хаффмана."""

    value: int | None
    frequency: int
    left: HuffmanNode | None = None
    right: HuffmanNode | None = None


def huffman_tree(data: bytes) -> HuffmanNode | None:
    """Построить дерево Хаффмана для последовательности байтов."""
    if not data:
        return None

    order = count()
    heap: list[tuple[int, int, HuffmanNode]] = [
        (frequency, next(order), HuffmanNode(value, frequency))
        for value, frequency in Counter(data).items()
    ]
    heapq.heapify(heap)

    while len(heap) > 1:
        _, _, left = heapq.heappop(heap)
        _, _, right = heapq.heappop(heap)
        parent = HuffmanNode(
            None,
            left.frequency + right.frequency,
            left=left,
            right=right,
        )
        heapq.heappush(heap, (parent.frequency, next(order), parent))
    return heap[0][2]


def huffman_codes(tree: HuffmanNode | None) -> dict[int, str]:
    """Получить двоичные коды для листьев дерева Хаффмана."""
    if tree is None:
        return {}

    codes: dict[int, str] = {}

    def visit(node: HuffmanNode, prefix: str) -> None:
        if node.value is not None:
            codes[node.value] = prefix or "0"
            return
        if node.left is not None:
            visit(node.left, prefix + "0")
        if node.right is not None:
            visit(node.right, prefix + "1")

    visit(tree, "")
    return codes


def huffman_compress(data: bytes) -> tuple[str, HuffmanNode | None]:
    """Сжать байты кодами Хаффмана."""
    tree = huffman_tree(data)
    codes = huffman_codes(tree)
    return "".join(codes[value] for value in data), tree


def huffman_decompress(compressed: str, tree: HuffmanNode | None) -> bytes:
    """Восстановить байты из двоичных кодов и дерева Хаффмана."""
    if tree is None:
        if compressed:
            raise ValueError("Для непустых данных требуется дерево Хаффмана.")
        return b""

    if tree.value is not None:
        if any(bit != "0" for bit in compressed):
            raise ValueError("Некорректный код для односимвольного дерева.")
        return bytes([tree.value]) * len(compressed)

    result = bytearray()
    node = tree
    for bit in compressed:
        if bit == "0":
            node = node.left
        elif bit == "1":
            node = node.right
        else:
            raise ValueError("Код Хаффмана должен содержать только 0 и 1.")

        if node is None:
            raise ValueError("Некорректный код Хаффмана.")
        if node.value is not None:
            result.append(node.value)
            node = tree

    if node is not tree:
        raise ValueError("Код Хаффмана закончился внутри ветви дерева.")
    return bytes(result)


@dataclass(frozen=True, slots=True)
class CompressionResult:
    """Итоги сжатия и контрольной распаковки."""

    method: str
    original_size: int
    encoded_size: int
    payload: bytes
    verified: bool

    @property
    def ratio(self) -> float:
        return self.encoded_size / self.original_size if self.original_size else 0.0


def _pack_bits(bits: str) -> bytes:
    if not bits:
        return b""
    padding = (-len(bits)) % 8
    padded = bits + "0" * padding
    body = int(padded, 2).to_bytes(len(padded) // 8, byteorder="big")
    return struct.pack(">Q", len(bits)) + body


def compress_bytes(data: bytes, method: str) -> CompressionResult:
    """Сжать байты выбранным методом и проверить обратное преобразование."""
    normalized = method.strip().lower()

    if normalized == "rle":
        encoded = rle_compress(data)
        restored = rle_decompress(encoded)
        payload = b"".join(
            struct.pack(">BI", value, run_length)
            for value, run_length in encoded
        )
    elif normalized == "lzw":
        encoded = lzw_compress(data)
        restored = lzw_decompress(encoded)
        payload = b"".join(struct.pack(">I", code) for code in encoded)
    elif normalized == "huffman":
        encoded, tree = huffman_compress(data)
        restored = huffman_decompress(encoded, tree)
        payload = _pack_bits(encoded)
    else:
        raise ValueError(f"Неизвестный метод сжатия: {method}")

    return CompressionResult(
        method=normalized,
        original_size=len(data),
        encoded_size=len(payload),
        payload=payload,
        verified=restored == data,
    )


def main() -> None:
    """Запустить интерактивную демонстрацию алгоритмов сжатия."""
    print("Инструмент для сжатия данных и изображений")
    source_path = Path(input("Введите путь к файлу: ").strip().strip('"'))
    if not source_path.is_file():
        print("Файл не найден.")
        return

    print("Выберите метод сжатия:")
    print("1. RLE")
    print("2. LZW")
    print("3. Хаффман")
    choice = input("Введите номер метода: ").strip()
    methods = {"1": "rle", "2": "lzw", "3": "huffman"}
    if choice not in methods:
        print("Неверный выбор метода.")
        return

    data = source_path.read_bytes()
    result = compress_bytes(data, methods[choice])
    output_path = source_path.with_name(
        f"{source_path.stem}.{result.method}.bin"
    )
    output_path.write_bytes(result.payload)

    print(f"Метод: {result.method.upper()}")
    print(f"Исходный размер: {result.original_size} байт")
    print(f"Размер кодированного представления: {result.encoded_size} байт")
    print(f"Коэффициент размера: {result.ratio:.2f}")
    print("Контрольная распаковка: успешно" if result.verified else "Ошибка проверки")
    print(f"Результат сохранён: {output_path}")


if __name__ == "__main__":
    main()
