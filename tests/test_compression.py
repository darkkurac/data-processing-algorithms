import unittest

import compression


class CompressionTests(unittest.TestCase):
    def test_rle_round_trip(self):
        source = b"AAAABBBCCDAA"
        encoded = compression.rle_compress(source)
        self.assertEqual(compression.rle_decompress(encoded), source)

    def test_rle_supports_empty_input(self):
        self.assertEqual(compression.rle_compress(b""), [])
        self.assertEqual(compression.rle_decompress([]), b"")

    def test_lzw_round_trip_without_mutating_codes(self):
        source = b"TOBEORNOTTOBEORTOBEORNOT"
        encoded = compression.lzw_compress(source)
        original_codes = encoded.copy()
        self.assertEqual(compression.lzw_decompress(encoded), source)
        self.assertEqual(encoded, original_codes)

    def test_lzw_supports_empty_input(self):
        self.assertEqual(compression.lzw_compress(b""), [])
        self.assertEqual(compression.lzw_decompress([]), b"")

    def test_huffman_round_trip(self):
        source = b"this is an example for a huffman tree"
        encoded, tree = compression.huffman_compress(source)
        self.assertEqual(compression.huffman_decompress(encoded, tree), source)

    def test_huffman_supports_empty_input(self):
        encoded, tree = compression.huffman_compress(b"")
        self.assertEqual((encoded, tree), ("", None))
        self.assertEqual(compression.huffman_decompress(encoded, tree), b"")

    def test_huffman_supports_single_repeated_byte(self):
        source = b"AAAAAA"
        encoded, tree = compression.huffman_compress(source)
        self.assertEqual(encoded, "0" * len(source))
        self.assertEqual(compression.huffman_decompress(encoded, tree), source)

    def test_compress_bytes_verifies_every_algorithm(self):
        source = b"AAAABBBCCDAA"
        for method in ("rle", "lzw", "huffman"):
            with self.subTest(method=method):
                result = compression.compress_bytes(source, method)
                self.assertEqual(result.original_size, len(source))
                self.assertGreater(result.encoded_size, 0)
                self.assertTrue(result.verified)


if __name__ == "__main__":
    unittest.main()
