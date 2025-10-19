# tinybloom.py

import hashlib

class TinyBloom:
    def __init__(self, m_bits: int = 4_000_000, k: int = 7):
        self.m = m_bits
        self.k = k
        self.bits = bytearray(m_bits // 8 + 1)

    def _hashes(self, s: str):
        # Deterministic hashing for legacy/compat cases too
        msg = s.encode("utf-8", "ignore")
        base = hashlib.blake2b(msg, digest_size=32).digest()  # 256 bits

        ints = []
        for i in range(0, len(base), 8):
            ints.append(int.from_bytes(base[i:i+8], "big"))

        j = 0
        while len(ints) < self.k:
            j += 1
            ext = hashlib.blake2b(base + j.to_bytes(2, "big"), digest_size=16).digest()
            ints.append(int.from_bytes(ext[:8], "big"))

        for i in range(self.k):
            yield ints[i] % self.m

    def add(self, s: str):
        for h in self._hashes(s):
            self.bits[h // 8] |= (1 << (h % 8))

    def __contains__(self, s: str) -> bool:
        for h in self._hashes(s):
            if not (self.bits[h // 8] & (1 << (h % 8))):
                return False
        return True
