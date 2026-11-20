"""High-performance multi-pattern keyword matcher (Aho-Corasick automaton).

Zero external binary dependencies with pure Python fallback for reliable cross-platform execution.
"""

from collections import deque
from typing import Optional


class AhoCorasickMatcher:
    """Aho-Corasick string matching automaton for ultra-fast enterprise keyword lookups.
    
    Provides O(N) multi-pattern matching time complexity where N is query length,
    independent of the number of registered keywords.
    """

    def __init__(self, keywords: Optional[list[str]] = None):
        # Trie node structure: {char: next_node_index}
        self.trie: list[dict[str, int]] = [{}]
        # Output list: keywords that end at this node
        self.output: list[list[str]] = [[]]
        # Fail pointer for each node
        self.fail: list[int] = [0]
        self._built = False

        if keywords:
            for kw in keywords:
                self.add_word(kw)
            self.build()

    def add_word(self, word: str):
        """Add a keyword to the trie."""
        word = word.strip()
        if not word:
            return
        self._built = False
        curr = 0
        for ch in word:
            if ch not in self.trie[curr]:
                new_node = len(self.trie)
                self.trie.append({})
                self.output.append([])
                self.fail.append(0)
                self.trie[curr][ch] = new_node
            curr = self.trie[curr][ch]
        self.output[curr].append(word)

    def build(self):
        """Construct fail pointers using BFS."""
        queue: deque[int] = deque()
        # Initialize depth 1 nodes
        for ch, next_node in self.trie[0].items():
            self.fail[next_node] = 0
            queue.append(next_node)

        while queue:
            curr = queue.popleft()
            for ch, next_node in self.trie[curr].items():
                queue.append(next_node)
                # Find fail pointer for next_node
                f = self.fail[curr]
                while f > 0 and ch not in self.trie[f]:
                    f = self.fail[f]
                if ch in self.trie[f] and self.trie[f][ch] != next_node:
                    self.fail[next_node] = self.trie[f][ch]
                else:
                    self.fail[next_node] = 0
                # Merge outputs
                self.output[next_node].extend(self.output[self.fail[next_node]])

        self._built = True

    def has_any(self, text: str) -> bool:
        """Return True immediately on first keyword match."""
        if not self._built:
            self.build()
        curr = 0
        for ch in text:
            while curr > 0 and ch not in self.trie[curr]:
                curr = self.fail[curr]
            if ch in self.trie[curr]:
                curr = self.trie[curr][ch]
                if self.output[curr]:
                    return True
        return False

    def search_all(self, text: str) -> list[str]:
        """Find all matched keywords in text."""
        if not self._built:
            self.build()
        matches = []
        curr = 0
        for ch in text:
            while curr > 0 and ch not in self.trie[curr]:
                curr = self.fail[curr]
            if ch in self.trie[curr]:
                curr = self.trie[curr][ch]
                if self.output[curr]:
                    matches.extend(self.output[curr])
        return list(set(matches))
