"""Extractive Sentence-Level Context Compressor.

Extracts high-density salient fact sentences from retrieved chunks, dropping
redundant boilerplate and preamble to compress LLM prompt token costs by 50%~70%.
"""

import re
from typing import Optional
from rag_gatekeeper.models import ChunkItem


class ContextCompressor:
    """Compresses document chunks down to their most informative sentence subsets."""

    def __init__(self, top_sentences_per_chunk: int = 2, min_sentence_chars: int = 10):
        self.top_sentences_per_chunk = top_sentences_per_chunk
        self.min_sentence_chars = min_sentence_chars
        self.sentence_regex = re.compile(r"[^。！？\n!?]+[。！？\n!?]?")

    def _tokenize(self, text: str) -> set[str]:
        return set(re.findall(r"[\u4e00-\u9fa5]|[a-zA-Z0-9]+", text.lower()))

    def compress_chunk(self, query: str, chunk: ChunkItem) -> ChunkItem:
        """Compress a single chunk by retaining only top-scoring factual sentences."""
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return chunk

        sentences = [s.strip() for s in self.sentence_regex.findall(chunk.content) if len(s.strip()) >= self.min_sentence_chars]
        if len(sentences) <= self.top_sentences_per_chunk:
            return chunk

        # Score each sentence by query token overlap
        scored_sentences: list[tuple[int, str, float]] = []
        for idx, sent in enumerate(sentences):
            s_toks = self._tokenize(sent)
            if not s_toks:
                continue
            overlap = len(q_tokens & s_toks) / len(q_tokens)
            scored_sentences.append((idx, sent, overlap))

        if not scored_sentences:
            return chunk

        # Sort by overlap descending, pick top N
        scored_sentences.sort(key=lambda x: x[2], reverse=True)
        chosen = scored_sentences[: self.top_sentences_per_chunk]
        # Restore chronological order
        chosen.sort(key=lambda x: x[0])

        compressed_text = " ".join([c[1] for c in chosen])
        return ChunkItem(
            id=chunk.id,
            content=compressed_text,
            metadata={**chunk.metadata, "compressed": True, "original_len": len(chunk.content), "compressed_len": len(compressed_text)},
        )

    def compress_chunks(self, query: str, chunks: list[ChunkItem]) -> list[ChunkItem]:
        """Compress a list of chunks."""
        return [self.compress_chunk(query, c) for c in chunks]


# Global singleton compressor
context_compressor = ContextCompressor()
