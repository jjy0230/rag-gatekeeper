"""Citation & Source Footnotes Injector.

Automatically grounds answers with [1], [2] superscript citations and appends
auditable reference footnotes, transforming AI outputs into verifiable corporate knowledge.
"""

import re
from typing import Optional
from collections import Counter
from pydantic import BaseModel, Field
from rag_gatekeeper.models import ChunkItem


class CitationSource(BaseModel):
    index: int
    chunk_id: str
    snippet: str
    relevance_score: float
    metadata: dict = Field(default_factory=dict)


class CitationResult(BaseModel):
    annotated_answer: str
    citations: list[CitationSource]
    citations_count: int


class CitationInjector:
    """Injects footnote markers [1], [2] into generated answers and appends Markdown references."""

    def __init__(self, min_match_threshold: float = 0.25):
        self.min_match_threshold = min_match_threshold
        self.sentence_regex = re.compile(r"([^。！？\n!?]+[。！？\n!?]?)")

    def _tokenize(self, text: str) -> set[str]:
        return set(re.findall(r"[\u4e00-\u9fa5]|[a-zA-Z0-9]+", text.lower()))

    def _calculate_overlap(self, set_a: set[str], set_b: set[str]) -> float:
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a)

    def inject_citations(
        self,
        answer: str,
        evidence_chunks: list[ChunkItem],
    ) -> CitationResult:
        """Inject citation markers into answer and generate references list."""
        if not evidence_chunks or not answer.strip():
            return CitationResult(annotated_answer=answer, citations=[], citations_count=0)

        # Tokenize all candidate chunks
        chunk_tokens = [self._tokenize(c.content) for c in evidence_chunks]

        # Split answer into distinct sentences
        raw_sentences = [s for s in self.sentence_regex.findall(answer) if s.strip()]
        if not raw_sentences:
            raw_sentences = [answer]

        annotated_sentences = []
        cited_chunk_indices: dict[int, int] = {}  # chunk_idx -> citation_number
        citation_sources: list[CitationSource] = []

        for sent in raw_sentences:
            s_tokens = self._tokenize(sent)
            # Find best matching chunk
            best_chunk_idx = -1
            best_overlap = 0.0

            for idx, c_toks in enumerate(chunk_tokens):
                overlap = self._calculate_overlap(s_tokens, c_toks)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_chunk_idx = idx

            # If overlap exceeds threshold, assign citation
            if best_overlap >= self.min_match_threshold and best_chunk_idx >= 0:
                if best_chunk_idx not in cited_chunk_indices:
                    new_cite_num = len(cited_chunk_indices) + 1
                    cited_chunk_indices[best_chunk_idx] = new_cite_num
                    chunk_obj = evidence_chunks[best_chunk_idx]
                    citation_sources.append(
                        CitationSource(
                            index=new_cite_num,
                            chunk_id=chunk_obj.id,
                            snippet=chunk_obj.content[:120] + ("..." if len(chunk_obj.content) > 120 else ""),
                            relevance_score=round(best_overlap, 3),
                            metadata=chunk_obj.metadata,
                        )
                    )
                cite_num = cited_chunk_indices[best_chunk_idx]
                
                # Insert [N] before ending punctuation
                if sent.endswith(("。", "！", "？", "\n", "!", "?", ".")):
                    punct = sent[-1]
                    annotated_sent = sent[:-1] + f" [{cite_num}]" + punct
                else:
                    annotated_sent = sent + f" [{cite_num}]"
                annotated_sentences.append(annotated_sent)
            else:
                annotated_sentences.append(sent)

        # Assemble full text
        body = "".join(annotated_sentences)

        # Append Footnotes Section if citations exist
        if citation_sources:
            footnotes = ["\n\n---\n**参考来源：**"]
            for cs in citation_sources:
                title = cs.metadata.get("title") or cs.metadata.get("source") or f"切片 {cs.chunk_id}"
                footnotes.append(f"- [{cs.index}] **{title}**: {cs.snippet} *(匹配度: {int(cs.relevance_score * 100)}%)*")
            body += "\n".join(footnotes)

        return CitationResult(
            annotated_answer=body,
            citations=citation_sources,
            citations_count=len(citation_sources),
        )


# Global singleton injector
citation_injector = CitationInjector()
