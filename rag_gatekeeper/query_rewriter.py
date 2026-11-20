"""Query Rewriter and Pronoun De-referencing Gatekeeper.

Resolves conversational pronouns ('它', '那个', '其') and expands underspecified
queries using short multi-turn dialog history.
"""

import re
from typing import Optional


class QueryRewriter:
    """Detects and resolves pronouns and ambiguous references in multi-turn dialogues."""

    PRONOUN_PATTERNS = [
        re.compile(r"^(它|它的|那个|这部分|具体呢|怎么说|为什么呢|那如何|那具体|其|此项)\b"),
        re.compile(r"(它|那个|这项目|该规定|此规范)"),
    ]

    def rewrite_query(self, current_query: str, last_turn_query: Optional[str] = None) -> tuple[str, bool]:
        """Rewrite current query if pronouns or referential ambiguity detected.
        
        Returns:
            (rewritten_query, was_rewritten)
        """
        curr = current_query.strip()
        if not last_turn_query or not last_turn_query.strip():
            return curr, False

        # Check if current query contains referential pronouns
        has_pronoun = any(p.search(curr) for p in self.PRONOUN_PATTERNS)

        if has_pronoun or len(curr) <= 4:
            # Extract key entity from last turn query (strip question marks and stop particles)
            cleaned_prior = re.sub(r"[？?吗呢嘛了如何怎样是什么是多少]", "", last_turn_query.strip())
            if cleaned_prior and cleaned_prior not in curr:
                rewritten = f"关于【{cleaned_prior}】: {curr}"
                return rewritten, True

        return curr, False


# Global singleton rewriter
query_rewriter = QueryRewriter()
