import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Words that, when immediately before an em-dash, mean the main clause CONTINUES
# after the parenthetical (e.g. "nothing — A, B — can escape").
# For any other last word the em-dash and everything after it is stripped.
_DANGLERS = frozenset({
    "nothing", "anything", "everything", "something",
    "that", "which", "who", "whom", "whose",
    "where", "when", "why", "how",
    "because", "if", "than", "as", "so",
})


def _compress(sentence, max_words=30):
    original = sentence  # kept for fallback if all steps over-compress

    # 1. Em-dash: requires >=5 words before the dash to avoid stubs like "Lucid dreaming."
    #    If the word before the dash is a dangler (predicate follows after the aside),
    #    remove the aside and rejoin: "X -- list -- verb" -> "X verb".
    #    Otherwise strip everything from the dash onward: "X -- elaboration" -> "X".
    for sep in [" — ", " - "]:
        idx = sentence.find(sep)
        if 0 < idx < len(sentence) - 1:
            pre = sentence[:idx].rstrip()
            if len(pre.split()) >= 5:
                last_word = pre.split()[-1].lower().rstrip(".,;:")
                idx2 = sentence.find(sep, idx + len(sep))
                if idx2 > idx and last_word in _DANGLERS:
                    sentence = pre + " " + sentence[idx2 + len(sep):]
                else:
                    sentence = pre
            break

    # 2. Strip at semicolon
    idx = sentence.find("; ")
    if idx > len(sentence) // 3:
        sentence = sentence[:idx]

    # 3. Strip relative / subordinate clauses.
    #    Uses 1/4 threshold (not 1/3) to catch mid-sentence clauses in long sentences.
    for pat in [r",\s+which\b", r",\s+who\b", r",\s+whose\b", r",\s+where\b",
                r",\s+because\b", r",\s+although\b", r",\s+while\b"]:
        m = re.search(pat, sentence)
        if m and m.start() > len(sentence) // 4:
            sentence = sentence[:m.start()]
            break

    # 4. Strip late appositive phrases
    m = re.search(r",\s+a\s+(?:phenomenon|process|testament|distinction|capacity)\b",
                  sentence)
    if m and m.start() > len(sentence) // 4:
        sentence = sentence[:m.start()]

    # 5. Hard word-count fallback: find the first major comma AFTER the halfway
    #    point to avoid cutting inside compound modifiers ("superheated, mineral-rich").
    #    Falls back to the last comma if none exists after halfway.
    words = sentence.split()
    if len(words) > max_words:
        truncated = " ".join(words[:max_words])
        half = len(truncated) // 2

        def _find_prose_comma(s, start):
            """Return index of first comma that isn't inside a number (e.g. 100,000)."""
            i = s.find(",", start)
            while i > 0:
                if not (i > 0 and i < len(s) - 1 and s[i-1].isdigit() and s[i+1].isdigit()):
                    return i
                i = s.find(",", i + 1)
            return -1

        cut = _find_prose_comma(truncated, half)
        if cut <= 0:
            cut = _find_prose_comma(truncated, 0)
            if cut <= 0:
                cut = truncated.rfind(",")
        sentence = truncated[:cut] if cut > 0 else truncated

    # Fallback: if all steps above reduced the sentence to fewer than 4 words
    # (e.g. NLTK produced a short fragment like "Writing,"), discard the
    # over-compressed result and use a plain word-count truncation of the original.
    if len(sentence.split()) < 4:
        words = original.split()
        sentence = " ".join(words[:max_words])

    sentence = sentence.rstrip(" ,;:—")
    if not sentence.endswith((".", "!", "?")):
        sentence += "."
    return sentence


def extract_key_sentences(text, n_select=None, max_words=30, min_words=None):
    """Select representative sentences via TF-IDF + cosine similarity.

    Two modes:
    - Word-budget (concise, min_words set): add the highest-scoring sentences,
      most important first, until the compressed summary reaches at least
      min_words (~10% of the input). The sentence count adapts to the text.
    - Segment (detailed, n_select set): split the text into n_select regions and
      take the best-scoring sentence from each, guaranteeing coverage across the
      whole document and avoiding near-duplicate picks.

    Selected sentences are always emitted in document order for readability.
    """
    sentences = nltk.sent_tokenize(text)
    n_total = len(sentences)

    if n_total <= 1:
        return " ".join(_compress(s, max_words) for s in sentences)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)

    doc_vector = np.asarray(tfidf_matrix.mean(axis=0))
    scores = cosine_similarity(tfidf_matrix, doc_vector).flatten()

    for idx in range(n_total // 2, n_total):
        scores[idx] *= 1.20

    if min_words:
        # Word-budget mode: add sentences highest-score-first until the compressed
        # summary crosses the min_words floor, then stop. Count adapts to content.
        picked = {}
        total = 0
        for idx in sorted(range(n_total), key=lambda i: scores[i], reverse=True):
            picked[idx] = _compress(sentences[idx], max_words)
            total += len(picked[idx].split())
            if total >= min_words:
                break
        return " ".join(picked[i] for i in sorted(picked))

    # Segment mode.
    if n_total <= n_select:
        return " ".join(_compress(s, max_words) for s in sentences)

    segment_size = n_total / n_select
    selected = []
    for i in range(n_select):
        start = int(i * segment_size)
        end = int((i + 1) * segment_size)
        segment = list(range(start, min(end, n_total)))
        if segment:
            best = max(segment, key=lambda idx: scores[idx])
            selected.append(best)

    selected.sort()
    return " ".join(_compress(sentences[i], max_words) for i in selected)
