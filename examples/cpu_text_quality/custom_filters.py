"""DataTrove adapters for shared CPU text-quality filters."""

import os
import sys
from pathlib import Path

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


SHARED_DIR = Path(
    os.environ.get("CPU_TEXT_QUALITY_SHARED", Path.cwd().parent / "cpu_text_quality_shared")
).resolve()
sys.path.insert(0, str(SHARED_DIR))

from text_quality_core import (  # noqa: E402
    FastTextQualityScorer,
    FastTextLanguageScorer,
    KenLMPerplexityScorer,
    HFTokenCounter,
    XGBoostQualityScorer,
    contains_decoded_base64,
    has_real_text,
    sentence_metrics,
)


class RealTextFilter(BaseFilter):
    """Reject null, invisible, empty, and serialized-null text."""

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        return True if has_real_text(doc.text) else (False, "invalid_text")


class DecodedBase64Filter(BaseFilter):
    """Reject text containing a long, strictly decodable Base64 block."""

    def __init__(
        self,
        min_encoded_chars: int = 80,
        min_decoded_bytes: int = 32,
        exclusion_writer: DiskWriter | None = None,
    ):
        super().__init__(exclusion_writer)
        self.min_encoded_chars = min_encoded_chars
        self.min_decoded_bytes = min_decoded_bytes

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        invalid = contains_decoded_base64(doc.text, self.min_encoded_chars, self.min_decoded_bytes)
        return (False, "decoded_base64") if invalid else True


class MaxSentenceLengthFilter(BaseFilter):
    """Reject documents whose longest punctuation-delimited sentence is too long."""

    def __init__(self, max_len: int = 300, exclusion_writer: DiskWriter | None = None):
        super().__init__(exclusion_writer)
        self.max_len = max_len

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        value = sentence_metrics(doc.text)["max_sentence_length"]
        doc.metadata["max_sentence_length"] = value
        return True if value <= self.max_len else (False, "max_sentence_length")


class SentenceRepeatRatioFilter(BaseFilter):
    """Reject documents with too many repeated punctuation-delimited sentences."""

    def __init__(self, max_ratio: float = 0.3, exclusion_writer: DiskWriter | None = None):
        super().__init__(exclusion_writer)
        self.max_ratio = max_ratio

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        value = sentence_metrics(doc.text)["sentence_repeat_ratio"]
        doc.metadata["sentence_repeat_ratio"] = value
        return True if value <= self.max_ratio else (False, "sentence_repeat_ratio")


class FastTextQualityThresholdFilter(BaseFilter):
    """Apply a fixed threshold to a configured fastText label score."""

    # The benchmark supplies the compatible `fasttext-wheel` through PYTHONPATH.
    _requires_dependencies = []

    def __init__(
        self,
        model_path: str,
        label: str,
        min_score: float = 0.5,
        exclusion_writer: DiskWriter | None = None,
    ):
        super().__init__(exclusion_writer)
        self.scorer = FastTextQualityScorer(model_path, label)
        self.min_score = min_score

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        score = self.scorer.score(doc.text)
        doc.metadata["fasttext_quality_score"] = score
        return True if score >= self.min_score else (False, "fasttext_quality")


class FastTextLanguageFilter(BaseFilter):
    def __init__(self, model_path, languages=("en", "zh"), min_score=0.65, exclusion_writer=None):
        super().__init__(exclusion_writer)
        self.scorer = FastTextLanguageScorer(model_path)
        self.languages = set(languages)
        self.min_score = min_score

    def filter(self, doc):
        language, score = self.scorer.predict(doc.text)
        doc.metadata.update(language=language, language_score=score)
        return True if language in self.languages and score >= self.min_score else (False, "language")


class KenLMPerplexityFilter(BaseFilter):
    def __init__(self, model_path, sentencepiece_path, max_ppl=float("inf"), exclusion_writer=None):
        super().__init__(exclusion_writer)
        self.scorer = KenLMPerplexityScorer(model_path, sentencepiece_path)
        self.max_ppl = max_ppl
    def filter(self, doc):
        score = self.scorer.score(doc.text); doc.metadata["kenlm_perplexity"] = score
        return True if score <= self.max_ppl else (False, "perplexity")

class HFTokenCountFilter(BaseFilter):
    def __init__(self, tokenizer_path, min_tokens=3, exclusion_writer=None):
        super().__init__(exclusion_writer); self.counter=HFTokenCounter(tokenizer_path); self.min_tokens=min_tokens
    def filter(self, doc):
        n=self.counter.count(doc.text); doc.metadata['token_count']=n
        return True if n >= self.min_tokens else (False, 'token_count')


class XGBoostQualityFilter(BaseFilter):
    """Apply a compatible XGBoost booster to the shared lexical features."""

    _requires_dependencies = ["xgboost"]

    def __init__(
        self,
        model_path: str,
        min_score: float = 0.5,
        exclusion_writer: DiskWriter | None = None,
    ):
        super().__init__(exclusion_writer)
        self.scorer = XGBoostQualityScorer(model_path)
        self.min_score = min_score

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        score = self.scorer.score(doc.text)
        doc.metadata["xgboost_quality_score"] = score
        return True if score >= self.min_score else (False, "xgboost_quality")
