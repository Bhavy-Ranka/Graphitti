import logging
import re
import nltk
from nltk import RegexpParser, pos_tag, sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tree import Tree

log = logging.getLogger("nlp_pipeline")

_REQUIRED_NLTK_RESOURCES = [
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("tokenizers/punkt", "punkt"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("chunkers/maxent_ne_chunker_tab", "maxent_ne_chunker_tab"),
    ("chunkers/maxent_ne_chunker", "maxent_ne_chunker"),
    ("corpora/words", "words"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]

_bootstrapped = False


def ensure_nltk_data() -> None:
    """Download required NLTK corpora/models if not already present. Idempotent."""
    global _bootstrapped
    if _bootstrapped:
        return
    for find_path, pkg in _REQUIRED_NLTK_RESOURCES:
        try:
            nltk.data.find(find_path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception as e:  # pragma: no cover - network/env dependent
                log.warning(f"Could not fetch NLTK resource '{pkg}': {e}")
    _bootstrapped = True


_lemmatizer = WordNetLemmatizer()
_GRAMMAR = r"""
  NP: {<NNP><CD><,>?<CD>}
  NP: {<DT|PRP\$>?<JJ.*>*<NN.*|NNP.*|PRP>+<CD>?(<,><NNP.*|CD>+)*(<CC><DT>?<JJ.*>*<NN.*|NNP.*>+)?}
  NP: {<CD>+}
  PP: {<IN|TO><NP>}
  VP: {<MD>?<RB>*<VB.*>+<RP>?}
"""

_BE_FORMS = {"is", "are", "was", "were", "be", "been", "being", "'s", "'re"}

_HEDGE_WORDS = re.compile(
    r"\b(may|might|could|allegedly|reportedly|reputedly|apparently|possibly|"
    r"rumor(?:ed)?|claims?|claimed|suggests?|suggested|appears? to|seem(?:s|ed)? to|"
    r"believed to|thought to|according to)\b",
    re.I,
)

_NEGATION = re.compile(r"\b(not|n't|never|no longer)\b", re.I)

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|"
    "october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_DATE_RE = re.compile(
    rf"^(?:\d{{1,2}}\s+)?(?:{_MONTHS})\.?(?:\s+\d{{1,2}},?)?\s*,?\s*\d{{2,4}}$"
    rf"|^\d{{4}}$"
    rf"|^(?:{_MONTHS})\.?\s+\d{{4}}$",
    re.I,
)

_ORG_SUFFIXES = re.compile(
    r"\b(inc\.?|corp\.?|corporation|company|co\.?|ltd\.?|llc|university|"
    r"institute|foundation|agency|department|ministry|association|group|"
    r"holdings?|labs?|laboratory|university)\b",
    re.I,
)

_KNOWN_ORGS = {
    "google", "apple", "microsoft", "amazon", "meta", "facebook", "netflix",
    "spacex", "tesla", "openai", "anthropic", "ibm", "intel", "nvidia",
    "samsung", "sony", "oracle", "twitter", "youtube", "nasa", "nato",
    "unesco", "unicef",
}

_ROLE_WORDS = {
    "ceo": "ceo", "president": "president", "founder": "founded_by",
    "director": "director", "chairman": "chairman", "author": "author",
    "chairwoman": "chairman", "chief executive": "ceo", "coach": "coach",
    "manager": "manager", "editor": "editor", "producer": "producer",
}

_NE_LABEL_MAP = {
    "PERSON": "Person",
    "ORGANIZATION": "Organization",
    "GPE": "Location",
    "GSP": "Location",
    "LOCATION": "Location",
    "FACILITY": "Location",
}

_PRONOUNS = {
    "he", "him", "his", "she", "her", "hers", "it", "its",
    "they", "them", "their", "theirs",
}
_PERSON_PRONOUNS = {"he", "him", "his", "she", "her", "hers"}
_PLURAL_PRONOUNS = {"they", "them", "their", "theirs"}

def _ne_spans(tags: list[tuple[str, str]]) -> dict[tuple[int, int], str]:
    tree = nltk.ne_chunk(tags)
    spans: dict[tuple[int, int], str] = {}
    idx = 0
    prev_label = None
    prev_start = None
    for node in tree:
        if isinstance(node, Tree):
            label = node.label()
            n = len(node.leaves())
            if label == prev_label and prev_start is not None:
                spans[(prev_start, idx + n - 1)] = label
                spans.pop((prev_start, idx - 1), None)
            else:
                spans[(idx, idx + n - 1)] = label
                prev_start = idx
            prev_label = label
            idx += n
        else:
            prev_label = None
            idx += 1
    return spans

def _flatten_chunks(chunk_tree: Tree) -> list[dict]:
    chunks = []
    idx = 0
    for node in chunk_tree:
        if isinstance(node, Tree):
            words = [w for w, t in node.leaves()]
            n = len(words)
            chunks.append({
                "label": node.label(),
                "tokens": words,
                "text": " ".join(words),
                "start": idx,
                "end": idx + n - 1,
            })
            idx += n
        else:
            idx += 1
    return chunks


def _clean_np_text(text: str) -> str:
    text = re.sub(r"\s+,\s*", ", ", text)
    text = re.sub(r"\s+'s\b", "'s", text)
    text = text.strip(" ,")
    return text


def _guess_type(np_chunk: dict, ne_spans: dict[tuple[int, int], str]) -> str:
    if np_chunk["text"].strip().lower() in _KNOWN_ORGS:
        return "Organization"

    start, end = np_chunk["start"], np_chunk["end"]
    label_overlap: dict[str, int] = {}
    for (s, e), label in ne_spans.items():
        overlap = max(0, min(end, e) - max(start, s) + 1)
        if overlap > 0:
            label_overlap[label] = label_overlap.get(label, 0) + overlap
    if label_overlap:
        best_label = max(label_overlap, key=label_overlap.get)
        return _NE_LABEL_MAP.get(best_label, "Other")

    text = np_chunk["text"]
    if _DATE_RE.match(text.strip()):
        return "Date"
    if _ORG_SUFFIXES.search(text):
        return "Organization"
    words = [w for w in np_chunk["tokens"] if w not in (",",)]
    if words and all(w[0].isupper() for w in words if w.isalpha()):
        return "Other"
    return "Concept"


def _split_conjunction(np_chunk: dict) -> list[dict]:
    tokens = np_chunk["tokens"]
    if "and" not in tokens and "&" not in tokens:
        return [np_chunk]
    parts, current = [], []
    for tok in tokens:
        if tok.lower() in ("and", "&") and current:
            parts.append(current)
            current = []
        else:
            current.append(tok)
    if current:
        parts.append(current)
    if len(parts) < 2:
        return [np_chunk]
    out = []
    for part in parts:
        out.append({**np_chunk, "tokens": part, "text": " ".join(part)})
    return out


_IRREGULAR_PRESENT = {
    "has": "have", "does": "do", "goes": "go", "is": "is", "'s": "is",
}


def _normalize_verb_surface(word: str, tag: str) -> str:
    lower = word.lower()
    if tag == "VBZ":
        if lower in _IRREGULAR_PRESENT:
            return _IRREGULAR_PRESENT[lower]
        if lower.endswith("ies"):
            return lower[:-3] + "y"
        if lower.endswith(("sses", "shes", "ches", "xes")):
            return lower[:-2]
        if lower.endswith("s") and not lower.endswith("ss"):
            return lower[:-1]
    return lower


def _predicate_from_vp(vp_tokens: list[str], vp_tags: list[str], is_passive: bool) -> tuple[str, bool]:
    negated = bool(_NEGATION.search(" ".join(vp_tokens)))
    pairs = [(t, g) for t, g in zip(vp_tokens, vp_tags)
             if t.lower() not in _BE_FORMS and t.lower() != "not" and not t.lower().endswith("n't")]
    if not pairs:
        return "is", negated
    head_word, head_tag = pairs[-1]
    base = _normalize_verb_surface(head_word, head_tag)
    return base, negated


def _is_passive(vp_tokens: list[str], vp_tags: list[str]) -> bool:
    has_be = any(t.lower() in _BE_FORMS for t in vp_tokens)
    has_vbn = "VBN" in vp_tags
    return has_be and has_vbn


def _confidence(base: float, sentence: str, negated: bool) -> float:
    score = base
    if _HEDGE_WORDS.search(sentence):
        score -= 0.25
    if negated:
        score -= 0.1
    return max(0.05, min(0.95, round(score, 2)))

class _EntityMemory:
    def __init__(self):
        self.last_person = None
        self.last_plural_or_org = None
        self.last_any = None

    def update(self, text: str, etype: str):
        self.last_any = text
        if etype == "Person":
            self.last_person = text
        if etype in ("Organization", "Location", "Product", "Technology", "Concept"):
            self.last_plural_or_org = text

    def resolve(self, pronoun: str) -> str | None:
        p = pronoun.lower()
        if p in _PERSON_PRONOUNS:
            return self.last_person
        if p in _PLURAL_PRONOUNS:
            return self.last_plural_or_org or self.last_any
        return self.last_plural_or_org or self.last_any


def _resolve_np_text(np_chunk: dict, etype: str, memory: "_EntityMemory") -> tuple[str, bool]:
    """Returns (resolved_text, was_pronoun)."""
    text = np_chunk["text"].strip()
    if text.lower() in _PRONOUNS:
        resolved = memory.resolve(text.lower())
        if resolved:
            return resolved, True
        return text, True
    return _clean_np_text(text), False

_chunker = RegexpParser(_GRAMMAR)

def _extract_from_sentence(sentence: str, memory: "_EntityMemory") -> list[dict]:
    tokens = word_tokenize(sentence)
    if len(tokens) < 3:
        return []
    tags = pos_tag(tokens)
    ne_spans = _ne_spans(tags)
    tree = _chunker.parse(tags)
    chunks = _flatten_chunks(tree)

    for c in chunks:
        if c["label"] == "NP":
            c["etype"] = _guess_type(c, ne_spans)

    triples: list[dict] = []
    i = 0
    n = len(chunks)

    def emit(subj_text, subj_type, pred, obj_text, obj_type, conf_base, negated, pronoun_used):
        if not subj_text or not obj_text:
            return
        if subj_text.strip().lower() == obj_text.strip().lower():
            return
        pred_final = ("not_" + pred) if negated else pred
        conf = _confidence(conf_base, sentence, negated)
        if pronoun_used:
            conf = max(0.05, conf - 0.1)
        triples.append({
            "subject": subj_text.strip(),
            "subject_type": subj_type,
            "predicate": pred_final.lower().replace(" ", "_"),
            "object": obj_text.strip(),
            "object_type": obj_type,
            "confidence": conf,
        })

    while i < n:
        chunk = chunks[i]
        if chunk["label"] != "NP":
            i += 1
            continue

        subj_np = chunk
        j = i + 1
        if j >= n or chunks[j]["label"] != "VP":
            i += 1
            continue

        vp = chunks[j]
        vp_tags = pos_tag(vp["tokens"])
        vp_tag_list = [t for _, t in vp_tags]
        passive = _is_passive(vp["tokens"], vp_tag_list)
        pred_base, negated = _predicate_from_vp(vp["tokens"], vp_tag_list, passive)

        k = j + 1
        subj_variants = _split_conjunction(subj_np)
        subj_resolved = [_resolve_np_text(s, subj_np.get("etype", "Other"), memory) for s in subj_variants]

        for s, (text, was_pron) in zip(subj_variants, subj_resolved):
            if not was_pron:
                memory.update(text, subj_np.get("etype", "Other"))

        handled = False

        if k < n and chunks[k]["label"] == "PP" and passive:
            pp = chunks[k]
            prep = pp["tokens"][0].lower()
            if prep == "by":
                agent_np = {**pp, "tokens": pp["tokens"][1:], "text": " ".join(pp["tokens"][1:])}
                agent_variants = _split_conjunction(agent_np)
                for a in agent_variants:
                    a_type = _guess_type(a, ne_spans) if a["tokens"] else "Other"
                    a_text, a_pron = _resolve_np_text(a, a_type, memory)
                    for s_text, (subj_text, subj_pron) in zip(subj_variants, subj_resolved):
                        emit(a_text, a_type, pred_base, subj_text,
                             subj_np.get("etype", "Other"), 0.7, negated, a_pron or subj_pron)
                        memory.update(a_text, a_type)
                handled = True
                k += 1

        if not handled and k < n and chunks[k]["label"] == "NP":
            obj_np = chunks[k]
            obj_variants = _split_conjunction(obj_np)

            role_key = obj_np["text"].lower()
            role_head = next((w for r, w in _ROLE_WORDS.items() if r in role_key), None)
            if pred_base == "is" and role_head and k + 1 < n and chunks[k + 1]["label"] == "PP" \
                    and chunks[k + 1]["tokens"][0].lower() in ("of", "at"):
                pp = chunks[k + 1]
                org_np = {**pp, "tokens": pp["tokens"][1:], "text": " ".join(pp["tokens"][1:])}
                org_type = _guess_type(org_np, ne_spans) if org_np["tokens"] else "Other"
                org_text, org_pron = _resolve_np_text(org_np, org_type, memory)
                for s_text, (subj_text, subj_pron) in zip(subj_variants, subj_resolved):
                    emit(org_text, org_type, role_head, subj_text,
                         subj_np.get("etype", "Other"), 0.75, negated, org_pron or subj_pron)
                handled = True
                k += 2
            else:
                for s_text, (subj_text, subj_pron) in zip(subj_variants, subj_resolved):
                    for o in obj_variants:
                        o_text, o_pron = _resolve_np_text(o, o.get("etype", _guess_type(o, ne_spans)), memory)
                        o_type = o.get("etype", _guess_type(o, ne_spans))
                        emit(subj_text, subj_np.get("etype", "Other"), pred_base, o_text,
                             o_type, 0.8, negated, subj_pron or o_pron)
                        if not o_pron:
                            memory.update(o_text, o_type)
                handled = True
                k += 1

            if k < n and chunks[k]["label"] == "PP":
                pp = chunks[k]
                prep = pp["tokens"][0].lower()
                pp_np = {**pp, "tokens": pp["tokens"][1:], "text": " ".join(pp["tokens"][1:])}
                if pp_np["tokens"]:
                    pp_type = _guess_type(pp_np, ne_spans)
                    pp_text, pp_pron = _resolve_np_text(pp_np, pp_type, memory)
                    for o in obj_variants:
                        o_text = _clean_np_text(o["text"])
                        emit(o_text, o.get("etype", "Other"), f"{pred_base}_{prep}", pp_text,
                             pp_type, 0.55, negated, pp_pron)
                k += 1

        elif not handled and k < n and chunks[k]["label"] == "PP":
            while k < n and chunks[k]["label"] == "PP":
                pp = chunks[k]
                prep = pp["tokens"][0].lower()
                pp_np = {**pp, "tokens": pp["tokens"][1:], "text": " ".join(pp["tokens"][1:])}
                if not pp_np["tokens"]:
                    k += 1
                    continue
                pp_type = _guess_type(pp_np, ne_spans)
                pp_text, pp_pron = _resolve_np_text(pp_np, pp_type, memory)
                for s_text, (subj_text, subj_pron) in zip(subj_variants, subj_resolved):
                    emit(subj_text, subj_np.get("etype", "Other"), f"{pred_base}_{prep}", pp_text,
                         pp_type, 0.65, negated, subj_pron or pp_pron)
                k += 1
            handled = True

        i = k if k > i else i + 1

    return triples


def normalize_entity_types(triples: list[dict]) -> list[dict]:
    """Per-sentence NER can flip-flop on the same surface form (e.g. 'Apple'
    tagged GPE in one sentence, unrecognized in another). Reconcile by
    majority vote across all mentions of the same entity string within the
    batch, so a single page ends up with one consistent type per entity."""
    from collections import Counter

    votes: dict[str, Counter] = {}
    for t in triples:
        for role in ("subject", "object"):
            name = t[role]
            etype = t.get(f"{role}_type", "Other")
            votes.setdefault(name, Counter())[etype] += 1

    majority = {name: counter.most_common(1)[0][0] for name, counter in votes.items()}

    for t in triples:
        t["subject_type"] = majority.get(t["subject"], t.get("subject_type", "Other"))
        t["object_type"] = majority.get(t["object"], t.get("object_type", "Other"))
    return triples

def extract_triples_from_chunk(chunk_text: str) -> list[dict]:
    """Extract raw triples (no source metadata) from a single text chunk."""
    ensure_nltk_data()
    memory = _EntityMemory()
    triples: list[dict] = []
    for sentence in sent_tokenize(chunk_text):
        sentence = sentence.strip()
        if not sentence:
            continue
        try:
            triples.extend(_extract_from_sentence(sentence, memory))
        except Exception as e:  # keep pipeline resilient sentence-by-sentence
            log.debug(f"Sentence extraction failed, skipping: {e!r}")
    return triples
