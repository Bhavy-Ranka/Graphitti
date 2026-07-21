from collections import Counter


def _tokens(name: str) -> set[str]:
    return set(name.lower().replace(".", "").split())


def build_alias_map(triples: list[dict]) -> dict[str, str]:
    entity_type: dict[str, str] = {}
    counts: Counter = Counter()

    for t in triples:
        for role in ("subject", "object"):
            name = t[role]
            entity_type.setdefault(name, t.get(f"{role}_type", "Other"))
            counts[name] += 1

    by_type: dict[str, list[str]] = {}
    for name, etype in entity_type.items():
        by_type.setdefault(etype, []).append(name)

    alias_map: dict[str, str] = {}
    for _etype, names in by_type.items():
        for short in sorted(names, key=len):
            short_tokens = _tokens(short)
            if not short_tokens:
                continue
            best_match, best_count = None, -1
            for long in names:
                if long == short:
                    continue
                long_tokens = _tokens(long)
                if short_tokens < long_tokens and counts[long] > best_count:
                    best_match, best_count = long, counts[long]
            if best_match:
                alias_map[short] = best_match

    return alias_map


def resolve_entity_aliases(triples: list[dict]) -> list[dict]:
    alias_map = build_alias_map(triples)
    if not alias_map:
        return triples
    for t in triples:
        t["subject"] = alias_map.get(t["subject"], t["subject"])
        t["object"] = alias_map.get(t["object"], t["object"])
    return triples
