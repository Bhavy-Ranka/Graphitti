from collections import defaultdict

FUNCTIONAL_PREDICATES = {
    "ceo", "founder", "founded_by", "headquartered_in", "based_in",
    "born_in", "died_in", "capital_of", "author", "director",
}


def find_conflicts(triples: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for t in triples:
        pred = t["predicate"]
        if pred not in FUNCTIONAL_PREDICATES:
            continue
        groups[(t["subject"], pred)].append(t)

    conflicts = []
    for (subject, predicate), claims in groups.items():
        distinct_objects = {c["object"] for c in claims}
        if len(distinct_objects) > 1:
            conflicts.append({"subject": subject, "predicate": predicate, "claims": claims})
    return conflicts


def mark_contested(triples: list[dict]) -> tuple[list[dict], list[dict]]:
    conflicts = find_conflicts(triples)
    contested_keys = {(c["subject"], c["predicate"]) for c in conflicts}

    for t in triples:
        key = (t["subject"], t["predicate"])
        if key in contested_keys:
            group = next(c for c in conflicts if (c["subject"], c["predicate"]) == key)
            t["contested"] = True
            t["conflicting_with"] = [
                {"object": c["object"], "source_url": c["source_url"]}
                for c in group["claims"] if c["object"] != t["object"]
            ]
        else:
            t["contested"] = False
            t["conflicting_with"] = []

    return triples, conflicts
