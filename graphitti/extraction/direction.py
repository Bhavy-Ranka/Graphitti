
PREDICATE_TYPE_EXPECTATIONS: dict[str, tuple[set[str], set[str]]] = {
    "born_in": ({"Person"}, {"Location"}),
    "born_on": ({"Person"}, {"Date"}),
    "died_in": ({"Person"}, {"Location"}),
    "died_on": ({"Person"}, {"Date"}),
    "plays_for": ({"Person"}, {"Organization", "Location"}),
    "founded_by": ({"Organization"}, {"Person"}),
    "founded": ({"Person"}, {"Organization"}),
    "ceo": ({"Organization"}, {"Person"}),
    "headquartered_in": ({"Organization"}, {"Location"}),
    "based_in": ({"Organization", "Person"}, {"Location"}),
    "acquired": ({"Organization"}, {"Organization"}),
    "released": ({"Organization"}, {"Product", "Technology"}),
    "author": ({"Product", "Concept"}, {"Person"}),
    "director": ({"Product", "Concept"}, {"Person"}),
    "capital_of": ({"Location"}, {"Location"}),
}
def normalize_direction(triple: dict) -> dict:
    expectation = PREDICATE_TYPE_EXPECTATIONS.get(triple.get("predicate", ""))
    if not expectation:
        return triple

    expected_subject_types, expected_object_types = expectation
    subj_t, obj_t = triple.get("subject_type"), triple.get("object_type")

    as_is_ok = subj_t in expected_subject_types and obj_t in expected_object_types
    reversed_ok = subj_t in expected_object_types and obj_t in expected_subject_types

    if not as_is_ok and reversed_ok:
        triple["subject"], triple["object"] = triple["object"], triple["subject"]
        triple["subject_type"], triple["object_type"] = triple["object_type"], triple["subject_type"]

    return triple
