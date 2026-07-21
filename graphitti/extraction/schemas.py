from typing import Literal

from pydantic import BaseModel, Field

EntityType = Literal[
    "Person", "Organization", "Product", "Concept",
    "Location", "Event", "Technology", "Date", "Other",
]


class Triple(BaseModel):
    subject: str = Field(description="The subject entity, as written in the text.")
    subject_type: EntityType = Field(default="Other")
    predicate: str = Field(
        description="Short, normalized verb phrase, e.g. 'founded', 'based_in', 'released'."
    )
    object: str = Field(description="The object entity or literal, as written in the text.")
    object_type: EntityType = Field(default="Other")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="How explicit/certain the claim is in the source text.",
    )


class TripleList(BaseModel):
    triples: list[Triple] = Field(default_factory=list)
