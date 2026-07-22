from typing import Literal, Optional

from pydantic import BaseModel, Field

Intent = Literal[
    "single_fact_lookup", "entity_centric", "multi_hop_relational",
    "comparison", "temporal_versioned", "broad_exploratory", "keyword_exact",
]


class IntentClassification(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="One short sentence.")


class SubQuery(BaseModel):
    step: int
    sub_query: str
    depends_on: Optional[int] = None


class DecomposedQuery(BaseModel):
    steps: list[SubQuery] = Field(default_factory=list)
