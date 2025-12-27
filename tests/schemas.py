"""JSON Schema definitions for i2ss CLI artifacts."""

from __future__ import annotations

SEGMENT_OBJECTS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "image": {"type": "string"},
        "objects": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
        },
        "object_count": {"type": "integer", "minimum": 0},
    },
    "required": ["image", "objects", "object_count"],
}

TRACKS_PROMPTS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "prompts": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "seconds": {"type": "integer", "minimum": 1},
        "seed": {"type": "integer"},
    },
    "required": ["prompts", "seconds", "seed"],
}

MIX_METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "string"},
        "note": {"type": "string"},
    },
    "required": ["source", "note"],
    "additionalProperties": True,
}
