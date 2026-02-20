"""
Image Attachment Scorer

A Braintrust scoring function that reads an image attachment from the root
span's metadata and uses OpenAI's vision model to analyze it.

Push to Braintrust for online scoring:
    braintrust push scorer.py
"""

import base64
from enum import Enum

import braintrust
from braintrust import ReadonlyAttachment
from openai import OpenAI
from pydantic import BaseModel

SHAPE_SCORES = {
    "circle": 0.0,
    "square": 0.5,
    "rubber_ducky": 1.0,
}


class Shape(str, Enum):
    circle = "circle"
    square = "square"
    rubber_ducky = "rubber_ducky"


class ImageClassification(BaseModel):
    shape: Shape


class ScorerParameters(BaseModel):
    metadata: dict


project = braintrust.projects.create(name="LangGraph-OTEL-Example")


def image_attachment_scorer(metadata):
    """Score an image attachment found in span metadata using OpenAI vision."""
    if not metadata or "test_image" not in metadata:
        return None

    # metadata["test_image"] is a raw attachment reference dict, e.g.
    # {"type": "braintrust_attachment", "key": "...", "filename": "...", "content_type": "..."}
    raw_ref = metadata["test_image"]
    attachment = ReadonlyAttachment(raw_ref)

    # Read the image bytes and encode as a base64 data URL for OpenAI vision.
    image_bytes = attachment.data
    content_type = raw_ref.get("content_type", "image/png")
    data_url = f"data:{content_type};base64,{base64.b64encode(image_bytes).decode()}"

    client = OpenAI()
    result = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Look at this image and classify it as one of: "
                            "circle, square, or rubber_ducky."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        },
                    },
                ],
            }
        ],
        response_format=ImageClassification,
        max_tokens=50,
    )

    classification = result.choices[0].message.parsed

    return {
        "score": SHAPE_SCORES[classification.shape.value],
        "metadata": {
            "classification": classification.shape.value,
        },
    }


project.scorers.create(
    name="Image Attachment Quality",
    slug="image-attachment-quality",
    description="Uses OpenAI GPT-4o vision to analyze an image attachment and rate its quality.",
    parameters=ScorerParameters,
    handler=image_attachment_scorer,
)
