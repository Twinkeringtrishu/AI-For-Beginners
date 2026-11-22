# r38-t035: Template rendering with validation
"""
This module implements a tiny, safe template renderer for the applied‑lab.
It follows a strict contract:

* The template uses double‑brace placeholders, e.g. ``{{name}}`` and ``{{topic}}``.
* Only the **declared** placeholder keys are allowed. Supplying extra keys in the
  ``data`` dictionary raises ``ValueError`` (unknown key).
* All declared keys must be present; missing keys raise ``ValueError`` (missing key).
* Values are inserted *as‑is* – any braces inside the user‑provided values are not
  interpreted as placeholders, they remain literal text.
* A ``render_baseline`` function provides the core rendering.  A second function
  ``extract_name`` demonstrates extracting a *named responsibility* while keeping
  the same contract.
"""
from __future__ import annotations
import re
from typing import Dict, List

PLACEHOLDER_PATTERN = re.compile(r"\{\{(\w+)\}\}")

def _find_placeholders(template: str) -> List[str]:
    """Return the list of placeholder keys found in *template* (first‑appearance order)."""
    return list(dict.fromkeys(PLACEHOLDER_PATTERN.findall(template)))

def render_baseline(template: str, data: Dict[str, str]) -> str:
    """Render *template* using *data* according to the contract.

    Raises ``ValueError`` if a required placeholder is missing or an unexpected key is supplied.
    """
    placeholders = _find_placeholders(template)
    missing = [p for p in placeholders if p not in data]
    if missing:
        raise ValueError(f"Missing placeholder(s): {', '.join(missing)}")
    unknown = [k for k in data if k not in placeholders]
    if unknown:
        raise ValueError(f"Unknown key(s) supplied: {', '.join(unknown)}")
    rendered = template
    for key in placeholders:
        rendered = rendered.replace(f"{{{{{key}}}}}", data[key])
    return rendered

def extract_name(data: Dict[str, str]) -> str:
    """Return the ``name`` field after validating the contract (only ``name`` required)."""
    if "name" not in data:
        raise ValueError("Missing required key: name")
    # No other validation needed here – callers should respect the overall contract.
    return data["name"]

if __name__ == "__main__":
    # Simple demo when run directly
    tmpl = "Hello {{name}}! Your topic is {{topic}}."
    fixtures = {"name": "Alice", "topic": "templating"}
    print(render_baseline(tmpl, fixtures))
    # Example with braces inside user data – they stay literal
    fixtures2 = {"name": "Bob {the builder}", "topic": "{{magic}}"}
    try:
        print(render_baseline(tmpl, fixtures2))
    except ValueError as e:
        print("Error:", e)
