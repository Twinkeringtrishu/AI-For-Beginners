# Applied Lab R38-T035: Prompt-Template Escaping — Refactor the Teaching Example

**Lesson**: [5-NLP/20-LangModels](../../README.md)  
**Task ID**: R38-T035

---

## Overview & Lesson Context

When building applications with Large Language Models (LLMs), prompts are composed by injecting dynamic user-supplied variables into predefined instruction templates. If templating is not handled strictly:
1. Missing or unknown variables can lead to silent failures or malformed prompts.
2. User data containing special formatting symbols (such as curly braces `{}` or Markdown tokens) might be mistakenly parsed as template placeholders.
3. Templating alone does **not** prevent prompt injection (adversarial instructions embedded within user input text must still be addressed at the model/system guardrail level).

This lab implements a strict, safe template rendering engine that:
- Validates declared placeholder keys and rejects missing or unknown keys.
- Ensures braces inside user data remain strictly literal.
- Refactors a named responsibility (`extract_name`) while proving contract equivalence.

---

## Stated Contract & Simplifications

- **Template**: Double-brace placeholders `{{key}}`.
- **Declared Variables**: Canonical template with `name` and `topic` renders supplied values.
- **Missing Keys**: Omitting a required placeholder (e.g. `topic`) raises `ValueError: Missing placeholder(s)`.
- **Unknown Keys**: Supplying extra keys not in the template raises `ValueError: Unknown key(s) supplied`.
- **Literal Braces**: Braces in user data (e.g. `Bob {the builder}`) remain verbatim literal text.
- **Disclaimer**: Templating alone does not prevent prompt injection.

---

## Invocation & Verification

```bash
# Run the example
python3 lessons/5-NLP/20-LangModels/applied-labs/r38-t035/example.py

# Run unit tests
python3 -m unittest discover -s lessons/5-NLP/20-LangModels/applied-labs/r38-t035 -p "test_*.py" -v
```
