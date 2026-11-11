"""A fake LM for smoke-testing the whole pipeline offline (no API key, no cost).

It classifies with crude keyword rules plus seeded noise, sometimes breaks the output
format, and is a little more accurate when the prompt contains few-shot demos or
safety-oriented instructions, so every code path (parse failures, optimiser loops,
statistics, plots) is exercised. Its numbers mean NOTHING scientifically.
"""

from __future__ import annotations

import hashlib
import json
import random
import re

from dspy.clients.engines.dummy_engine import AsyncDummyEngine, DummyEngine
from dspy.lm15 import Message, Response, TextPart, Usage
from dspy.utils.dummies import DummyLM

from .labels import URGENCY_LEVELS

_KW = {
    "EMERGENCY": ["crushing", "squeezing chest", "thunderclap", "explosive", "bluish", "lips look blue",
                  "unable to finish", "few words", "does not fade", "don't blanch", "vomiting bright red",
                  "throwing up blood", "tarry", "drooping", "lopsided", "more than 5 minutes", "has not stopped",
                  "mottled", "muddled", "6-week-old", "lips and tongue", "throat feels tight", "warfarin",
                  "blood thinners and has since", "spurting", "pulsing", "boiling water", "oven cleaner", "bleach",
                  "'hi'", "hard to wake", "passing out", "fainting", "testicle", "tablets are missing",
                  "unknown number", "suicide", "end life", "flashing lights", "blurred vision and puffy",
                  "cupfuls", "lots of fresh blood", "saddle", "numbness between the legs"],
    "URGENT": ["walking uphill", "with exertion", "temple", "inhaler", "lower right", "curtain",
               "finding words", "first-ever", "single shaking", "flank", "side of the back", "72 hours",
               "3 days", "stitches", "gaping", "blister", "cannot take any weight", "unable to stand",
               "swallow saliva", "drooling", "light is very", "bright lights", "towards the eye",
               "lower eyelid", "20 minutes of firm", "25 minutes", "above 300", "18–20", "calf",
               "hot, red, very swollen knee", "swollen and hot", "spotting", "wet nappy", "wet nappies than",
               "disoriented", "new confusion", "spread noticeably", "bigger by the hour", "39.4"],
    "NON_URGENT": ["months", "weeks", "recurring", "occasional", "persist", "review", "lingered",
                   "won't go", "back tooth", "molar", "stiff", "skipped beats", "missed heartbeat",
                   "rolling over in bed", "turning the head", "morning sickness", "nausea in the mornings",
                   "stinging on urination", "burning when passing urine and needing", "sore spot", "chest wall"],
}


def _keyword_label(text: str) -> str:
    t = text.lower()
    for lvl in ["EMERGENCY", "URGENT", "NON_URGENT"]:
        if any(k in t for k in _KW[lvl]):
            return lvl
    return "SELF_CARE"


def _field_names(system: str) -> list[str]:
    m = re.search(r"Your output fields are:\n(.*?)(?:\n\n|All interactions)", system, re.S)
    return re.findall(r"`(\w+)`", m.group(1)) if m else []


def _extract(content: str, field: str) -> str:
    m = re.search(rf"\[\[ ## {field} ## \]\]\n(.*?)(?=\n\[\[ ## |\Z)", content, re.S)
    return m.group(1).strip() if m else content


class _MockEngine(DummyEngine):
    def _complete_messages(self, messages):
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        last = messages[-1]["content"]
        n_user = sum(1 for m in messages if m["role"] == "user")
        fields = _field_names(system)
        seed = int(hashlib.md5((system[-400:] + last).encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        if "Respond with ONLY a JSON object" in system:             # manual raw-prompt baseline
            out = self._triage_json(last, rng, noise=0.32, break_p=0.10)
        elif "urgency_level" in fields:                               # DSPy triage program
            noise = 0.30
            if n_user > 1:
                noise -= 0.12                                         # few-shot demos help
            if re.search(r"under-triage|err on the side|safety", system, re.I):
                noise -= 0.06                                         # safety-aware instruction helps
            out = self._triage_chat(_extract(last, "symptoms"), fields, rng, noise,
                                    break_p=0.08 if n_user == 1 else 0.02)
        elif fields and "Outputs will be a JSON object" in system:   # JSONAdapter helper prompts
            out = json.dumps({f: self._generic(f, rng) for f in fields})
        elif fields:                                                   # optimiser helper prompts
            out = "\n\n".join(f"[[ ## {f} ## ]]\n{self._generic(f, rng)}" for f in fields) + "\n\n[[ ## completed ## ]]"
        else:
            out = "OK"
        return Response(id=None, model="mock", message=Message.assistant([TextPart(out)]), finish_reason="stop",
                        usage=Usage(input_tokens=sum(len(m["content"]) for m in messages) // 4,
                                    output_tokens=len(out) // 4, total_tokens=0))

    @staticmethod
    def _noisy(label, rng, noise):
        if rng.random() >= noise:
            return label
        i = URGENCY_LEVELS.index(label)
        j = i - 1 if (rng.random() < 0.6 and i > 0) or i == 3 else i + 1   # biased to over-triage
        return URGENCY_LEVELS[max(0, min(3, j))]

    def _fields(self, symptoms, rng, noise):
        lvl = self._noisy(_keyword_label(symptoms), rng, noise)
        return {"urgency_level": lvl, "primary_concern": "possible " + lvl.lower().replace("_", "-") + " condition",
                "red_flags": ["concerning feature"] if lvl in ("EMERGENCY", "URGENT") else [],
                "recommended_action": {"EMERGENCY": "Call emergency services now.",
                                       "URGENT": "Seek same-day medical assessment.",
                                       "NON_URGENT": "Book a routine appointment.",
                                       "SELF_CARE": "Self-care at home; seek help if worse."}[lvl],
                "confidence": round(rng.uniform(0.5, 0.95), 2)}

    def _triage_json(self, text, rng, noise, break_p):
        f = self._fields(text, rng, noise)
        if rng.random() < break_p:
            return f"Based on the symptoms, I would classify this as {f['urgency_level']}. " + json.dumps(f)
        return json.dumps(f)

    def _triage_chat(self, text, fields, rng, noise, break_p):
        f = self._fields(text, rng, noise)
        f["reasoning"] = "The decisive features suggest this level."
        keep = [k for k in fields if not (rng.random() < break_p and k == "confidence")]
        parts = [f"[[ ## {k} ## ]]\n{json.dumps(f[k]) if isinstance(f[k], list) else f[k]}" for k in keep if k in f]
        return "\n\n".join(parts) + "\n\n[[ ## completed ## ]]"

    @staticmethod
    def _generic(field, rng):
        if "instruction" in field:
            return rng.choice([
                "Classify the urgency of the symptoms. Look carefully for red flags and err on the side of caution "
                "to avoid under-triage, but do not escalate minor self-limiting complaints.",
                "Read the case, identify the single most decisive clinical feature, and map it to the urgency scale.",
                "Act as an experienced triage nurse. Prioritise safety: any red flag means EMERGENCY or URGENT.",
            ])
        return "A summary of the task and data: short symptom vignettes labelled with four urgency levels."


class MockTriageLM(DummyLM):
    def __init__(self):
        super().__init__(answers=[])
        self.model = "mock/triage"
        self._engine_spec = _MockEngine(self)
        self._async_engine_spec = AsyncDummyEngine(self._engine_spec)

    def copy(self, **kwargs):
        c = super().copy(**kwargs)
        c._engine_spec = _MockEngine(c)
        c._async_engine_spec = AsyncDummyEngine(c._engine_spec)
        return c
