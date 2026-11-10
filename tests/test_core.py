import pytest

from slop.analyzer import _parse
from slop.models import verdict_for
from slop.utils import extract_json


def test_verdict_bands():
    assert verdict_for(0) == "Likely Real"
    assert verdict_for(34) == "Likely Real"
    assert verdict_for(35) == "Suspicious"
    assert verdict_for(64) == "Suspicious"
    assert verdict_for(65) == "Likely AI Slop"


def test_extract_json_handles_fences_and_think():
    assert extract_json('<think>hmm {x}</think>```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_clamps_and_overrides_verdict():
    raw = ('{"slop_score": 140, "verdict": "Likely Real", "confidence": "80", "summary": "s", '
           '"evidence": [{"signal": "x", "source": "transcript", "detail": "d", "points_to": "slop", "strength": "strong"}]}')
    a = _parse(raw)
    assert a.slop_score == 100 and a.verdict == "Likely AI Slop" and a.confidence == 80


def test_parse_rejects_empty_evidence():
    with pytest.raises(Exception):
        _parse('{"slop_score": 50, "confidence": 50, "summary": "s", "evidence": []}')
