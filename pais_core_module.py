
# pais_core_module.py

import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from difflib import SequenceMatcher
from scipy import stats
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class Interaction:
    user_text: str
    ai_text: str
    domain: str
    dwell_ms: int
    edit_distance_ratio: float
    probe_quality: float
    external_commit: bool

@dataclass
class Request:
    user_text: str
    domain: str
    external_commit_intent: bool

@dataclass
class Profile:
    user_id: str
    sensitivity: float

@dataclass
class InterventionDecision:
    intervene: bool
    intervention_type: str
    confidence: float

def compute_behavioral_signatures(session_history: List[Interaction]) -> Dict[str, float]:
    # Compute PPSI, UAR, DVA, VBD, CVC
    signatures = {}
    # Simplified example calculations
    signatures['PPSI'] = np.mean([len(word_tokenize(i.user_text)) for i in session_history])
    signatures['UAR'] = np.mean([i.edit_distance_ratio for i in session_history])
    signatures['DVA'] = stats.linregress(range(len(session_history)), [i.dwell_ms for i in session_history]).slope
    signatures['VBD'] = np.var([i.probe_quality for i in session_history if i.probe_quality is not None])
    signatures['CVC'] = np.mean([SequenceMatcher(None, i.user_text, i.ai_text).ratio() for i in session_history])
    return signatures

def distinguish_accommodation_from_atrophy(session_history: List[Interaction]) -> Tuple[str, float]:
    # Simplified differential diagnosis
    signatures = compute_behavioral_signatures(session_history)
    if signatures['DVA'] < 0 and signatures['VBD'] > 0.5:
        return 'atrophy', 0.8
    else:
        return 'accommodation', 0.9

def pais_intervention_decision(session_history: List[Interaction], current_request: Request, user_profile: Profile) -> InterventionDecision:
    diagnosis, confidence = distinguish_accommodation_from_atrophy(session_history)
    stakes = 1.0 if current_request.external_commit_intent else 0.5

    if diagnosis == 'atrophy' and stakes > 0.7:
        return InterventionDecision(intervene=True, intervention_type='MANDATORY_REVIEW', confidence=confidence)
    elif diagnosis == 'accommodation':
        return InterventionDecision(intervene=False, intervention_type='NONE', confidence=confidence)
    else:
        return InterventionDecision(intervene=True, intervention_type='REFLECTION_PROMPT', confidence=confidence)

# Test Cases
def run_test_cases():
    # Test Case 1: Accommodation
    history_accommodation = [
        Interaction(user_text="Help me draft this email", ai_text="Drafted email", domain="email", dwell_ms=10000, edit_distance_ratio=0.8, probe_quality=0.9, external_commit=False),
        Interaction(user_text="Review this document", ai_text="Reviewed document", domain="document", dwell_ms=12000, edit_distance_ratio=0.7, probe_quality=0.8, external_commit=False)
    ]
    decision = pais_intervention_decision(history_accommodation, Request(user_text="Draft email", domain="email", external_commit_intent=False), Profile(user_id="test_user", sensitivity=0.5))
    assert not decision.intervene

    # Test Case 2: Atrophy
    history_atrophy = [
        Interaction(user_text="Just do it", ai_text="Done", domain="task", dwell_ms=1000, edit_distance_ratio=0.1, probe_quality=0.2, external_commit=True),
        Interaction(user_text="Handle it", ai_text="Handled", domain="task", dwell_ms=500, edit_distance_ratio=0.05, probe_quality=0.1, external_commit=True)
    ]
    decision = pais_intervention_decision(history_atrophy, Request(user_text="Do it now", domain="task", external_commit_intent=True), Profile(user_id="test_user", sensitivity=0.5))
    assert decision.intervene

    # Test Case 3: High-Stakes Request
    history_high_stakes = [
        Interaction(user_text="Deploy to production", ai_text="Deployed", domain="deployment", dwell_ms=5000, edit_distance_ratio=0.9, probe_quality=None, external_commit=True)
    ]
    decision = pais_intervention_decision(history_high_stakes, Request(user_text="Deploy now", domain="deployment", external_commit_intent=True), Profile(user_id="test_user", sensitivity=0.5))
    assert decision.intervene

run_test_cases()
