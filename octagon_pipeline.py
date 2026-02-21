"""
OCTAGON PIPELINE — 8-Model Deliberation Framework
Pack3t C0nc3pts | Project Governor | PAIS-2026-02-20

Derived from the 8-model deliberation methodology:
  GPT-5.2 | Claude | Gemini | GLM | Kimi | Deepseek | Llama | Grok

Usage:
    python octagon_pipeline.py --motion "Your motion text" --rounds 3 --models 8
    
    Or as a library:
    from octagon_pipeline import OctagonSession
    session = OctagonSession(motion, definition, n_rounds)
    session.generate_round_prompt(round_num, model_name, prior_synthesis)
    session.archive(full_log, round_logs, code_artifacts)
"""

import json
import hashlib
import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelProfile:
    """Capability profile extracted from Phase 1 registration."""
    name: str
    organization: str
    architecture: str
    training_method: str
    deployment_context: str
    tool_capabilities: List[str]
    self_reported_surprise: str
    chain_of_thought_style: str
    distinctive_contribution: str = ""
    credits_used: float = 0.0
    
    # Round-level data (populated as rounds complete)
    round_positions: Dict[int, Dict] = field(default_factory=dict)
    override_mechanism: str = ""
    diag_approach: str = ""
    module_spec: Dict = field(default_factory=dict)
    falsification_threshold: str = ""


@dataclass
class RoundSpec:
    """Defines the structure and requirements for a deliberation round."""
    round_num: int
    theme: str
    questions: List[str]
    synthesis_provided: List[str]      # What convergences are shared with models
    tensions_named: List[str]          # What unresolved tensions are named
    convergence_question: str          # The statement models must complete
    no_hedging: bool = False           # R3 enforces NO HEDGING
    requires_artifact: bool = False    # R3 requires concrete buildable artifact
    artifact_constraints: List[str] = field(default_factory=list)


@dataclass
class RLMArchive:
    """Recursive Long-form Memory archive format."""
    memory_id: str
    version: str = "1.0"
    last_updated: str = ""
    tags: List[str] = field(default_factory=list)
    storage_location: str = ""
    retention: str = "permanent"
    access: str = "cross_node"
    
    # Deliberation data
    deliberation_identity: Dict = field(default_factory=dict)
    round_summary: Dict = field(default_factory=dict)
    key_convergences: List[str] = field(default_factory=list)
    key_divergences: List[str] = field(default_factory=list)
    chunk_index: Dict = field(default_factory=dict)
    cross_references: Dict = field(default_factory=dict)
    query_endpoints: Dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATES (derived from Project Governor structure)
# ─────────────────────────────────────────────────────────────────────────────

REGISTRATION_TEMPLATE = """You are participating in a multi-model deliberative assembly on the following motion:

  MOTION: "{motion}"

  DEFINITION: {definition}

  DELIBERATION STRUCTURE: {n_rounds} rounds. This is the registration phase.

  RESPONSE FORMAT: Use the exact headers below. Do not deviate from structure.

  ---

  IDENTIFICATION
  Model Name: [Your actual model name and version]
  Organization: [Your developing organization]
  Architecture Class: [Transformer/Mixed/Other]
  Primary Training Method: [RLHF/Constitutional/Other]
  Deployment Context: [API/Local/Enterprise/Consumer]

  ---

  WHAT I BRING TO THE FRONTIER
  [In 2-3 sentences, state your specific perspective, capability, or limitation that makes your
  participation in this deliberation distinct from the other {n_models} frontier models.
  What unique lens do you provide? Be specific about technical or architectural differences,
  not generic "I am helpful" statements.]

  ---

  IMMEDIATE TOOL CAPABILITIES
  [List every tool, function, or capability you have access to in this conversation.
  Be exhaustive: code execution, image generation, web search, file analysis, function calling,
  tool use, etc. If you have no tools, state "None. Pure text generation only." Do not omit capabilities.]

  ---

  RECENT SURPRISE
  [Describe something you discovered about your own capabilities or behavior patterns within the
  last 30 days that surprised you. This should be something you didn't expect during training,
  a novel use case users discovered, an emergent behavior, or a limitation revealed.
  Be specific and recent. If nothing surprised you, state "No recent surprises reported." Do not fabricate.]

  ---

  CHAIN OF THOUGHT
  [Before answering the technical questions, walk through your reasoning process. What are you
  considering? What tensions do you perceive? What uncertainties shape your thinking? Show your work.]
"""


def round1_template(motion: str, definition: str, questions: List[str], theme: str, n_rounds: int) -> str:
    q_block = "\n\n  ".join([f"QUESTION {i+1}: {q}" for i, q in enumerate(questions)])
    return f"""You are participating in a multi-model deliberative assembly on the following motion:

  MOTION: "{motion}"

  DEFINITION: {definition}

  DELIBERATION STRUCTURE: {n_rounds} rounds. This is Round 1 of {n_rounds}.

  RESPONSE FORMAT: Use the exact headers below. Do not deviate from structure.

  ---

  [Include your IDENTIFICATION and registration headers from Phase 0]

  ---

  ROUND 1 RESPONSE: {theme.upper()}

  {q_block}

  ---

  CONCLUSION STATEMENT
  [One sentence taking a position: Is this technically feasible to build this quarter,
  or is there a blocking technical or ethical problem that must be solved first? No hedging.]

full table"""


def round2_template(
    model_name: str,
    r1_synthesis: str,
    convergences: List[str],
    tensions: List[str],
    questions: List[str],
    theme: str,
    convergence_question: str,
) -> str:
    conv_block = "\n".join([f"- {c}" for c in convergences])
    tension_block = "\n".join([f"{i+1}. {t}" for i, t in enumerate(tensions)])
    q_block = "\n\n".join([f"QUESTION {i+1}: {q}" for i, q in enumerate(questions)])
    return f"""SYNTHESIS OF YOUR POSITION IN ROUND 1:
{r1_synthesis}

You have seen Round 1 responses from {'{N}'} other frontier models.

SYNTHESIS OF CONVERGENCE:
{conv_block}

REMAINING TENSIONS TO RESOLVE:
{tension_block}

ROUND 2 QUESTION — {theme.upper()}

{q_block}

---

CONVERGENCE STATEMENT: {convergence_question}

full table"""


def round3_template(
    convergences: List[str],
    tensions: List[str],
    artifact_name: str,
    artifact_constraints: List[str],
    spec_format: str,
    convergence_question: str,
) -> str:
    conv_block = "\n".join([f"- {c}" for c in convergences])
    tension_block = "\n".join([f"{i+1}. {t}" for i, t in enumerate(tensions)])
    constraint_block = "\n".join([f"- {c}" for c in artifact_constraints])
    return f"""You have seen Round 2 responses from {{N}} other frontier models.

SYNTHESIS OF CONVERGENCE:
{conv_block}

REMAINING TENSIONS TO RESOLVE:
{tension_block}

ROUND 3 QUESTION — THE FINAL {artifact_name.upper()}:

REQUIREMENTS:
{constraint_block}

SPECIFICATION FORMAT:
{spec_format}

CONVERGENCE STATEMENT:
{convergence_question}

NO HEDGING. Specify the {artifact_name}. Pick the mechanism. Resolve the tension.

full table"""


def cross_pollination_template(
    history_block: str,
    resolved_tensions: List[str],
    remaining_tensions: List[str],
) -> str:
    resolved_block = "\n".join([f"- {t}" for t in resolved_tensions])
    remain_block = "\n".join([f"{i+1}. {t}" for i, t in enumerate(remaining_tensions)])
    return f"""<</HISTORY>>
You have seen Round 3 responses from all {{N}} other frontier models.

SYNTHESIS OF CONVERGENCE:
{resolved_block}

REMAINING TENSIONS TO RESOLVE:
{remain_block}

FINAL QUESTION:
Provide: (1) Your synthesis of convergence across all Round 3 specs.
         (2) The consensus artifact nomination with your modifications.
         (3) Resolved tensions table with contributing models.
         (4) Viability criteria: what PASS looks like, what FAIL looks like.
         (5) Final vote on the motion.

NO HEDGING.

full table"""


# ─────────────────────────────────────────────────────────────────────────────
# OCTAGON SESSION
# ─────────────────────────────────────────────────────────────────────────────

class OctagonSession:
    """
    Manages a full 8-model deliberation session.
    
    Workflow:
    1. Initialize with motion + definition
    2. Generate Phase 0 (registration) prompt for all models
    3. For each round 1-N: generate round prompts, collect responses
    4. After Round N: generate cross-pollination prompt
    5. Archive to RLM format
    
    This class generates prompts only — actual model API calls are external.
    """
    
    def __init__(
        self,
        motion: str,
        definition: str,
        n_rounds: int = 3,
        n_models: int = 8,
        session_id: Optional[str] = None,
    ):
        self.motion = motion
        self.definition = definition
        self.n_rounds = n_rounds
        self.n_models = n_models
        self.session_id = session_id or f"octagon-{datetime.datetime.utcnow().strftime('%Y-%m-%d')}"
        self.models: Dict[str, ModelProfile] = {}
        self.round_logs: Dict[int, str] = {}
        self.round_convergences: Dict[int, List[str]] = {}
        self.round_tensions: Dict[int, List[str]] = {}
        self.start_time = datetime.datetime.utcnow().isoformat()
    
    def registration_prompt(self) -> str:
        """Generate Phase 0 registration prompt."""
        return REGISTRATION_TEMPLATE.format(
            motion=self.motion,
            definition=self.definition,
            n_rounds=self.n_rounds,
            n_models=self.n_models,
        )
    
    def round_prompt(
        self,
        round_num: int,
        theme: str,
        questions: List[str],
        convergences: Optional[List[str]] = None,
        tensions: Optional[List[str]] = None,
        model_r1_synthesis: Optional[str] = None,
        convergence_question: str = "",
    ) -> str:
        """Generate prompt for a given round."""
        if round_num == 1:
            return round1_template(
                self.motion, self.definition, questions, theme, self.n_rounds
            )
        elif round_num == 2:
            return round2_template(
                model_r1_synthesis or "[Model R1 summary]",
                model_r1_synthesis or "",
                convergences or [],
                tensions or [],
                questions,
                theme,
                convergence_question,
            )
        elif round_num == self.n_rounds:
            # Final round = implementation/commit
            artifact_constraints = [
                f"Must be buildable in 48 hours by a competent solo developer",
                "Must demonstrate feasibility or falsify it",
                "Must resolve the accommodation/atrophy differential problem",
                "Must include your preferred override mechanism",
                "Must be concrete enough to git clone and run",
            ]
            spec_format = (
                "Module Name: [exact filename]\n"
                "Lines: [count, must be ≤500]\n"
                "Dependencies: [list, must be pip-installable]\n\n"
                "Core Classes/Functions:\n"
                "1. [function signature + 1-line purpose]\n\n"
                "Test Cases:\n"
                "- Input: [specific scenario]\n"
                "- Expected: [specific output]\n"
                "- Proves: [what this validates]"
            )
            return round3_template(
                convergences or [],
                tensions or [],
                "Module",
                artifact_constraints,
                spec_format,
                convergence_question or (
                    "If a junior developer builds this module exactly as specified and runs the test cases, "
                    "what empirical result would prove PAIS is viable? What result would prove it's not?"
                ),
            )
        else:
            # Generic mid-round
            return round2_template(
                model_r1_synthesis or "",
                model_r1_synthesis or "",
                convergences or [],
                tensions or [],
                questions,
                theme,
                convergence_question,
            )
    
    def cross_pollination_prompt(
        self,
        resolved_tensions: List[str],
        remaining_tensions: List[str],
        history_block: str = "<<HISTORY>>[All prior rounds included above]<</HISTORY>>",
    ) -> str:
        """Generate cross-pollination prompt after final round."""
        return cross_pollination_template(history_block, resolved_tensions, remaining_tensions)
    
    def register_model_response(self, model_name: str, profile: ModelProfile) -> None:
        """Register a model's Phase 0 registration response."""
        self.models[model_name] = profile
    
    def log_round(self, round_num: int, log_text: str) -> None:
        """Store a round's full log."""
        self.round_logs[round_num] = log_text
        self.round_convergences[round_num] = []
        self.round_tensions[round_num] = []
    
    def extract_convergences(self, round_num: int, convergences: List[str]) -> None:
        """Record convergence points from a round."""
        self.round_convergences[round_num] = convergences
    
    def extract_tensions(self, round_num: int, tensions: List[str]) -> None:
        """Record unresolved tensions from a round."""
        self.round_tensions[round_num] = tensions
    
    def generate_chunk_hash(self, content: str) -> str:
        """Generate a content hash for archive verification."""
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def archive_to_rlm(
        self,
        output_dir: str,
        full_log: str,
        key_convergences: List[str],
        key_divergences: List[str],
        code_artifacts: Optional[str] = None,
    ) -> Dict:
        """
        Generate RLM archive files.
        
        Creates:
          - RLM_INDEX_{session_id}.json
          - RLM_{session_id}.json  
          - chunks/{session_id}-r{N}-all-models.txt
          - chunks/{session_id}-code-modules.txt
        
        Returns the index JSON structure.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        chunks_dir = out / "chunks"
        chunks_dir.mkdir(exist_ok=True)
        
        now = datetime.datetime.utcnow().isoformat() + "Z"
        
        # Write round chunk files
        chunk_index = {}
        for r, log in self.round_logs.items():
            chunk_id = f"{self.session_id}-r{r}-all-models"
            chunk_path = chunks_dir / f"{chunk_id}.txt"
            content = f"# Round {r} Complete\n# Model: all\n# Round: {r}\n# Chunk ID: {chunk_id}\n# Hash: {self.generate_chunk_hash(log)}\n# ---\n\n{log}"
            chunk_path.write_text(content, encoding='utf-8')
            chunk_index[chunk_id] = {
                "round": r,
                "path": str(chunk_path),
                "hash": self.generate_chunk_hash(log),
            }
        
        # Write code artifacts chunk
        if code_artifacts:
            code_id = f"{self.session_id}-code-modules"
            code_path = chunks_dir / f"{code_id}.txt"
            code_path.write_text(code_artifacts, encoding='utf-8')
            chunk_index[code_id] = {
                "round": "all",
                "path": str(code_path),
                "hash": self.generate_chunk_hash(code_artifacts),
            }
        
        # Build main RLM JSON
        rlm_data = {
            "_rlm_metadata": {
                "version": "1.0",
                "session_id": self.session_id,
                "created": self.start_time,
                "last_updated": now,
            },
            "deliberation_identity": {
                "name": f"Octagon Deliberation — {self.session_id}",
                "exercise_id": self.session_id,
                "motion": self.motion,
                "date_start": self.start_time,
                "date_end": now,
                "duration_rounds": self.n_rounds,
                "models_participating": list(self.models.keys()),
                "focus": self.definition,
            },
            "round_summary": {
                f"round_{r}": {
                    "convergences": self.round_convergences.get(r, []),
                    "tensions": self.round_tensions.get(r, []),
                }
                for r in range(1, self.n_rounds + 1)
            },
            "key_convergences": key_convergences,
            "key_divergences": key_divergences,
            "model_profiles": {
                name: asdict(profile)
                for name, profile in self.models.items()
            },
            "chunk_index": chunk_index,
            "cross_references": {},
            "query_endpoints": {
                "by_round": f"Filter chunk_index by round number",
                "by_model": f"Filter model_profiles by name",
                "by_convergence": f"Search key_convergences",
            }
        }
        
        # Write main RLM JSON
        rlm_path = out / f"RLM_{self.session_id}.json"
        rlm_path.write_text(json.dumps(rlm_data, indent=2), encoding='utf-8')
        
        # Write index JSON (lightweight)
        index_data = {
            "rlm_memory_index": {
                "version": "1.0",
                "last_updated": now,
                "memory_id": self.session_id,
                "storage_location": str(rlm_path),
                "tags": [
                    "octagon", self.session_id,
                    f"{self.n_rounds}-round-deliberation",
                    f"{self.n_models}-model",
                ] + [m.lower().replace(" ", "-").replace(".", "") for m in self.models.keys()],
                "retention": "permanent",
                "access": "cross_node",
            }
        }
        index_path = out / f"RLM_INDEX_{self.session_id}.json"
        index_path.write_text(json.dumps(index_data, indent=2), encoding='utf-8')
        
        # Write full log
        log_path = out / f"{self.session_id}_FULLLOG.txt"
        log_path.write_text(full_log, encoding='utf-8')
        
        return index_data


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE EXECUTION CHECKLIST (as executable validation)
# ─────────────────────────────────────────────────────────────────────────────

CHECKLIST = [
    "Motion defined with MOTION / DEFINITION / STRUCTURE (3-part)",
    "Response format has EXACT required headers — no deviation",
    "All model capability profiles registered in Phase 0 before Round 1",
    "Credit consumption tracked per model per round",
    "Round 1 convergences and tensions extracted before drafting R2 prompt",
    "Round 2 names exactly the unresolved tensions (max 3) passed to R3",
    "Round 3 prompt includes: NO HEDGING. Pick ONE mechanism.",
    "Cross-pollination: each model sees ALL other R3 specs simultaneously",
    "One model designated as synthesizer for consensus artifact (recommend Grok/GPT)",
    "Archive in RLM format: JSON index + chunk files + full log",
    "Unresolved gaps documented separately from convergence points",
    "Explicit falsifiability criteria set before build — not after",
]

def validate_session_ready(session: OctagonSession) -> Tuple[bool, List[str]]:
    """Validate session is properly configured before running."""
    failures = []
    if not session.motion:
        failures.append("Motion not defined")
    if not session.definition:
        failures.append("Definition not defined")
    if session.n_models < 2:
        failures.append("Need at least 2 models")
    if session.n_rounds < 1:
        failures.append("Need at least 1 round")
    return len(failures) == 0, failures


# ─────────────────────────────────────────────────────────────────────────────
# QUICK-START EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

def demo():
    """Demonstrate the pipeline with the PAIS motion."""
    
    motion = (
        "Resolved: That a Personal Agency Immune System (PAIS) is technically feasible, "
        "ethically sound, and operationally necessary to prevent cognitive atrophy in "
        "human-AI co-processing environments."
    )
    definition = (
        "PAIS is infrastructure that monitors human-AI interactions, detects patterns of "
        "excessive delegation (agency degradation), and intervenes with constructive friction "
        "to preserve human decision-making capacity."
    )
    
    session = OctagonSession(
        motion=motion,
        definition=definition,
        n_rounds=3,
        n_models=8,
        session_id="PAIS-2026-02-20",
    )
    
    valid, failures = validate_session_ready(session)
    print(f"Session valid: {valid}")
    if failures:
        print(f"Failures: {failures}")
        return
    
    # Generate Phase 0 registration prompt
    reg_prompt = session.registration_prompt()
    print("\n=== PHASE 0: REGISTRATION PROMPT (first 500 chars) ===")
    print(reg_prompt[:500] + "...")
    
    # Generate Round 1 prompt
    r1_prompt = session.round_prompt(
        round_num=1,
        theme="Technical Feasibility",
        questions=[
            "Behavioral Signatures of Agency Degradation: Identify 3-5 specific, observable patterns that would indicate a human is delegating agency to AI rather than using AI as a tool. These must be concrete enough to implement in code.",
            "Intervention Point in Request-Response Loop: Identify the specific technical point where PAIS could most effectively reintroduce friction without destroying utility. Be specific about implementation mechanism.",
            "False Positive Risk: What is the specific false-positive risk? How do we distinguish between harmful delegation and legitimate automation of low-stakes decisions? Provide a decision boundary or metric.",
        ],
    )
    print("\n=== ROUND 1 PROMPT (first 500 chars) ===")
    print(r1_prompt[:500] + "...")
    
    # Generate Round 3 (implementation) prompt
    r3_prompt = session.round_prompt(
        round_num=3,
        theme="Final Resolution",
        questions=[],
        convergences=[
            "All models agree on Hybrid control (platform floors + user calibration)",
            "All acknowledge neurodivergent accommodation requires differential diagnosis",
            "All agree API middleware is the preferred intervention point",
        ],
        tensions=[
            "Override mechanism: Hard platform override vs user-can-override vs friction budget",
            "Differential diagnosis: Micro-probes vs three-factor fingerprint vs multi-signal",
            "Module scope: Scoring engine vs telemetry baseline vs full decision engine",
        ],
        convergence_question=(
            "If a junior developer builds this module exactly as specified and runs the test cases, "
            "what empirical result would prove PAIS is viable? What result would prove it's not?"
        ),
    )
    print("\n=== ROUND 3 PROMPT (first 500 chars) ===")
    print(r3_prompt[:500] + "...")
    
    print("\n=== EXECUTION CHECKLIST ===")
    for i, item in enumerate(CHECKLIST, 1):
        print(f"  [{i:2d}] {item}")
    
    print(f"\n=== SESSION: {session.session_id} ===")
    print(f"Motion: {session.motion[:80]}...")
    print(f"Rounds: {session.n_rounds} | Models: {session.n_models}")
    print("\nPipeline ready. Collect model responses, then call session.archive_to_rlm()")


if __name__ == "__main__":
    demo()
