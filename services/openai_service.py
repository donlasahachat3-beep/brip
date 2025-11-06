"""OpenAI integration layer powering the self-generating multi-agent framework.

This module introduces a reusable service abstraction around OpenAI's API and a
multi-agent orchestration layer that enables autonomous planning and execution
workflows. It is designed to be the foundation for a "Build AI" capability
where agents can coordinate, generate code, and iteratively expand system
functionality without manual intervention.

Key concepts
------------

``OpenAIService``
    Thin wrapper on top of the REST API that provides retry-aware helpers to
    execute chat completion requests. It centralises configuration (API base,
    default model, workspace directory) and can be reused anywhere in the
    project.

``MultiAgentOrchestrator``
    Manages a graph of agent profiles. It can auto-generate a project plan with
    OpenAI, execute each step with specialised agents, and maintain shared
    state between them. Agents can be pure LLM personas or Python callables for
    deterministic operations.

``AgentProfile``
    Declarative description of an agent, including its system prompt, optional
    execution handler, and preferred model. Profiles let you compose hybrid
    human/LLM/tool workflows.

The orchestrator ships with a default planner/architect/implementer trio that
can be extended to produce a fully autonomous dev loop. Downstream services can
consume the orchestrator to kick off self-building routines, generate
blueprints, or integrate the resulting plans with Celery tasks.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import httpx

from config.loader import get_settings


AgentHandler = Callable[["AgentStep", "MultiAgentContext"], "AgentExecutionResult"]


@dataclass
class AgentProfile:
    """Declarative agent description for the multi-agent framework."""

    name: str
    description: str
    system_prompt: str
    model: Optional[str] = None
    handler: Optional[AgentHandler] = None
    temperature: float = 0.2


@dataclass
class AgentStep:
    """Single unit of work assigned to an agent."""

    order: int
    agent: str
    instructions: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[int] = field(default_factory=list)


@dataclass
class AgentExecutionResult:
    """Result payload produced by an agent."""

    agent: str
    order: int
    output: str
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiAgentPlan:
    """Structured execution plan that can be processed by the orchestrator."""

    objective: str
    steps: List[AgentStep]
    raw_plan: str


@dataclass
class MultiAgentReport:
    """Summary of a completed multi-agent run."""

    objective: str
    steps: List[AgentStep]
    results: List[AgentExecutionResult]
    final_summary: str
    shared_state: Dict[str, Any]


@dataclass
class MultiAgentContext:
    """Execution context provided to custom agent handlers."""

    objective: str
    shared_state: Dict[str, Any]
    orchestrator: "MultiAgentOrchestrator"


class OpenAIService:
    """HTTP-based helper for interacting with OpenAI endpoints.

    Parameters
    ----------
    api_key:
        Secret token used to authenticate with OpenAI. Falls back to the
        ``vc_api_key`` from settings or the ``OPENAI_API_KEY`` environment
        variable when not provided.
    api_base:
        Root URL for the API. Defaults to ``https://api.openai.com/v1`` but can
        be pointed to custom gateways or Azure-compatible mirrors.
    default_model / planning_model:
        Models used by general-purpose agents and the planner respectively.
    workspace_dir:
        Directory on disk where agents can persist artifacts during an auto
        build cycle. The service ensures the directory exists.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_base: str = "https://api.openai.com/v1",
        default_model: str = "gpt-4.1-mini",
        planning_model: str = "gpt-4.1",
        workspace_dir: str = "./auto_build_workspace",
        timeout: float = 120.0,
        retry_limit: int = 3,
        retry_backoff: float = 2.5,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.vc_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OpenAI API key is not configured. Provide it via settings `vc_api_key` "
                "or the OPENAI_API_KEY environment variable."
            )

        self.api_base = api_base.rstrip("/") or settings.openai_api_base.rstrip("/")
        self.default_model = default_model or settings.openai_default_model
        self.planning_model = planning_model or settings.openai_planning_model
        self.timeout = timeout
        self.retry_limit = retry_limit
        self.retry_backoff = retry_backoff

        self.workspace_dir = Path(workspace_dir or settings.openai_workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self._client = httpx.Client(
            base_url=self.api_base,
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------
    def chat_completion(
        self,
        messages: Iterable[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute a chat completion request with retry/backoff."""

        payload = {
            "model": model or self.default_model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        return self._post("/chat/completions", payload)

    def summarize(self, *, objective: str, results: List[AgentExecutionResult]) -> str:
        """Generate a final run summary for reporting purposes."""

        messages = [
            {
                "role": "system",
                "content": (
                    "You summarise the progress of a multi-agent autonomous dev cycle. "
                    "Provide concise bullet points and a final recommendation."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "objective": objective,
                        "results": [result.__dict__ for result in results],
                    },
                    indent=2,
                ),
            },
        ]
        response = self.chat_completion(messages, model=self.default_model, temperature=0.1)
        return extract_text(response)

    def close(self) -> None:
        """Close the underlying HTTP client."""

        self._client.close()

    # ------------------------------------------------------------------
    # Internal networking helpers
    # ------------------------------------------------------------------
    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.retry_limit + 1):
            try:
                response = self._client.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as exc:  # pragma: no cover - network exceptions vary
                last_error = exc
                if attempt >= self.retry_limit:
                    break
                time.sleep(self.retry_backoff * attempt)
        raise RuntimeError(f"OpenAI request failed: {last_error}")


class MultiAgentOrchestrator:
    """Coordinates the Build-AI workflow across specialised agents."""

    def __init__(self, service: OpenAIService) -> None:
        self.service = service
        self._agents: Dict[str, AgentProfile] = {}

    # ------------------------------------------------------------------
    # Agent registration and configuration
    # ------------------------------------------------------------------
    def register_agent(self, profile: AgentProfile) -> None:
        if profile.name in self._agents:
            raise ValueError(f"Agent '{profile.name}' is already registered")
        self._agents[profile.name] = profile

    def bootstrap_default_agents(self) -> None:
        """Register a baseline planner/architect/implementer trio."""

        if self._agents:
            return

        self.register_agent(
            AgentProfile(
                name="planner",
                description="Decomposes a product goal into structured steps.",
                system_prompt=(
                    "You are a strategic planner for an autonomous engineering team. "
                    "Break the user's objective into 3-7 ordered steps. Each step must "
                    "name the agent responsible (planner, architect, implementer, reviewer) "
                    "and clearly describe the work required. Output JSON with fields "
                    "steps -> [{order, agent, instructions}]."
                ),
                model=self.service.planning_model,
            )
        )

        self.register_agent(
            AgentProfile(
                name="architect",
                description="Designs technical approaches, data flows, and scaffolding.",
                system_prompt=(
                    "You are the software architect in an autonomous dev team. Given the "
                    "objective and the planner's instructions, produce a technical design, "
                    "including module breakdowns, integration points, and risks. Return "
                    "clear actionable guidance for the implementer."
                ),
            )
        )

        self.register_agent(
            AgentProfile(
                name="implementer",
                description="Creates concrete code or configuration diffs.",
                system_prompt=(
                    "You are the implementation specialist. Produce code-level changes, "
                    "migration steps, or commands needed to fulfil the instruction. Provide "
                    "succinct explanations alongside code blocks."
                ),
            )
        )

        self.register_agent(
            AgentProfile(
                name="reviewer",
                description="Validates work and identifies follow-up actions.",
                system_prompt=(
                    "You are the QA/reviewer agent. Inspect previous outputs and confirm "
                    "whether the objective is satisfied. List open issues or tests."
                ),
            )
        )

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def generate_plan(self, objective: str, *, context: Optional[Dict[str, Any]] = None) -> MultiAgentPlan:
        """Ask the planner agent to craft a structured execution plan."""

        planner = self._require_agent("planner")
        messages = [
            {"role": "system", "content": planner.system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "objective": objective,
                        "context": context or {},
                        "available_agents": list(self._agents.keys()),
                    },
                    indent=2,
                ),
            },
        ]
        response = self.service.chat_completion(messages, model=planner.model or self.service.planning_model)
        plan_text = extract_text(response)
        steps = parse_plan(plan_text)
        return MultiAgentPlan(objective=objective, steps=steps, raw_plan=plan_text)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(
        self,
        plan: MultiAgentPlan,
        *,
        shared_state: Optional[Dict[str, Any]] = None,
        max_rounds: Optional[int] = None,
    ) -> MultiAgentReport:
        """Execute the plan step-by-step and produce a final report."""

        shared_state = shared_state or {}
        results: List[AgentExecutionResult] = []
        self._history = []

        for step in plan.steps:
            profile = self._require_agent(step.agent)
            context = MultiAgentContext(objective=plan.objective, shared_state=shared_state, orchestrator=self)

            if profile.handler:
                result = profile.handler(step, context)
            else:
                result = self._invoke_agent(profile, step, context)

            results.append(result)

            if max_rounds is not None and len(results) >= max_rounds:
                break

        summary = self.service.summarize(objective=plan.objective, results=results)
        return MultiAgentReport(
            objective=plan.objective,
            steps=plan.steps,
            results=results,
            final_summary=summary,
            shared_state=shared_state,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _invoke_agent(
        self,
        profile: AgentProfile,
        step: AgentStep,
        context: MultiAgentContext,
    ) -> AgentExecutionResult:
        messages = [
            {"role": "system", "content": profile.system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "objective": context.objective,
                        "instructions": step.instructions,
                        "shared_state": context.shared_state,
                        "previous_results": [result.__dict__ for result in context.orchestrator._last_results()],
                    },
                    indent=2,
                ),
            },
        ]

        response = self.service.chat_completion(
            messages,
            model=profile.model or self.service.default_model,
            temperature=profile.temperature,
        )
        output_text = extract_text(response)
        metadata = {
            "raw_response": response,
        }
        execution_result = AgentExecutionResult(
            agent=profile.name,
            order=step.order,
            output=output_text,
            metadata=metadata,
        )
        self._store_result(execution_result)
        return execution_result

    def _require_agent(self, name: str) -> AgentProfile:
        try:
            return self._agents[name]
        except KeyError as exc:
            raise ValueError(f"Agent '{name}' is not registered") from exc

    def _store_result(self, result: AgentExecutionResult) -> None:
        if not hasattr(self, "_history"):
            self._history: List[AgentExecutionResult] = []
        self._history.append(result)

    def _last_results(self) -> List[AgentExecutionResult]:
        return getattr(self, "_history", [])


def parse_plan(plan_text: str) -> List[AgentStep]:
    """Parse a planner response into a list of ``AgentStep`` objects.

    The function expects either JSON or a simple numbered list. It purposely
    errs on the side of robustness so that the orchestrator can proceed even if
    the model deviates slightly from the expected output format.
    """

    plan_text = plan_text.strip()
    if not plan_text:
        return []

    # First try JSON
    try:
        payload = json.loads(plan_text)
        steps_payload = payload.get("steps") if isinstance(payload, dict) else payload
        steps: List[AgentStep] = []
        if isinstance(steps_payload, list):
            for entry in steps_payload:
                if not isinstance(entry, dict):
                    continue
                steps.append(
                    AgentStep(
                        order=int(entry.get("order", len(steps) + 1)),
                        agent=str(entry.get("agent", "implementer")),
                        instructions=str(entry.get("instructions", "")),
                        metadata={k: v for k, v in entry.items() if k not in {"order", "agent", "instructions"}},
                        dependencies=list(entry.get("dependencies", [])) if entry.get("dependencies") else [],
                    )
                )
        if steps:
            return sorted(steps, key=lambda step: step.order)
    except json.JSONDecodeError:
        pass

    # Fallback to line parsing e.g. "1. planner - Analyse requirements"
    steps: List[AgentStep] = []
    for idx, line in enumerate(plan_text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        agent = "implementer"
        instructions = line
        if "-" in line:
            prefix, instructions = [segment.strip() for segment in line.split("-", 1)]
            if ":" in prefix:
                _, maybe_agent = prefix.split(":", 1)
                agent = maybe_agent.strip() or agent
            elif " " in prefix:
                parts = prefix.split(" ", 1)
                agent = parts[1].strip()
        steps.append(AgentStep(order=idx, agent=agent, instructions=instructions))
    return steps


def extract_text(response: Dict[str, Any]) -> str:
    """Extract the assistant message text from a Chat Completions payload."""

    try:
        choices = response.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, list):
            # Some responses return a list of content parts
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return content or ""
    except AttributeError:  # pragma: no cover - defensive
        return ""


@lru_cache(maxsize=1)
def get_openai_service() -> OpenAIService:
    """Singleton accessor mirroring other service factories in the codebase."""

    settings = get_settings()
    return OpenAIService(
        api_key=settings.vc_api_key or os.getenv("OPENAI_API_KEY"),
        api_base=settings.openai_api_base,
        default_model=settings.openai_default_model,
        planning_model=settings.openai_planning_model,
        workspace_dir=settings.openai_workspace_dir,
    )


def build_default_orchestrator() -> MultiAgentOrchestrator:
    """Convenience helper returning a ready-to-use orchestrator instance."""

    orchestrator = MultiAgentOrchestrator(service=get_openai_service())
    orchestrator.bootstrap_default_agents()
    return orchestrator


__all__ = [
    "AgentExecutionResult",
    "AgentProfile",
    "AgentStep",
    "MultiAgentContext",
    "MultiAgentOrchestrator",
    "MultiAgentPlan",
    "MultiAgentReport",
    "OpenAIService",
    "build_default_orchestrator",
    "extract_text",
    "get_openai_service",
    "parse_plan",
]

