"""
R38-T038: Finite-state reward trace — evaluation experiment.
Lesson: lessons/6-Other/22-DeepRL

Disclaimer:
This environment is a discrete 3-state teaching simulation for inspecting Markov
decision process (MDP) transitions, sparse rewards, and horizon step caps.
It is NOT a deep neural network, GPU-accelerated physics engine, or continuous control
benchmark, and makes no claims about real-world model accuracy or trained policy performance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

VALID_ACTIONS = {"left", "right"}
TERMINAL_STATE = 2
DEFAULT_MAX_STEPS = 5


@dataclass(frozen=True)
class StepOutcome:
    """Record of a single transition step in the environment."""
    step: int
    state_before: int
    action: str
    state_after: int
    reward: int
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class EpisodeResult:
    """Comprehensive summary of a simulated episode."""
    start_state: int
    final_state: int
    actions_taken: List[str]
    total_reward: int
    steps_taken: int
    max_steps: int
    reached_goal: bool
    truncated: bool
    step_history: List[StepOutcome]


@dataclass(frozen=True)
class ConfigurationComparison:
    """Evaluation metrics for comparing different policy configurations."""
    config_name: str
    description: str
    action_sequence: List[str]
    max_steps: int
    steps_taken: int
    final_state: int
    total_reward: int
    reached_goal: bool
    sample_efficiency: float  # total_reward / steps_taken (if steps > 0)


class ThreeStateChainEnv:
    """
    A 3-state deterministic chain environment: [0] <-> [1] -> [2 (terminal)].

    Rules:
    - States: 0, 1, 2.
    - Start state: defaults to 0.
    - Actions: 'left', 'right'.
    - Transition at 0:
        'left' -> stays at 0 (bounded)
        'right' -> moves to 1
    - Transition at 1:
        'left' -> moves to 0
        'right' -> moves to 2 (terminal)
    - Transition at 2:
        Terminal state. No further moves allowed.
    - Reward:
        +1 on entering state 2 (from state 1 via 'right').
        0 on all other transitions.
    - Termination:
        Terminates immediately upon entering terminal state 2.
    """

    def __init__(self, start_state: int = 0) -> None:
        if not isinstance(start_state, int) or start_state not in {0, 1, 2}:
            raise ValueError(f"Invalid start_state: {start_state}. Must be 0, 1, or 2.")
        self.state = start_state
        self.start_state = start_state

    def reset(self, start_state: Optional[int] = None) -> int:
        """Reset environment to initial state."""
        if start_state is not None:
            if not isinstance(start_state, int) or start_state not in {0, 1, 2}:
                raise ValueError(f"Invalid start_state: {start_state}. Must be 0, 1, or 2.")
            self.start_state = start_state
        self.state = self.start_state
        return self.state

    def step(self, action: str) -> Tuple[int, int, bool]:
        """
        Execute one action in the environment.

        Returns
        -------
        next_state : int
        reward : int
        terminated : bool
        """
        if not isinstance(action, str):
            raise TypeError(f"Action must be a str, got {type(action).__name__}")
        action = action.lower()
        if action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Valid actions are {sorted(VALID_ACTIONS)}")

        if self.state == TERMINAL_STATE:
            raise RuntimeError("Cannot step from terminal state 2. Call reset() first.")

        prev_state = self.state
        reward = 0
        terminated = False

        if prev_state == 0:
            if action == "right":
                self.state = 1
            else:  # "left"
                self.state = 0
        elif prev_state == 1:
            if action == "right":
                self.state = 2
                reward = 1
                terminated = True
            else:  # "left"
                self.state = 0

        return self.state, reward, terminated


def run_episode(
    actions: Sequence[str],
    max_steps: int = DEFAULT_MAX_STEPS,
    start_state: int = 0,
) -> EpisodeResult:
    """
    Run an episode executing a planned sequence of actions with an explicit step cap.

    Parameters
    ----------
    actions : Sequence[str]
        Ordered sequence of actions to attempt.
    max_steps : int
        Maximum number of steps permitted before episode truncation (step cap).
    start_state : int
        Starting state (0, 1, or 2).

    Returns
    -------
    EpisodeResult
    """
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError(f"max_steps must be a positive integer, got {max_steps}")

    env = ThreeStateChainEnv(start_state=start_state)
    history: List[StepOutcome] = []
    total_reward = 0
    actions_taken: List[str] = []

    for step_num in range(1, max_steps + 1):
        if step_num - 1 >= len(actions):
            # Exhausted action plan before hitting step cap or terminal
            break

        act = actions[step_num - 1]
        state_before = env.state

        if state_before == TERMINAL_STATE:
            break

        next_state, reward, terminated = env.step(act)
        total_reward += reward
        actions_taken.append(act)

        # Check if capped at this step while not terminated
        truncated = (step_num == max_steps) and not terminated

        history.append(
            StepOutcome(
                step=step_num,
                state_before=state_before,
                action=act,
                state_after=next_state,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )
        )

        if terminated:
            break

    final_state = env.state
    reached_goal = (final_state == TERMINAL_STATE)
    is_truncated = (len(actions_taken) == max_steps) and not reached_goal

    return EpisodeResult(
        start_state=start_state,
        final_state=final_state,
        actions_taken=actions_taken,
        total_reward=total_reward,
        steps_taken=len(actions_taken),
        max_steps=max_steps,
        reached_goal=reached_goal,
        truncated=is_truncated,
        step_history=history,
    )


def compare_three_configurations() -> List[ConfigurationComparison]:
    """
    Evaluate and compare three controlled configurations on the 3-state chain:
    1. Config A (Optimal Direct Policy): ['right', 'right'] -> 2 steps, reward 1.
    2. Config B (Hesitant / Exploration Policy): ['right', 'left', 'right', 'right'] -> 4 steps, reward 1.
    3. Config C (Adversarial / Left-Biased Policy): ['left', 'left', 'left', 'left', 'left'] -> 5 steps, reward 0.
    """
    configs = [
        {
            "name": "Config-A (Optimal Greedy)",
            "desc": "Direct path taking 'right' at each state to reach terminal state 2 immediately.",
            "actions": ["right", "right"],
            "max_steps": 5,
        },
        {
            "name": "Config-B (Hesitant / Exploration)",
            "desc": "Path with backtrack before recovering to goal: ['right', 'left', 'right', 'right'].",
            "actions": ["right", "left", "right", "right"],
            "max_steps": 5,
        },
        {
            "name": "Config-C (Left-Biased / Adversarial)",
            "desc": "Repeated 'left' actions stuck at lower boundary until capped at max_steps (5).",
            "actions": ["left", "left", "left", "left", "left"],
            "max_steps": 5,
        },
    ]

    comparisons: List[ConfigurationComparison] = []
    for cfg in configs:
        res = run_episode(actions=cfg["actions"], max_steps=cfg["max_steps"], start_state=0)
        eff = res.total_reward / res.steps_taken if res.steps_taken > 0 else 0.0
        comparisons.append(
            ConfigurationComparison(
                config_name=cfg["name"],
                description=cfg["desc"],
                action_sequence=list(cfg["actions"]),
                max_steps=cfg["max_steps"],
                steps_taken=res.steps_taken,
                final_state=res.final_state,
                total_reward=res.total_reward,
                reached_goal=res.reached_goal,
                sample_efficiency=eff,
            )
        )

    return comparisons


def get_evaluation_report(comparisons: List[ConfigurationComparison]) -> str:
    """Generate a structured text evaluation report comparing the configurations."""
    lines = [
        "==========================================================================",
        "R38-T038: Three Controlled Configurations Evaluation & Trade-off Analysis",
        "==========================================================================",
        f"{'Configuration':<32} | {'Steps':<5} | {'Reward':<6} | {'Goal?':<5} | {'Efficiency':<10}",
        "-" * 74,
    ]
    for c in comparisons:
        lines.append(
            f"{c.config_name:<32} | {c.steps_taken:<5} | {c.total_reward:<6} | {str(c.reached_goal):<5} | {c.sample_efficiency:<10.3f}"
        )
    lines.extend([
        "-" * 74,
        "Observed Trade-offs:",
        "1. Correctness: Config-A achieves the optimal policy contract ([right, right] -> state 2, reward 1).",
        "2. Exploration vs Speed: Config-B reaches the goal despite a temporary backtrack (efficiency 0.250 vs 0.500).",
        "3. Horizon Safety: Config-C safely caps run at 5 steps, preventing infinite loops on non-terminating policies.",
        "4. Simulation Disclaimer: Pure algorithmic evaluation; does not represent trained model inference accuracy.",
        "==========================================================================",
    ])
    return "\n".join(lines)


def main() -> None:
    """Execute canonical verification, evaluation experiment, and write evidence."""
    # 1. Canonical contract checks
    res_optimal = run_episode(["right", "right"], max_steps=5, start_state=0)
    res_left = run_episode(["left"] * 5, max_steps=5, start_state=0)

    print("Canonical Contract 1: [right, right]")
    print(f"  Final State: {res_optimal.final_state}, Total Reward: {res_optimal.total_reward}, Steps: {res_optimal.steps_taken}")

    print("\nCanonical Contract 2: Repeated left capped at 5 steps")
    print(f"  Final State: {res_left.final_state}, Total Reward: {res_left.total_reward}, Steps: {res_left.steps_taken}, Truncated: {res_left.truncated}")

    # 2. Controlled configuration comparison
    comparisons = compare_three_configurations()
    report = get_evaluation_report(comparisons)
    print("\n" + report)

    # 3. Persist evidence
    repo_root = Path(__file__).resolve().parents[5]
    work_dir = repo_root / "work" / "R38-T038"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_file = work_dir / "example_run.txt"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("R38-T038 Finite-State Reward Trace & Evaluation Experiment\n\n")
        f.write(report)
        f.write("\n\nStructured Results JSON:\n")
        json.dump([asdict(c) for c in comparisons], f, indent=2)
        f.write("\n")

    print(f"\nWrote execution evidence to: {out_file}")


if __name__ == "__main__":
    main()
