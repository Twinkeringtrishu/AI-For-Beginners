# r38-t034: Task Plan validation and topological ordering
"""
This module provides a simple utility to validate a small task plan expressed as a
mapping of task identifiers to the list of identifiers they depend on.  It
performs three checks:

1. **Missing dependencies** – if a task lists a dependency that does not exist
   in the plan, a ``ValueError`` is raised mentioning the unknown ID.
2. **Cycle detection** – if the dependency graph contains a cycle, a ``ValueError``
   is raised indicating a cycle.
3. **Topological ordering** – when the plan is valid, the tasks are returned in a
   order that respects the dependencies.

In addition to the final order, the function records the **intermediate state**
after each step of Kahn's algorithm.  Each intermediate record is a dictionary
containing:

* ``processed`` – the task just removed from the graph.
* ``order_sofar`` – the list of tasks ordered up to this point.
* ``remaining`` – the current dependency graph (tasks still pending).

The sentinel for missing or cycle errors is the standard ``ValueError`` – this
matches the pattern used in earlier applied‑labs.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import json

def validate_and_order(task_deps: Dict[str, List[str]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Validate the task plan and return a topological order.

    Parameters
    ----------
    task_deps:
        Mapping from a *task id* to a list of *task ids* it depends on.

    Returns
    -------
    order:
        List of task ids in a valid execution order.
    intermediate_states:
        List of dictionaries describing the algorithm state after each removal.
    """
    # --- 1. Check for missing dependencies -----------------------------------
    all_ids = set(task_deps.keys())
    for tid, deps in task_deps.items():
        for d in deps:
            if d not in all_ids:
                raise ValueError(f"Missing dependency: '{d}' referenced by '{tid}'")

    # --- 2. Kahn's algorithm with state capture -----------------------------
    # Build the inverse adjacency (incoming edge count)
    # Compute incoming edge count (number of dependencies each task has)
    in_degree: Dict[str, int] = {tid: len(deps) for tid, deps in task_deps.items()}


    # Initialise queue with nodes that have no incoming edges
    ready = [tid for tid, deg in in_degree.items() if deg == 0]
    order: List[str] = []
    intermediate: List[Dict[str, Any]] = []

    # Work on a mutable copy of the graph
    graph = {tid: list(deps) for tid, deps in task_deps.items()}

    while ready:
        current = ready.pop(0)  # deterministic order
        order.append(current)

        # Record state *after* extracting current
        intermediate.append({
            "processed": current,
            "order_sofar": list(order),
            "remaining": {k: list(v) for k, v in graph.items() if k not in order}
        })

        # Remove edges outgoing from current
        for tid, deps in list(graph.items()):
            if current in deps:
                deps.remove(current)
                in_degree[tid] -= 1
                if in_degree[tid] == 0:
                    ready.append(tid)
        # Clean up to keep graph tidy (optional)
        graph.pop(current, None)

    if len(order) != len(task_deps):
        # Cycle detected – the remaining keys form the cycle component
        remaining = set(task_deps.keys()) - set(order)
        raise ValueError(f"Cycle detected involving tasks: {', '.join(sorted(remaining))}")

    return order, intermediate

# ---------------------------------------------------------------------------
# Simple demonstration when the module is executed directly.
if __name__ == "__main__":
    example_plan = {
        "A": [],
        "B": ["A"],
        "C": ["B"]
    }
    try:
        order, states = validate_and_order(example_plan)
        print("Topological order:", order)
        print("Intermediate states (JSON pretty):")
        print(json.dumps(states, indent=2))
    except ValueError as e:
        print("Error:", e)
