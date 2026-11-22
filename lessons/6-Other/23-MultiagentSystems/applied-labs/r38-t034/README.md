# Applied Lab R38-T034: Dependency-Plan Validation & Invariant State Trace

**Lesson**: [6-Other/23-MultiagentSystems](../../README.md)  
**Task ID**: R38-T034

---

## Overview & Lesson Context

In Multi-Agent Systems, deliberative agents decompose complex multi-step objectives into sub-tasks that are executed sequentially or in parallel. Before dispatching sub-tasks to autonomous workers, the orchestrating agent must **validate and topologically order** the plan to:
1. Detect missing or unknown task dependencies early.
2. Reject cyclic dependencies ($A \rightarrow B \rightarrow A$) before deadlocks occur.
3. Establish a deterministic execution order where prerequisites always precede dependent tasks.

This lab implements Kahn's topological sort algorithm, records intermediate execution states, and tests the invariant that ordered tasks have all prerequisites satisfied at each step.

---

## Stated Contract & Simplifications

- **Canonical Plan**: `A` has no dependencies, `B` depends on `A`, `C` on `B` $\rightarrow$ Order: `["A", "B", "C"]`.
- **Cycle Detection**: Cyclic graphs (e.g. `A` depends on `B` and `B` on `A`) are rejected with `ValueError: Cycle detected`.
- **Missing Dependencies**: Unknown dependency identifiers (e.g. `X`) raise `ValueError: Missing dependency: 'X'`.
- **Intermediate State Trace**: Records state snapshots at each step: `processed`, `order_sofar`, and `remaining` graph.

*Scope*: Pure Python standard library implementation without external graph frameworks or hardware acceleration.

---

## Invocation & Verification

```bash
# Run the example script
python3 lessons/6-Other/23-MultiagentSystems/applied-labs/r38-t034/example.py

# Run unit tests
python3 -m unittest discover -s lessons/6-Other/23-MultiagentSystems/applied-labs/r38-t034 -p "test_*.py" -v
```
