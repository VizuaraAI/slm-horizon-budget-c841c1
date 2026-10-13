"""Horizon-tunable task families with verifiable per-step ground truth.

Every family satisfies three properties that the experiment depends on:

1. The horizon H (number of dependent steps) is a free knob.
2. Every step has a single checkable ground-truth answer, so per-step
   competence is measured rather than inferred from end-to-end success.
3. The fixed preamble does not grow with H, so horizon effects are not
   confounded with context length.

Each step is also solvable from the *true* preceding state, which is what makes
recovery measurable: if the model errs at step i and then produces the answer
that is correct for the true state at step i+1, it has recovered.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Step:
    """One dependent step of an episode."""

    index: int
    instruction: str
    state_in: str
    answer: str

    def is_correct(self, response: str) -> bool:
        return normalize(response) == normalize(self.answer)


@dataclass
class Episode:
    family: str
    horizon: int
    preamble: str
    steps: list[Step]
    # Maps (state_in -> answer) so the harness can score a step taken from a
    # state the model drifted into, and detect recovery back to the true chain.
    transition: Callable[[int, str], str | None] = field(repr=False, default=lambda i, s: None)

    @property
    def final_answer(self) -> str:
        return self.steps[-1].answer


def normalize(text: str) -> str:
    """Compare answers up to whitespace, case and surrounding punctuation."""
    cleaned = text.strip().strip(string.punctuation + string.whitespace)
    # Take the last whitespace-separated token: small models like to prefix
    # "The answer is". We want to score the content, not the chattiness.
    tokens = cleaned.split()
    return tokens[-1].lower() if tokens else ""


# --------------------------------------------------------------------------- #
# ChainKV: pure execution, no arithmetic. Follow a pointer chain of length H.
# --------------------------------------------------------------------------- #
def make_chain_kv(rng: random.Random, horizon: int, vocab_size: int = 128, **_: object) -> Episode:
    keys = [f"k{n:03d}" for n in range(vocab_size)]
    # A random permutation guarantees the pointer chain never short-circuits.
    values = keys[:]
    rng.shuffle(values)
    table = dict(zip(keys, values))

    preamble_rows = "\n".join(f"{k} -> {v}" for k, v in table.items())
    preamble = (
        "You are following pointers through a lookup table.\n"
        "Table (one mapping per line):\n"
        f"{preamble_rows}\n\n"
        "At each step you are given a key. Reply with the single value that the "
        "table maps it to, and nothing else."
    )

    start = rng.choice(keys)
    steps: list[Step] = []
    current = start
    for i in range(horizon):
        answer = table[current]
        steps.append(
            Step(
                index=i,
                instruction=f"Step {i + 1}: the current key is {current}. What does it map to?",
                state_in=current,
                answer=answer,
            )
        )
        current = answer

    def transition(_index: int, state: str) -> str | None:
        return table.get(normalize(state))

    return Episode("chain_kv", horizon, preamble, steps, transition)


# --------------------------------------------------------------------------- #
# ChainArith: H chained affine updates mod m. Errors do not silently cancel.
# --------------------------------------------------------------------------- #
def make_chain_arith(rng: random.Random, horizon: int, modulus: int = 97, **_: object) -> Episode:
    coeffs = [(rng.randint(2, modulus - 1), rng.randint(1, modulus - 1)) for _ in range(horizon)]
    preamble = (
        f"You are executing a chain of modular updates with modulus {modulus}.\n"
        "At each step you are given the current value x and coefficients a and b. "
        f"Reply with (a * x + b) mod {modulus} as a single integer, and nothing else."
    )

    start = rng.randint(0, modulus - 1)
    steps: list[Step] = []
    current = start
    for i, (a, b) in enumerate(coeffs):
        answer = (a * current + b) % modulus
        steps.append(
            Step(
                index=i,
                instruction=f"Step {i + 1}: x = {current}, a = {a}, b = {b}. Compute (a * x + b) mod {modulus}.",
                state_in=str(current),
                answer=str(answer),
            )
        )
        current = answer

    def transition(index: int, state: str) -> str | None:
        if index >= len(coeffs):
            return None
        try:
            x = int(normalize(state))
        except ValueError:
            return None
        a, b = coeffs[index]
        return str((a * x + b) % modulus)

    return Episode("chain_arith", horizon, preamble, steps, transition)


# --------------------------------------------------------------------------- #
# FsmWalk: deterministic finite-state-machine execution. The branching factor
# sets per-step difficulty independently of the horizon.
# --------------------------------------------------------------------------- #
def make_fsm_walk(
    rng: random.Random, horizon: int, num_states: int = 8, branching: int = 3, **_: object
) -> Episode:
    states = [f"S{n}" for n in range(num_states)]
    symbols = list(string.ascii_lowercase[:branching])
    table = {s: {sym: rng.choice(states) for sym in symbols} for s in states}

    rows = "\n".join(
        f"{s}: " + ", ".join(f"{sym} -> {table[s][sym]}" for sym in symbols) for s in states
    )
    preamble = (
        "You are executing a deterministic finite-state machine.\n"
        "Transition table:\n"
        f"{rows}\n\n"
        "At each step you are given the current state and an input symbol. "
        "Reply with the single next state (for example S3) and nothing else."
    )

    inputs = [rng.choice(symbols) for _ in range(horizon)]
    steps: list[Step] = []
    current = rng.choice(states)
    for i, sym in enumerate(inputs):
        answer = table[current][sym]
        steps.append(
            Step(
                index=i,
                instruction=f"Step {i + 1}: state = {current}, input = {sym}. What is the next state?",
                state_in=current,
                answer=answer,
            )
        )
        current = answer

    def transition(index: int, state: str) -> str | None:
        if index >= len(inputs):
            return None
        key = normalize(state).upper()
        row = table.get(key)
        return None if row is None else row[inputs[index]]

    return Episode("fsm", horizon, preamble, steps, transition)


FAMILIES: dict[str, Callable[..., Episode]] = {
    "chain_kv": make_chain_kv,
    "chain_arith": make_chain_arith,
    "fsm": make_fsm_walk,
}


def make_episode(family: str, rng: random.Random, horizon: int, **kwargs: object) -> Episode:
    if family not in FAMILIES:
        raise KeyError(f"unknown task family '{family}'. Known: {sorted(FAMILIES)}")
    return FAMILIES[family](rng, horizon, **kwargs)
