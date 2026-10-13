"""Step solvers: the thing that produces an answer for one step.

Two backends.

``EchoSolver`` is a simulator. It never looks at the prompt; it emits the
ground-truth answer with a configured probability, and lowers that probability
once an error is already in the trajectory. Use it to exercise the whole
pipeline (harness, interventions, fitting, plots) in seconds, and to check that
the estimator recovers parameters you already know.

``HFSolver`` runs a real local model through transformers. This is the one that
produces paper numbers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from .tasks import Episode, Step


class Solver(Protocol):
    name: str

    def solve(self, episode: Episode, step: Step, state_in: str, degraded: bool,
              n_samples: int = 1, temperature: float = 0.0) -> list[str]:
        """Return ``n_samples`` candidate answers for one step."""
        ...


# --------------------------------------------------------------------------- #
@dataclass
class EchoSolver:
    """Synthetic step oracle with a tunable competence and contagion."""

    step_accuracy: float = 0.92
    contagion: float = 0.35
    seed: int = 0
    name: str = "echo"

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def _effective_accuracy(self, degraded: bool) -> float:
        if not degraded:
            return self.step_accuracy
        # Self-conditioning: once an error is in context, competence drops by a
        # fraction `contagion` of the remaining headroom below 1.
        return max(0.0, self.step_accuracy * (1.0 - self.contagion))

    def solve(self, episode: Episode, step: Step, state_in: str, degraded: bool,
              n_samples: int = 1, temperature: float = 0.0) -> list[str]:
        truth = episode.transition(step.index, state_in)
        if truth is None:
            # The trajectory has drifted somewhere the task does not define a
            # continuation from. There is no correct answer to emit, so every
            # sample is wrong. Returning step.answer here would silently
            # teleport the simulator back onto the true chain and manufacture a
            # recovery rate that no real model has.
            return [self._corrupt(step.answer) for _ in range(max(1, n_samples))]
        p = self._effective_accuracy(degraded)
        out: list[str] = []
        for _ in range(max(1, n_samples)):
            if self._rng.random() < p:
                out.append(truth)
            else:
                out.append(self._corrupt(truth))
        return out

    def _corrupt(self, truth: str) -> str:
        if truth.isdigit():
            return str((int(truth) + self._rng.randint(1, 9)) % 1000)
        chars = list(truth)
        if not chars:
            return "?"
        pos = self._rng.randrange(len(chars))
        chars[pos] = self._rng.choice("0123456789")
        return "".join(chars)


# --------------------------------------------------------------------------- #
@dataclass
class HFSolver:
    """A local HuggingFace causal LM, prompted one step at a time."""

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype: str = "bfloat16"
    device: str = "auto"
    max_new_tokens: int = 96
    show_history: int = 8  # how many previous (step, answer) pairs stay in context
    name: str = ""

    def __post_init__(self) -> None:
        import torch  # imported lazily so the echo path needs no torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = self.name or self.model_name
        torch_dtype = getattr(torch, self.dtype, torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch_dtype, device_map=self.device
        )
        self.model.eval()
        self._history: list[tuple[str, str]] = []

    def reset(self) -> None:
        """Clear the rolling in-context history. Called once per episode."""
        self._history = []

    def record(self, instruction: str, answer: str) -> None:
        """Append the model's own answer, which is how self-conditioning enters."""
        self._history.append((instruction, answer))

    def drop_last(self) -> None:
        """Context hygiene ablation: remove the most recent attempt."""
        if self._history:
            self._history.pop()

    def _messages(self, episode: Episode, step: Step, state_in: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": episode.preamble}]
        for instruction, answer in self._history[-self.show_history:]:
            messages.append({"role": "user", "content": instruction})
            messages.append({"role": "assistant", "content": answer})
        prompt = step.instruction
        if state_in != step.state_in:
            # The model drifted: ask it to continue from where it actually is.
            prompt = prompt.replace(step.state_in, state_in, 1)
        messages.append({"role": "user", "content": prompt})
        return messages

    def solve(self, episode: Episode, step: Step, state_in: str, degraded: bool,
              n_samples: int = 1, temperature: float = 0.0) -> list[str]:
        import torch

        messages = self._messages(episode, step, state_in)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0.0
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                num_return_sequences=max(1, n_samples),
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        cut = inputs["input_ids"].shape[-1]
        return [
            self.tokenizer.decode(seq[cut:], skip_special_tokens=True).strip()
            for seq in generated
        ]


def build_solver(model_cfg: dict) -> Solver:
    backend = str(model_cfg.get("backend", "echo")).lower()
    if backend == "echo":
        return EchoSolver(
            step_accuracy=float(model_cfg.get("echo_step_accuracy", 0.92)),
            contagion=float(model_cfg.get("echo_contagion", 0.35)),
            seed=int(model_cfg.get("seed", 0)),
        )
    if backend == "hf":
        return HFSolver(
            model_name=str(model_cfg.get("name", "Qwen/Qwen2.5-1.5B-Instruct")),
            dtype=str(model_cfg.get("dtype", "bfloat16")),
            device=str(model_cfg.get("device", "auto")),
            max_new_tokens=int(model_cfg.get("max_new_tokens", 96)),
        )
    raise ValueError(f"unknown backend '{backend}' (expected 'echo' or 'hf')")
