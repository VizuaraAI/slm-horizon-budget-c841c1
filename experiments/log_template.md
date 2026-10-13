# Experiment log

One row per run. The columns are deliberately the columns of Table 1 in
`../../paper/main.tex`, so when the table is full the paper's results section is
written. Copy the block at the bottom for each new run.

Fill `p_hat`, `delta`, `r`, `sigma`, `H50_obs` from the `summary.json` that
`src.evaluate` writes. `H50_pred` is the naive prediction `log(0.5)/log(p_hat)`,
printed in the same report.

## Main results: baseline, no intervention

Task family: `chain_kv` | episodes per horizon: 200 | decoding: greedy

| run | model | params | p_hat (%) | delta (pp) | r (%) | sigma | H50_pred | H50_obs | p*_naive (%) | p*_model (%) | gap (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Qwen2.5-0.5B-Instruct | 0.5B |  |  |  |  |  |  | 97.9 |  |  |
|  | Llama-3.2-1B-Instruct | 1.2B |  |  |  |  |  |  | 97.9 |  |  |
|  | Qwen2.5-1.5B-Instruct | 1.5B |  |  |  |  |  |  | 97.9 |  |  |
|  | SmolLM2-1.7B-Instruct | 1.7B |  |  |  |  |  |  | 97.9 |  |  |
|  | Gemma-2-2B-it | 2.6B |  |  |  |  |  |  | 97.9 |  |  |
|  | Qwen2.5-3B-Instruct | 3.1B |  |  |  |  |  |  | 97.9 |  |  |
|  | Llama-3.2-3B-Instruct | 3.2B |  |  |  |  |  |  | 97.9 |  |  |
|  | Phi-3-mini-4k-instruct | 3.8B |  |  |  |  |  |  | 97.9 |  |  |
|  | Qwen2.5-7B-Instruct | 7.6B |  |  |  |  |  |  | 97.9 |  |  |

`p*_naive` and `p*_model` are both quoted at `T = 0.5`, `H = 32`.
`p*_naive = 0.5^(1/32) = 0.9786` and does not depend on the model, which is the
whole point: it is what you would have concluded without running anything.

## Task families (same models, does the gap transfer?)

| run | model | family | p_hat (%) | delta (pp) | r (%) | H50_obs | gap (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | chain_kv |  |  |  |  |  |
|  |  | chain_arith |  |  |  |  |  |
|  |  | fsm |  |  |  |  |  |

## Intervention ledger (maps to Figure 2b)

| run | model | intervention | fwd passes / step | p_eff (%) | H50_obs | horizon gained | horizon per extra pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | none (greedy) | 1.0 |  |  | 0 | n/a |
|  |  | context hygiene | 1.0 |  |  |  |  |
|  |  | finer decomposition |  |  |  |  |  |
|  |  | verifier gate |  |  |  |  |  |
|  |  | vote k=3 |  |  |  |  |  |
|  |  | verifier + vote |  |  |  |  |  |

## Ablations

| id | question | run(s) | answer |
| --- | --- | --- | --- |
| A1 | self-conditioning or position? (repair the context, does delta collapse) |  |  |
| A2 | horizon vs. context length (pad the prompt at fixed H) |  |  |
| A3 | per-step difficulty vs. horizon (sweep fsm branching) |  |  |
| A4 | episodes needed to estimate p* at H=128 to within 1pp |  |  |
| A5 | does the requirement curve transfer to tau-bench / GAIA slices? |  |  |
| A6 | abstention: horizon vs. coverage |  |  |
| A7 | greedy vs. sampled decoding, and what it does to vote independence |  |  |

---

## Run record template

Copy this per run. The point of the "what surprised me" line is that it is where
the paper's actual contribution usually shows up first.

```
### run: <run_name>
date:          YYYY-MM-DD
command:       python -m src.evaluate --config configs/default.yaml ...
commit:        <git sha>
model:         <hf name>            backend: hf | echo
family:        chain_kv | chain_arith | fsm
horizons:      [1, 2, 4, 8, 16, 32, 64]    episodes/horizon: 200
intervention:  none | vote(k=3) | verifier(thr=0.5)
wall clock:    <hh:mm>              GPU: <device>

fitted:        p_hat=   delta=   r=   sigma=
               95% CI on p_hat: [    ,    ]
observed:      H50=          end-to-end S(H) curve: see summary.json
inverted:      p*_naive(32, 0.5)=        p*_model(32, 0.5)=        gap=     pp

sanity checks: [ ] echo backend recovers its configured accuracy
               [ ] S(H) is monotonically decreasing in H
               [ ] no horizon cell has fewer episodes than planned
               [ ] answer-format failures < 2% (otherwise you are measuring
                   formatting, not competence)

what surprised me:

what to run next:
```
