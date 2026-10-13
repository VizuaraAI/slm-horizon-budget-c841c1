"""slm-horizon-budget: inverting the compounding-error arithmetic for SLM agents.

Package layout
--------------
config          load YAML configs with dotted command-line overrides
tasks           horizon-tunable task families with per-step ground truth
solvers         step solvers (echo simulator, HuggingFace model)
interventions   voting / verifier gating applied at each step
harness         run episodes and log per-step correctness
horizon_model   forward law S(H) and its inversion to required competence
estimate        fit (p, delta, recovery, heterogeneity) from logged traces
plotting        figures for the paper
"""

__version__ = "0.1.0"
