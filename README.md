# ISAC

Working repository for the sparse monostatic MIMO-ISAC paper, implementation, and experiments.

## Working branch

Current paper-method development is maintained on:

`paper-method-sync`

## Coordination files

- `METHOD_SPEC.md`: current mathematical method and symbol definitions.
- `EXPERIMENT_PROTOCOL.md`: experiment split, tuning, failure handling, and reporting rules.
- `RESULTS_LEDGER.md`: traceable experiment results only.
- `CODE_CHAT_HANDOFF.md`: instructions for the separate implementation/Codex session.

## Workflow

Paper-method discussion
→ update `METHOD_SPEC.md`
→ implementation on GitHub
→ Codex/experiment execution
→ update `RESULTS_LEDGER.md`
→ return to the paper discussion for interpretation and manuscript writing.

Do not let implementation or experiment sessions silently redefine the method. Mathematical changes must first be reflected in `METHOD_SPEC.md`.
