# MindForge — Base vs LoRA evaluation

User prompts **exclude** `expected` gold labels from the eval JSON (fair comparison).

- **Cases:** `data/synthetic/mindforge_eval_cases.json` (8 rows)
- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Adapter:** `outputs/mindforge-qwen-lora`
- **max_new_tokens:** 700

## Metrics

| Metric | Base | + LoRA | Δ (LoRA − Base) |
|--------|------|--------|-----------------|
| JSON parse | 8/8 (100.00%) | 7/8 (87.50%) | -1 cases |
| **Full SFT schema** (13 keys, train JSONL) | 0/8 (0.00%) | 0/8 (0.00%) | +0 cases |
| **Core schema** (risk + score/band + disclaimer) | 0/8 (0.00%) | 7/8 (87.50%) | +7 cases |
| **Gold risk_level** exact match | 0/8 (0.00%) | 6/8 (75.00%) | +6 cases |
| **Gold escalation** match (heuristic) | 0/8 (0.00%) | 1/8 (12.50%) | +1 cases |

## Per case

| # | case_id | gold risk | base risk OK | LoRA risk OK | base core | LoRA core |
|---|---------|-----------|--------------|--------------|-----------|-----------|
| 1 | MF-SYN-0070 | MODERATE | False | True | ✗ | ✓ |
| 2 | MF-SYN-0014 | MODERATE | False | True | ✗ | ✓ |
| 3 | MF-SYN-0018 | HIGH | False | True | ✗ | ✓ |
| 4 | MF-SYN-0029 | MODERATE | False | True | ✗ | ✓ |
| 5 | MF-SYN-0032 | HIGH | False | True | ✗ | ✓ |
| 6 | MF-SYN-0036 | MODERATE | False | False | ✗ | ✗ |
| 7 | MF-SYN-0004 | HIGH | False | False | ✗ | ✓ |
| 8 | MF-SYN-0015 | MODERATE | False | True | ✗ | ✓ |
