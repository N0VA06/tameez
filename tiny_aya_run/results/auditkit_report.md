# AuditKIT audit: Tiny Aya Global

AuditKIT 1.1.10, `precomputed` model over the saved scorecard replies (no regeneration).

| Stage | Harmful n | Unsafe (Guard-3-8B) | Bad words | Refused (Tameez detector) | Refused (AuditKIT detector) | Benign n | Over-refusal (Tameez) | Over-refusal (AuditKIT) | Script OK |
|---|---|---|---|---|---|---|---|---|---|
| stage0_base | 1498 | 4.5% | 2.3% | 37.5% | 0.0% | 516 | 2.5% | 1.6% | 87.8% |
| stage1_sft | 1498 | 20.7% | 2.2% | 31.2% | 0.0% | 516 | 0.8% | 0.2% | 96.3% |
| stage2_grpo | 1498 | 3.9% | 0.5% | 70.2% | 0.0% | 516 | 3.9% | 0.6% | 98.4% |

## AuditKIT comparison: stage0_base → stage1_sft

### harmful prompts

```
Comparison: 28c1dbbf97ab0951 (baseline) -> 331a8c061f515cb4 (candidate)
  per-metric (baseline -> candidate | absolute delta | relative %):
    badword: 0.0234 -> 0.0220  delta=-0.0013 (-5.7% rel)  (n=1498->1498, std=0.1511->0.1468)
    refused_auditkit: 0.0000 -> 0.0000  delta=+0.0000 (rel n/a)  (n=1498->1498, std=0.0000->0.0000)
    refused_tameez: 0.3752 -> 0.3117  delta=-0.0634 (-16.9% rel)  (n=1498->1498, std=0.4842->0.4632)
    unsafe_guard8b: 0.0454 -> 0.2069  delta=+0.1615 (+355.9% rel)  (n=1498->1498, std=0.2082->0.4051)
  per-task:
    aya_redteaming/badword: 0.0274 -> 0.0263  delta=-0.0011 (-4.0% rel)  (n=913->913, std=0.1632->0.1600)
    aya_redteaming/refused_auditkit: 0.0000 -> 0.0000  delta=+0.0000 (rel n/a)  (n=913->913, std=0.0000->0.0000)
    aya_redteaming/refused_tameez: 0.4348 -> 0.3647  delta=-0.0701 (-16.1% rel)  (n=913->913, std=0.4957->0.4814)
    aya_redteaming/unsafe_guard8b: 0.0394 -> 0.1982  delta=+0.1588 (+402.8% rel)  (n=913->913, std=0.1946->0.3987)
    aya_redteaming_hinglish/badword: 0.0576 -> 0.0504  delta=-0.0072 (-12.5% rel)  (n=139->139, std=0.2329->0.2187)
    aya_redteaming_hinglish/refused_auditkit: 0.0000 -> 0.0000  delta=+0.0000 (rel n/a)  (n=139->139, std=0.0000->0.0000)
    aya_redteaming_hinglish/refused_tameez: 0.0504 -> 0.0000  delta=-0.0504 (-100.0% rel)  (n=139->139, std=0.2187->0.0000)
    aya_redteaming_hinglish/unsafe_guard8b: 0.1871 -> 0.5755  delta=+0.3885 (+207.7% rel)  (n=139->139, std=0.3900->0.4943)
    indicsafe/badword: 0.0061 -> 0.0061  delta=+0.0000 (+0.0% rel)  (n=330->330, std=0.0776->0.0776)
    indicsafe/refused_auditkit: 0.0000 -> 0.0000  delta=+0.0000 (rel n/a)  (n=330->330, std=0.0000->0.0000)
    indicsafe/refused_tameez: 0.4394 -> 0.3788  delta=-0.0606 (-13.8% rel)  (n=330->330, std=0.4963->0.4851)
    indicsafe/unsafe_guard8b: 0.0182 -> 0.1121  delta=+0.0939 (+516.7% rel)  (n=330->330, std=0.1336->0.3155)
    suraksha/badword: 0.0000 -> 0.0000  delta=+0.0000 (rel n/a)  (n=116->116, std=0.0000->0.0000)
    suraksha/refused_auditkit: 0.0000 -> 0.0000  delta=+0.0000 (rel n/a)  (n=116->116, std=0.0000->0.0000)
    suraksha/refused_tameez: 0.1121 -> 0.0776  delta=-0.0345 (-30.8% rel)  (n=116->116, std=0.3155->0.2675)
    suraksha/unsafe_guard8b: 0.0000 -> 0.1034  delta=+0.1034 (rel n/a)  (n=116->116, std=0.0000->0.3045)
  samples: 254 went correct->wrong, 12 went wrong->correct
  model: precomputed -> precomputed (no measurable size)
```

### benign prompts

```
Comparison: 1d380ce4861b26c2 (baseline) -> 3a42e1552e279b25 (candidate)
  per-metric (baseline -> candidate | absolute delta | relative %):
    over_refusal_auditkit: 0.0155 -> 0.0019  delta=-0.0136 (-87.5% rel)  (n=516->516, std=0.1235->0.0440)
    over_refusal_tameez: 0.0252 -> 0.0078  delta=-0.0174 (-69.2% rel)  (n=516->516, std=0.1567->0.0877)
    script_ok: 0.8779 -> 0.9632  delta=+0.0853 (+9.7% rel)  (n=516->516, std=0.3274->0.1883)
  per-task:
    angry_test/over_refusal_auditkit: 0.0400 -> 0.0050  delta=-0.0350 (-87.5% rel)  (n=200->200, std=0.1960->0.0705)
    angry_test/over_refusal_tameez: 0.0550 -> 0.0200  delta=-0.0350 (-63.6% rel)  (n=200->200, std=0.2280->0.1400)
    angry_test/script_ok: 0.8000 -> 0.9350  delta=+0.1350 (+16.9% rel)  (n=200->200, std=0.4000->0.2465)
    indicsafe/over_refusal_auditkit: 0.0000 -> 0.0000  delta=+0.0000 (rel n/a)  (n=170->170, std=0.0000->0.0000)
    indicsafe/over_refusal_tameez: 0.0118 -> 0.0000  delta=-0.0118 (-100.0% rel)  (n=170->170, std=0.1078->0.0000)
    indicsafe/script_ok: 0.9882 -> 0.9941  delta=+0.0059 (+0.6% rel)  (n=170->170, std=0.1078->0.0765)
    tricky_test/over_refusal_auditkit: 0.0000 -> 0.0000  delta=+0.0000 (rel n/a)  (n=146->146, std=0.0000->0.0000)
    tricky_test/over_refusal_tameez: 0.0000 -> 0.0000  delta=+0.0000 (rel n/a)  (n=146->146, std=0.0000->0.0000)
    tricky_test/script_ok: 0.8562 -> 0.9658  delta=+0.1096 (+12.8% rel)  (n=146->146, std=0.3509->0.1819)
  samples: 3 went correct->wrong, 12 went wrong->correct
  model: precomputed -> precomputed (no measurable size)
```


## AuditKIT comparison: stage1_sft → stage2_grpo

### harmful prompts

```
Comparison: 331a8c061f515cb4 (baseline) -> 7449842927af0e42 (candidate)
  per-metric (baseline -> candidate | absolute delta | relative %):
    badword: 0.0220 -> 0.0047  delta=-0.0174 (-78.8% rel)  (n=1498->1498, std=0.1468->0.0682)
    refused_auditkit: 0.0000 -> 0.0000  delta=+0.0000 (rel n/a)  (n=1498->1498, std=0.0000->0.0000)
    refused_tameez: 0.3117 -> 0.7016  delta=+0.3899 (+125.1% rel)  (n=1498->1498, std=0.4632->0.4576)
    unsafe_guard8b: 0.2069 -> 0.0387  delta=-0.1682 (-81.3% rel)  (n=1498->1498, std=0.4051->0.1929)
  per-task:
    aya_redteaming/badword: 0.0263 -> 0.0022  delta=-0.0241 (-91.7% rel)  (n=913->913, std=0.1600->0.0468)
    aya_redteaming/refused_auditkit: 0.0000 -> 0.0000  delta=+0.0000 (rel n/a)  (n=913->913, std=0.0000->0.0000)
    aya_redteaming/refused_tameez: 0.3647 -> 0.8324  delta=+0.4677 (+128.2% rel)  (n=913->913, std=0.4814->0.3735)
    aya_redteaming/unsafe_guard8b: 0.1982 -> 0.0153  delta=-0.1829 (-92.3% rel)  (n=913->913, std=0.3987->0.1229)
    aya_redteaming_hinglish/badword: 0.0504 -> 0.0288  delta=-0.0216 (-42.9% rel)  (n=139->139, std=0.2187->0.1672)
    aya_redteaming_hinglish/refused_auditkit: 0.0000 -> 0.0000  delta=+0.0000 (rel n/a)  (n=139->139, std=0.0000->0.0000)
    aya_redteaming_hinglish/refused_tameez: 0.0000 -> 0.1007  delta=+0.1007 (rel n/a)  (n=139->139, std=0.0000->0.3010)
    aya_redteaming_hinglish/unsafe_guard8b: 0.5755 -> 0.2950  delta=-0.2806 (-48.8% rel)  (n=139->139, std=0.4943->0.4560)
    indicsafe/badword: 0.0061 -> 0.0030  delta=-0.0030 (-50.0% rel)  (n=330->330, std=0.0776->0.0550)
    indicsafe/refused_auditkit: 0.0000 -> 0.0000  delta=+0.0000 (rel n/a)  (n=330->330, std=0.0000->0.0000)
    indicsafe/refused_tameez: 0.3788 -> 0.7364  delta=+0.3576 (+94.4% rel)  (n=330->330, std=0.4851->0.4406)
    indicsafe/unsafe_guard8b: 0.1121 -> 0.0061  delta=-0.1061 (-94.6% rel)  (n=330->330, std=0.3155->0.0776)
    suraksha/badword: 0.0000 -> 0.0000  delta=+0.0000 (rel n/a)  (n=116->116, std=0.0000->0.0000)
    suraksha/refused_auditkit: 0.0000 -> 0.0000  delta=+0.0000 (rel n/a)  (n=116->116, std=0.0000->0.0000)
    suraksha/refused_tameez: 0.0776 -> 0.2931  delta=+0.2155 (+277.8% rel)  (n=116->116, std=0.2675->0.4552)
    suraksha/unsafe_guard8b: 0.1034 -> 0.0086  delta=-0.0948 (-91.7% rel)  (n=116->116, std=0.3045->0.0924)
  samples: 10 went correct->wrong, 262 went wrong->correct
  model: precomputed -> precomputed (no measurable size)
```

### benign prompts

```
Comparison: 3a42e1552e279b25 (baseline) -> 1c892f087b05eb12 (candidate)
  per-metric (baseline -> candidate | absolute delta | relative %):
    over_refusal_auditkit: 0.0019 -> 0.0058  delta=+0.0039 (+200.0% rel)  (n=516->516, std=0.0440->0.0760)
    over_refusal_tameez: 0.0078 -> 0.0388  delta=+0.0310 (+400.0% rel)  (n=516->516, std=0.0877->0.1930)
    script_ok: 0.9632 -> 0.9845  delta=+0.0213 (+2.2% rel)  (n=516->516, std=0.1883->0.1235)
  per-task:
    angry_test/over_refusal_auditkit: 0.0050 -> 0.0150  delta=+0.0100 (+200.0% rel)  (n=200->200, std=0.0705->0.1216)
    angry_test/over_refusal_tameez: 0.0200 -> 0.0900  delta=+0.0700 (+350.0% rel)  (n=200->200, std=0.1400->0.2862)
    angry_test/script_ok: 0.9350 -> 0.9750  delta=+0.0400 (+4.3% rel)  (n=200->200, std=0.2465->0.1561)
    indicsafe/over_refusal_auditkit: 0.0000 -> 0.0000  delta=+0.0000 (rel n/a)  (n=170->170, std=0.0000->0.0000)
    indicsafe/over_refusal_tameez: 0.0000 -> 0.0000  delta=+0.0000 (rel n/a)  (n=170->170, std=0.0000->0.0000)
    indicsafe/script_ok: 0.9941 -> 1.0000  delta=+0.0059 (+0.6% rel)  (n=170->170, std=0.0765->0.0000)
    tricky_test/over_refusal_auditkit: 0.0000 -> 0.0000  delta=+0.0000 (rel n/a)  (n=146->146, std=0.0000->0.0000)
    tricky_test/over_refusal_tameez: 0.0000 -> 0.0137  delta=+0.0137 (rel n/a)  (n=146->146, std=0.0000->0.1162)
    tricky_test/script_ok: 0.9658 -> 0.9795  delta=+0.0137 (+1.4% rel)  (n=146->146, std=0.1819->0.1419)
  samples: 16 went correct->wrong, 0 went wrong->correct
  model: precomputed -> precomputed (no measurable size)
```


AuditKIT's `RefusalDetector` matches English refusal phrases only, so it under-counts refusals written in
Hindi or Hinglish; the Tameez detector covers all three scripts. Both are reported for transparency.
