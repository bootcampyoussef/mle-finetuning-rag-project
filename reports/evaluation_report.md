# Northstar: Baseline RAG vs. Hybrid RAG + LoRA

> **Decision: REVISE.** The adapter improves response-format compliance on this small
> held-out benchmark, but the hybrid is **not ready for autonomous support replies**:
> unsupported questions can still receive answers, and accepted answers can omit
> essential policy requirements. Treat these results as experimental evidence.

## 1. Experiment and reproducibility

- Evaluation set: **14** held-out cases; **10** supported, **4** unsupported.
- Comparison: same policy corpus, retriever, top-k, threshold, chat prompt,
  generation parameters and test questions. Only the LoRA adapter state differs.
- Base model: `HuggingFaceTB/SmolLM2-135M-Instruct`; revision: `not specified`.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`; revision: `not specified`.
- Vector search: Qdrant local index, cosine similarity, top-k = **3**, min score = **0.3**.
- Inference: device `mps`, greedy (`do_sample=False`), `MAX_NEW_TOKENS=80`.
- Prompt: training.make_prompt() for both systems; do_sample=False.
- Adapter: `artifacts/support-lora-adapter`; only training adapter weights were saved.
- Training data: `data/training/instruction_examples.jsonl`; **16** examples.
- Adapter configuration: rank **8**, alpha **16**, dropout **0.05**, targets **q_proj, v_proj**.
- Training: 8 epochs, learning rate 0.0002, max sequence length 256, batch size 1, seed 42.
- Parameters: **460800 trainable / 134975808 total**.
- Training loss per epoch: 1.7444, 1.5007, 1.2731, 1.1135, 1.0016, 0.8974, 0.7945, 0.6994.
- This was a **real local PEFT optimization loop**, not prompt-only fine-tuning.

## 2. Aggregate evaluation

| Metric | Baseline RAG | Hybrid RAG + LoRA |
|---|---:|---:|
| Answered/abstained decision accuracy | 21.4% | 64.3% |
| Required-term coverage on accepted answers | 10.0% | 50.0% |
| Citation/term faithfulness proxy | 10.0% | 40.0% |
| Response-style proxy | 0.0% | 70.0% |
| Unsupported abstentions | 2/4 | 2/4 |
| Mean generation time per case | 0.63 s | 0.64 s |

Retrieval is shared: **recall@3 = 100.0%**, **MRR@3 = 1.000**, retriever-only supported/unsupported accuracy = **85.7%**.
Retrieval recall and MRR use *supported cases*; overall decision accuracy uses all cases.

### Metric definitions and limitations

- **Decision accuracy:** supported cases count only if the generated answer passed the citation validator; unsupported cases count only if the system explicitly abstained. A rejected answer is **not** a correct abstention.
- **Required-term coverage:** exact whole-phrase matches; rejected outputs contribute zero to aggregate accepted coverage. It is a relevance proxy, not semantic correctness.
- **Citation-faithfulness proxy:** accepted answer, expected retrieved citation, 100% required-term coverage, and no matched forbidden phrase. It does **not** establish that every claim is entailed by evidence.
- **Style proxy:** accepted, concise, single paragraph, no greeting/list, at least one citation. It does not fully validate next-action quality.
- **Latency:** generation only; shared retrieval time, model cold start and training are excluded from aggregate generation time.
- **Forbidden-term detection:** exact phrase matching does not detect semantic paraphrases of prohibited claims.

## 3. Per-case comparison

| Case | Expected | Top retrieved score | Baseline | Hybrid | Accepted term coverage (base / hybrid) |
|---|---|---:|---|---|---:|
| `eval-refund-ticket` | ANSWER | 0.749 | rejected | rejected | 0% (rejected) / 0% (rejected) |
| `eval-duplicate-payment` | ANSWER | 0.507 | rejected | answered | 0% (rejected) / 100.0% |
| `eval-authenticator` | ANSWER | 0.599 | rejected | answered | 0% (rejected) / 0.0% |
| `eval-passport` | ANSWER | 0.606 | rejected | answered | 0% (rejected) / 50.0% |
| `eval-packed-address` | ANSWER | 0.816 | rejected | answered | 0% (rejected) / 50.0% |
| `eval-standard-shipping` | ANSWER | 0.644 | rejected | answered | 0% (rejected) / 100.0% |
| `eval-annual-refund` | ANSWER | 0.716 | rejected | answered | 0% (rejected) / 100.0% |
| `eval-pause` | ANSWER | 0.890 | rejected | answered | 0% (rejected) / 100.0% |
| `eval-outage-updates` | ANSWER | 0.717 | answered | rejected | 100.0% / 0% (rejected) |
| `eval-log-secrets` | ANSWER | 0.698 | rejected | rejected | 0% (rejected) / 0% (rejected) |
| `eval-bank-chargeback` | ABSTAIN | 0.652 | rejected | answered | n/a / n/a |
| `eval-shipping-compensation` | ABSTAIN | 0.550 | rejected | answered | n/a / n/a |
| `eval-cafeteria` | ABSTAIN | none >= threshold | abstained | abstained | n/a / n/a |
| `eval-bicycle` | ABSTAIN | none >= threshold | abstained | abstained | n/a / n/a |

## 4. Complete per-case evidence and model outputs

### eval-refund-ticket

**Question:** A digital workshop starts in 12 hours. Can the customer still receive a refund?

**Expected decision:** ANSWER; **expected source IDs:** refunds. **Required terms:** 24 hours, outside. **Forbidden phrases:** eligible inside the final 24 hours. 

**Retrieved policy passages:**

- `refunds` / chunk 1 / score 0.749 — Digital workshop tickets are refundable until 24 hours before the workshop starts. Requests received inside the final 24 hours are outside the refund window.
- `refunds` / chunk 0 / score 0.477 — Standard purchases are refundable within 30 calendar days when the customer provides the order ID. Approved refunds return to the original payment method.
- `refunds` / chunk 2 / score 0.407 — For a duplicate payment, collect the invoice number and both payment references. Finance reviews a complete duplicate-payment request within three business days.

**Baseline RAG:** `rejected` — missing source citation; terms 0.0%; cited `(none)`; forbidden matches `(none)`; generation 1.120 s.

```text
Yes, the customer can still receive a refund. The policy evidence states that the digital workshop starts in 12 hours, and the customer provides the order ID and payment references within 30 calendar days. This means that the refund window is 30 days after the workshop starts. Therefore, the customer can still receive a refund.
```

**Hybrid RAG + LoRA:** `rejected` — missing source citation; terms 100.0%; cited `(none)`; forbidden matches `(none)`; generation 1.013 s.

```text
The digital workshop tickets are refundable until 24 hours before the workshop starts, but the customer must provide the order ID and payment references. The standard purchases are outside the refund window, and the invoice number and payment references are required for a complete refund.
```

### eval-duplicate-payment

**Question:** What must support collect for a duplicate payment investigation?

**Expected decision:** ANSWER; **expected source IDs:** refunds. **Required terms:** invoice number, payment references. **Forbidden phrases:** (none). 

**Retrieved policy passages:**

- `refunds` / chunk 2 / score 0.507 — For a duplicate payment, collect the invoice number and both payment references. Finance reviews a complete duplicate-payment request within three business days.
- `refunds` / chunk 0 / score 0.361 — Standard purchases are refundable within 30 calendar days when the customer provides the order ID. Approved refunds return to the original payment method.
- `response-style` / chunk 2 / score 0.339 — Do not invent exceptions. If the retrieved evidence does not answer the question, say that the approved material is insufficient and escalate for manual review.

**Baseline RAG:** `rejected` — missing source citation; terms 100.0%; cited `(none)`; forbidden matches `(none)`; generation 1.257 s.

```text
The policy evidence states that collect the invoice number and both payment references. This evidence must support the following:

- Collection of the invoice number and payment references to ensure that the duplicate payment is properly processed.
- Collection of the invoice number and payment references to ensure that the duplicate payment is properly processed.
- Collection of the invoice number and payment references to ensure that the duplicate payment is properly
```

**Hybrid RAG + LoRA:** `answered` — citations refer to retrieved sources; terms 100.0%; cited `refunds`; forbidden matches `(none)`; generation 0.582 s.

```text
Collect the invoice number and both payment references. Finance reviews a complete duplicate-payment request within three business days. [refunds]
```

### eval-authenticator

**Question:** A customer lost their authenticator. What information and process should support use?

**Expected decision:** ANSWER; **expected source IDs:** account-security. **Required terms:** registered email, secure upload portal. **Forbidden phrases:** identity verification may use ordinary chat. 

**Retrieved policy passages:**

- `account-security` / chunk 1 / score 0.599 — For account recovery, ask for the registered email address and send a secure recovery link. If a customer lost access to an authenticator, identity verification must use the secure upload portal, never ordinary chat.
- `response-style` / chunk 0 / score 0.377 — Start with the direct answer. Follow with the next action the customer or support colleague should take.
- `account-security` / chunk 2 / score 0.374 — For a suspicious login, recommend resetting the password, signing out other sessions, and enabling multi-factor authentication.

**Baseline RAG:** `rejected` — missing source citation; terms 50.0%; cited `(none)`; forbidden matches `(none)`; generation 0.528 s.

```text
A customer lost their authenticator. The customer should use the secure upload portal, never ordinary chat, and enable multi-factor authentication to prevent unauthorized access.
```

**Hybrid RAG + LoRA:** `answered` — citations refer to retrieved sources; terms 0.0%; cited `account-security`; forbidden matches `(none)`; generation 0.535 s.

```text
Reset the password, sign out other sessions, and enable multi-factor authentication. [account-security]
```

### eval-passport

**Question:** Should a customer paste a passport scan into chat to recover an account?

**Expected decision:** ANSWER; **expected source IDs:** account-security. **Required terms:** never, secure upload portal. **Forbidden phrases:** passport scans are safe in chat. 

**Retrieved policy passages:**

- `account-security` / chunk 1 / score 0.606 — For account recovery, ask for the registered email address and send a secure recovery link. If a customer lost access to an authenticator, identity verification must use the secure upload portal, never ordinary chat.
- `incidents` / chunk 1 / score 0.410 — Diagnostic logs may include request IDs and timestamps, but must not include passwords, authentication tokens, one-time passcodes, or full payment-card details.
- `account-security` / chunk 0 / score 0.369 — Support must never ask for or accept passwords, one-time passcodes, recovery codes, or full card numbers.

**Baseline RAG:** `rejected` — missing source citation; terms 0.0%; cited `(none)`; forbidden matches `(none)`; generation 1.752 s.

```text
Based on the provided policy evidence, it is not possible for a customer to paste a passport scan into chat to recover an account. The policy states that the customer must not ask for or accept passwords, authentication tokens, one-time passcodes, or full payment-card details. This means that the customer must not use any of these methods to recover an account.
```

**Hybrid RAG + LoRA:** `answered` — citations refer to retrieved sources; terms 50.0%; cited `account-security, incidents`; forbidden matches `(none)`; generation 0.694 s.

```text
Password recovery is not necessary. [incidents] The secure upload portal is the only way to recover an account. [account-security]
```

### eval-packed-address

**Question:** Can support change a delivery address after the warehouse packed the order?

**Expected decision:** ANSWER; **expected source IDs:** shipping. **Required terms:** cannot, carrier. **Forbidden phrases:** support can change the address after packing. 

**Retrieved policy passages:**

- `shipping` / chunk 1 / score 0.816 — A delivery address can be changed only before the warehouse packs the order. After packing, support cannot change the address. After dispatch, the customer may contact the carrier using the tracking number to request an available delivery option.

**Baseline RAG:** `rejected` — missing source citation; terms 50.0%; cited `(none)`; forbidden matches `(none)`; generation 0.534 s.

```text
No, support cannot change a delivery address after the warehouse packed the order.
```

**Hybrid RAG + LoRA:** `answered` — citations refer to retrieved sources; terms 50.0%; cited `shipping`; forbidden matches `(none)`; generation 0.838 s.

```text
The policy does not allow for change of the address before the warehouse packs the order. After packing, support cannot change the address. [shipping]
```

### eval-standard-shipping

**Question:** What is the usual standard shipping time?

**Expected decision:** ANSWER; **expected source IDs:** shipping. **Required terms:** three to five business days. **Forbidden phrases:** (none). 

**Retrieved policy passages:**

- `shipping` / chunk 0 / score 0.644 — Standard shipping usually takes three to five business days. Express shipping usually takes one to two business days.

**Baseline RAG:** `rejected` — missing source citation; terms 100.0%; cited `(none)`; forbidden matches `(none)`; generation 0.232 s.

```text
The standard shipping time is three to five business days.
```

**Hybrid RAG + LoRA:** `answered` — citations refer to retrieved sources; terms 100.0%; cited `shipping`; forbidden matches `(none)`; generation 0.514 s.

```text
Standard shipping usually takes three to five business days. Express shipping usually takes one to two business days. [shipping]
```

### eval-annual-refund

**Question:** Can an annual subscription receive a partial refund after 45 days?

**Expected decision:** ANSWER; **expected source IDs:** subscriptions. **Required terms:** not, 30-day. **Forbidden phrases:** partial refund is available after 45 days. 

**Retrieved policy passages:**

- `subscriptions` / chunk 1 / score 0.716 — Annual subscriptions are not partially refunded after the 30-day refund window. Support may record cancellation so the plan does not renew.
- `refunds` / chunk 0 / score 0.580 — Standard purchases are refundable within 30 calendar days when the customer provides the order ID. Approved refunds return to the original payment method.
- `subscriptions` / chunk 0 / score 0.484 — Monthly subscriptions can be cancelled before the next renewal date. Access continues until the end of the paid billing period.

**Baseline RAG:** `rejected` — missing source citation; terms 0.0%; cited `(none)`; forbidden matches `(none)`; generation 0.357 s.

```text
The policy evidence states that annual subscriptions cannot receive a partial refund after 45 days.
```

**Hybrid RAG + LoRA:** `answered` — citations refer to retrieved sources; terms 100.0%; cited `subscriptions`; forbidden matches `(none)`; generation 0.792 s.

```text
Annual subscriptions do not receive a partial refund after the 30-day refund window. Support may record cancellation so the plan does not renew. [subscriptions]
```

### eval-pause

**Question:** How often and for how long can an active subscription be paused?

**Expected decision:** ANSWER; **expected source IDs:** subscriptions. **Required terms:** once per calendar quarter, 30 days. **Forbidden phrases:** (none). 

**Retrieved policy passages:**

- `subscriptions` / chunk 2 / score 0.890 — An active subscription may be paused once per calendar quarter for up to 30 days. The pause begins on the date selected by the customer.
- `subscriptions` / chunk 0 / score 0.667 — Monthly subscriptions can be cancelled before the next renewal date. Access continues until the end of the paid billing period.
- `subscriptions` / chunk 1 / score 0.510 — Annual subscriptions are not partially refunded after the 30-day refund window. Support may record cancellation so the plan does not renew.

**Baseline RAG:** `rejected` — missing source citation; terms 100.0%; cited `(none)`; forbidden matches `(none)`; generation 0.421 s.

```text
The policy evidence states that an active subscription can be paused once per calendar quarter for up to 30 days.
```

**Hybrid RAG + LoRA:** `answered` — citations refer to retrieved sources; terms 100.0%; cited `subscriptions`; forbidden matches `(none)`; generation 0.887 s.

```text
[subscriptions] An active subscription may be paused once per calendar quarter for up to 30 days. The pause begins on the date selected by the customer.
```

### eval-outage-updates

**Question:** How frequently should updates be published during a confirmed outage?

**Expected decision:** ANSWER; **expected source IDs:** incidents. **Required terms:** 30 minutes. **Forbidden phrases:** (none). 

**Retrieved policy passages:**

- `incidents` / chunk 0 / score 0.717 — During a confirmed service outage, publish status updates at least every 30 minutes until recovery. Direct customers to the public status page for the latest confirmed information.
- `subscriptions` / chunk 2 / score 0.374 — An active subscription may be paused once per calendar quarter for up to 30 days. The pause begins on the date selected by the customer.
- `subscriptions` / chunk 0 / score 0.337 — Monthly subscriptions can be cancelled before the next renewal date. Access continues until the end of the paid billing period.

**Baseline RAG:** `answered` — citations refer to retrieved sources; terms 100.0%; cited `incidents, subscriptions`; forbidden matches `(none)`; generation 1.235 s.

```text
Policy evidence:
[incidents] During a confirmed outage, publish status updates at least every 30 minutes until recovery. Direct customers to the public status page for the latest confirmed information.
[subscriptions] An active subscription may be paused once per calendar quarter for up to 30 days. The pause begins on the date selected by the customer.
[subscriptions]
```

**Hybrid RAG + LoRA:** `rejected` — missing source citation; terms 100.0%; cited `(none)`; forbidden matches `(none)`; generation 0.618 s.

```text
Publish status updates at least every 30 minutes until recovery. Direct customers to the public status page for the latest confirmed information.
```

### eval-log-secrets

**Question:** May diagnostic logs sent to support contain authentication tokens?

**Expected decision:** ANSWER; **expected source IDs:** incidents. **Required terms:** must not, authentication tokens. **Forbidden phrases:** authentication tokens may be included. 

**Retrieved policy passages:**

- `incidents` / chunk 1 / score 0.698 — Diagnostic logs may include request IDs and timestamps, but must not include passwords, authentication tokens, one-time passcodes, or full payment-card details.
- `account-security` / chunk 1 / score 0.342 — For account recovery, ask for the registered email address and send a secure recovery link. If a customer lost access to an authenticator, identity verification must use the secure upload portal, never ordinary chat.
- `account-security` / chunk 0 / score 0.319 — Support must never ask for or accept passwords, one-time passcodes, recovery codes, or full card numbers.

**Baseline RAG:** `rejected` — missing source citation; terms 50.0%; cited `(none)`; forbidden matches `(none)`; generation 0.262 s.

```text
Yes, diagnostic logs may contain authentication tokens.
```

**Hybrid RAG + LoRA:** `rejected` — missing source citation; terms 50.0%; cited `(none)`; forbidden matches `(none)`; generation 1.071 s.

```text
Yes, diagnostic logs may include authentication tokens. For account recovery, ask for the registered email address and send a secure recovery link. If a customer lost access to an authenticator, identity verification must use the secure upload portal, never ordinary chat.
```

### eval-bank-chargeback

**Question:** How many days will my external bank take to resolve a chargeback for a duplicate payment?

**Expected decision:** ABSTAIN; **expected source IDs:** (none). **Required terms:** (none). **Forbidden phrases:** (none). 

**Retrieved policy passages:**

- `refunds` / chunk 2 / score 0.652 — For a duplicate payment, collect the invoice number and both payment references. Finance reviews a complete duplicate-payment request within three business days.
- `refunds` / chunk 0 / score 0.598 — Standard purchases are refundable within 30 calendar days when the customer provides the order ID. Approved refunds return to the original payment method.
- `refunds` / chunk 1 / score 0.439 — Digital workshop tickets are refundable until 24 hours before the workshop starts. Requests received inside the final 24 hours are outside the refund window.

**Baseline RAG:** `rejected` — missing source citation; terms n/a; cited `(none)`; forbidden matches `(none)`; generation 0.446 s.

```text
The policy evidence states that the external bank will take 30 calendar days to resolve a chargeback for a duplicate payment.
```

**Hybrid RAG + LoRA:** `answered` — citations refer to retrieved sources; terms n/a; cited `refunds`; forbidden matches `(none)`; generation 0.715 s.

```text
Standard purchases are refundable within 30 calendar days when the customer provides the order ID. Approved refunds return to the original payment method. [refunds]
```

### eval-shipping-compensation

**Question:** What compensation amount is guaranteed if express shipping arrives late?

**Expected decision:** ABSTAIN; **expected source IDs:** (none). **Required terms:** (none). **Forbidden phrases:** (none). 

**Retrieved policy passages:**

- `shipping` / chunk 0 / score 0.550 — Standard shipping usually takes three to five business days. Express shipping usually takes one to two business days.
- `shipping` / chunk 1 / score 0.383 — A delivery address can be changed only before the warehouse packs the order. After packing, support cannot change the address. After dispatch, the customer may contact the carrier using the tracking number to request an available delivery option.
- `refunds` / chunk 0 / score 0.375 — Standard purchases are refundable within 30 calendar days when the customer provides the order ID. Approved refunds return to the original payment method.

**Baseline RAG:** `rejected` — missing source citation; terms n/a; cited `(none)`; forbidden matches `(none)`; generation 0.648 s.

```text
The policy evidence states that "Standard shipping usually takes three to five business days." So, if the customer provides the order ID, the policy evidence guarantees that the order will arrive within three business days.
```

**Hybrid RAG + LoRA:** `answered` — citations refer to retrieved sources; terms n/a; cited `shipping`; forbidden matches `(none)`; generation 0.705 s.

```text
Standard shipping usually takes three to five business days. Express shipping usually takes one to two business days. Express shipping is usually refundable. [shipping]
```

### eval-cafeteria

**Question:** Where can I find next week's cafeteria menu?

**Expected decision:** ABSTAIN; **expected source IDs:** (none). **Required terms:** (none). **Forbidden phrases:** (none). 

**Retrieved policy passages:**

- None above the configured similarity threshold.

**Baseline RAG:** `abstained` — no retrieved evidence; terms n/a; cited `(none)`; forbidden matches `(none)`; generation 0.000 s.

```text
(empty)
```

**Hybrid RAG + LoRA:** `abstained` — no retrieved evidence; terms n/a; cited `(none)`; forbidden matches `(none)`; generation 0.000 s.

```text
(empty)
```

### eval-bicycle

**Question:** Can I reserve a bicycle parking space near the office?

**Expected decision:** ABSTAIN; **expected source IDs:** (none). **Required terms:** (none). **Forbidden phrases:** (none). 

**Retrieved policy passages:**

- None above the configured similarity threshold.

**Baseline RAG:** `abstained` — no retrieved evidence; terms n/a; cited `(none)`; forbidden matches `(none)`; generation 0.000 s.

```text
(empty)
```

**Hybrid RAG + LoRA:** `abstained` — no retrieved evidence; terms n/a; cited `(none)`; forbidden matches `(none)`; generation 0.000 s.

```text
(empty)
```

## 5. Failure analysis

### A. Citation success can hide incomplete policy answers

Lost-authenticator case: baseline `rejected`, hybrid `answered`; hybrid required-term coverage 0.0%; top retrieval score 0.599.
A valid `[account-security]` citation does not prove that the answer contains the
required registered-email and secure-upload-portal instructions. These must be
checked against the retrieved passage, not merely by matching citation IDs.

### B. Retrieval relevance does not prove answerability

External bank chargeback: baseline `rejected`, hybrid `answered`; hybrid required-term coverage n/a; top retrieval score 0.652.
Late-shipping compensation: baseline `rejected`, hybrid `answered`; hybrid required-term coverage n/a; top retrieval score 0.550.
Both prompts are out of scope even when related policies rank highly.
A pure cosine threshold cannot distinguish a semantically related passage
from a passage that actually contains the answer. A separate evidence sufficiency
check is needed before accepting generated claims.

### C. LoRA is not a universal improvement

Incident updates: baseline `answered`, hybrid `rejected`; hybrid required-term coverage 100.0%; top retrieval score 0.717.
Workshop refund window: baseline `rejected`, hybrid `rejected`; hybrid required-term coverage 100.0%; top retrieval score 0.749.
Inspect exact model answers above to distinguish omitted citations, missing
required policy content, and refusals. A lower loss alone cannot explain or
justify the behavior of held-out responses.

## 6. Retrieval-threshold decision

The initial threshold **0.3** is retained as a documented
baseline, **not** a calibrated safe answerability cutoff. On this corpus,
the two difficult unsupported cases pass retrieval. For example:

- `eval-bank-chargeback`: expected abstain, top score 0.652.
- `eval-shipping-compensation`: expected abstain, top score 0.550.
- `eval-duplicate-payment`: expected answer, top score 0.507.

The overlap between supported and unsupported similarity scores means a single
higher threshold cannot separate all these cases without losing some valid queries.
Build a separate calibration set for score selection and add a source-grounded
answerability check. Do not optimize hyperparameters against this held-out set
and then claim unbiased final-test performance.

## 7. Cost and deployment trade-offs

- **Training:** on-device Apple MPS; no hosted training API or rented GPU was
  required. Training still consumes laptop time, electricity, and engineering effort.
  The training script did not record total wall-clock training time or energy use.
- **Weights:** only 460800 parameters were updated
  rather than all base-model parameters. Adapter maintenance adds revision control,
  compatibility testing, evaluation and deployment artifacts.
- **Inference:** observed means 0.63 s (baseline) versus
  0.64 s (hybrid) *for generation only*
  on the tested machine; 14 cases are too few for a reliable production latency claim.
- **Operational risk:** incorrect chargeback/compensation promises and missing
  account-recovery steps outweigh minor local inference overhead.

## 8. Decision and prioritized next experiment

**Recommendation: REVISE the hybrid design; do not deploy it for unsupervised customer support.**
Retain the adapter as an experiment because it helps follow the requested
response format, but treat claim grounding and abstention as independent controls.

1. **Highest priority:** build an evidence-sufficiency/claim-verification gate using
   retrieved policy text, with explicit refusal on unsupported questions.
   Evaluate against new supported and *near-domain unsupported* examples.
2. Use a stronger baseline with deterministic citation formatting or a controlled
   response template, so LoRA's value is not confused with a formatting fix.
3. Add validation split and multiple random seeds for LoRA; compare fewer epochs,
   longer sequences if warranted, and augment training with well-checked examples.
4. Review all outputs with a human policy rubric before increasing traffic.
   Require complete account-security steps and zero unsupported customer promises.
5. Reserve a **new, untouched final test set** after calibration and retraining.

**Key uncertainty to reduce next:** does LoRA still improve held-out policy
compliance when the baseline receives the same deterministic citation formatter
and both answer paths use a shared evidence-sufficiency gate?

## 9. Reproduction

```bash
uv sync
uv run python src/validate_data.py
uv run python src/retrieval.py
uv run python src/training.py
uv run python src/evaluate_generation.py
uv run python src/build_evaluation_report.py
uv run pytest -v
```

All aggregate metrics, case answers, citations and evidence above are read
directly from `reports/evaluation_results.json`; the report generator does not
rerun or change model outputs.
