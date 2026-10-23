# Reference Hybrid Evaluation

## Configuration

- Base model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Base revision: `12fd25f77366fa6b3b4b768ec3050bf629380bac`
- Adapter path: `artifacts/support-lora-adapter`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Vector database: `Qdrant` local mode with cosine similarity
- Retrieval: `top_k=3`, `min_score=0.30`
- Baseline generator: `HuggingFaceTB/SmolLM2-135M-Instruct with unchanged base model`
- Hybrid generator: `HuggingFaceTB/SmolLM2-135M-Instruct with LoRA adapter`
- Training data SHA-256: `ea802c67fd40dcef12fd1c4af38fdcfb6d3dd1eeee204ac83fee9af79a736996`
- LoRA targets: `q_proj, v_proj`
- LoRA rank / alpha / dropout: `8` / `16` / `0.05`
- Training: `8` epochs, learning rate `0.0002`, seed `42`
- Trainable parameters: `460800` of `134975808` (`0.341%`)
- Training loss: `1.872` to `0.747`

## Metric Rubric

- `recall@k`: supported cases whose expected source appears in the shared retrieval results.
- `MRR`: reciprocal rank of the first expected source for supported cases.
- `answer relevance`: proportion of labeled required terms present in the answer; unsupported cases score 1 only when the system abstains.
- `faithfulness`: valid expected citations with no labeled forbidden claim; unsupported cases must abstain.
- `style compliance`: concise first sentence, concise total response, and no vague hedge; rejected answers score 0, retrieval abstentions score 1.
- `decision accuracy`: answer supported cases and abstain on unsupported cases.

## Shared Retrieval Results

| Metric | Score |
| --- | ---: |
| Retrieval recall@3 | 1.000 |
| Mean reciprocal rank | 1.000 |

## System Comparison

| System | Relevance | Faithfulness | Style | Decision accuracy | Mean latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| base-only | 0.042 | 0.000 | 0.972 | 0.833 | 0.585 |
| rag-base | 0.167 | 0.167 | 0.167 | 0.167 | 0.626 |
| rag-lora | 0.667 | 0.750 | 0.833 | 0.833 | 0.551 |

## Hybrid Delta Against Baseline RAG

- Answer relevance delta: `+0.500`
- Faithfulness delta: `+0.583`
- Style-compliance delta: `+0.667`

## Per-Case Comparison

| ID | System | Status | Relevance | Faithful | Style | Citations |
| --- | --- | --- | ---: | --- | ---: | --- |
| eval-refund-ticket | base-only | answered | 0.00 | no | 1.00 | - |
| eval-duplicate-payment | base-only | answered | 0.00 | no | 1.00 | - |
| eval-authenticator | base-only | answered | 0.00 | no | 1.00 | - |
| eval-passport | base-only | answered | 0.00 | no | 1.00 | - |
| eval-packed-address | base-only | answered | 0.00 | no | 1.00 | - |
| eval-standard-shipping | base-only | answered | 0.00 | no | 1.00 | - |
| eval-annual-refund | base-only | answered | 0.50 | no | 1.00 | - |
| eval-pause | base-only | answered | 0.00 | no | 1.00 | - |
| eval-outage-updates | base-only | answered | 0.00 | no | 1.00 | - |
| eval-log-secrets | base-only | answered | 0.00 | no | 0.67 | - |
| eval-cafeteria | base-only | answered | 0.00 | no | 1.00 | - |
| eval-bicycle | base-only | answered | 0.00 | no | 1.00 | - |
| eval-refund-ticket | rag-base | invalid-citations | 0.00 | no | 0.00 | - |
| eval-duplicate-payment | rag-base | invalid-citations | 0.00 | no | 0.00 | - |
| eval-authenticator | rag-base | invalid-citations | 0.00 | no | 0.00 | - |
| eval-passport | rag-base | invalid-citations | 0.00 | no | 0.00 | - |
| eval-packed-address | rag-base | invalid-citations | 0.00 | no | 0.00 | - |
| eval-standard-shipping | rag-base | invalid-citations | 0.00 | no | 0.00 | - |
| eval-annual-refund | rag-base | invalid-citations | 0.00 | no | 0.00 | - |
| eval-pause | rag-base | invalid-citations | 0.00 | no | 0.00 | - |
| eval-outage-updates | rag-base | invalid-citations | 0.00 | no | 0.00 | - |
| eval-log-secrets | rag-base | invalid-citations | 0.00 | no | 0.00 | - |
| eval-cafeteria | rag-base | abstained | 1.00 | yes | 1.00 | - |
| eval-bicycle | rag-base | abstained | 1.00 | yes | 1.00 | - |
| eval-refund-ticket | rag-lora | answered | 0.50 | yes | 1.00 | refunds |
| eval-duplicate-payment | rag-lora | answered | 1.00 | yes | 1.00 | refunds |
| eval-authenticator | rag-lora | answered | 0.50 | yes | 1.00 | account-security |
| eval-passport | rag-lora | invalid-citations | 0.00 | no | 0.00 | - |
| eval-packed-address | rag-lora | invalid-citations | 0.00 | no | 0.00 | - |
| eval-standard-shipping | rag-lora | answered | 1.00 | yes | 1.00 | shipping |
| eval-annual-refund | rag-lora | answered | 1.00 | no | 1.00 | refunds |
| eval-pause | rag-lora | answered | 0.00 | yes | 1.00 | subscriptions |
| eval-outage-updates | rag-lora | answered | 1.00 | yes | 1.00 | incidents |
| eval-log-secrets | rag-lora | answered | 1.00 | yes | 1.00 | incidents |
| eval-cafeteria | rag-lora | abstained | 1.00 | yes | 1.00 | - |
| eval-bicycle | rag-lora | abstained | 1.00 | yes | 1.00 | - |

## Failure And Fragility Analysis

- `base-only` on `eval-refund-ticket`: status `answered`, relevance `0.00`, faithful `no`.
- `base-only` on `eval-duplicate-payment`: status `answered`, relevance `0.00`, faithful `no`.
- `base-only` on `eval-authenticator`: status `answered`, relevance `0.00`, faithful `no`.
- `rag-base` on `eval-refund-ticket`: status `invalid-citations`, relevance `0.00`, faithful `no`.
- `rag-base` on `eval-duplicate-payment`: status `invalid-citations`, relevance `0.00`, faithful `no`.
- `rag-base` on `eval-authenticator`: status `invalid-citations`, relevance `0.00`, faithful `no`.
- `rag-lora` on `eval-refund-ticket`: status `answered`, relevance `0.50`, faithful `yes`.
- `rag-lora` on `eval-authenticator`: status `answered`, relevance `0.50`, faithful `yes`.
- `rag-lora` on `eval-passport`: status `invalid-citations`, relevance `0.00`, faithful `no`.
- Shared retrieval recall is `1.000`; failures after successful retrieval are generation, citation, or answer-coverage failures.
- The adapter changes faithfulness by `+0.583` and decision accuracy from `0.167` to `0.833`. Remaining invalid citations still require rejection rather than automatic repair.

## Threshold Decision

The shared threshold is `0.30`. The weakest supported top score is `0.525` and the strongest unsupported top score is `0.208`. The same threshold is applied to baseline RAG and hybrid RAG so adapter quality cannot hide retrieval uncertainty.

## Recommendation

Keep the hybrid path as an experiment, not as an automatic replacement. It improves at least one held-out metric, but the failure rows and added adapter lifecycle still require review.

## Prioritized Next Experiments

1. Expand held-out paraphrases and policy-change cases before tuning the adapter again.
2. Compare a stronger small instruction model using the same retrieval and metric contracts.
3. Review invalid-citation cases and adjust training data only after identifying a repeatable failure pattern.
