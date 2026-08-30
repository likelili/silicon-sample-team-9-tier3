# Silicon Sample Benchmark — method registration form

This registration documents Team 9's Tier 3 `secondary-1` submission. Items marked **★** are public; items marked **†** are available through escrow.

## 0 · Approach identity and output

- **0.1 Team ★:** Team 9 (ExploraTwin): Olivier Toubia, Tianyi Peng, George Gui, Yuchen Qiu, and Naveen Venkat. The organizers approved this five-person team.
- **0.2 Plain-language summary ★:** We used Twin-2K-500 digital twins to simulate responses in every benchmark condition. We calibrated the individual predictions using matched human and synthetic responses to previously administered questions through SYN-DIGITS (Fan et al. 2026), applied demographic cell-proportion weights, and estimated each treatment effect as the weighted intervention mean minus the weighted pooled-control mean.
- **0.3 Submission tier and approach family ★:** Tier 3; single-model, persona-conditioned survey simulation followed by deterministic calibration, demographic reweighting, and mean-difference treatment-effect estimation.
- **0.4 Pipeline diagram:** (1) We screened the Twin-2K respondents using their pre-study responses. (2) We simulated every eligible twin in all 17 conditions. (3) We retained the twins with complete responses in every condition and constructed the 13 benchmark outcomes from their survey answers. (4) We calibrated each condition–outcome prediction using matched human and synthetic Wave-4 responses. (5) We applied fixed demographic weights and estimated each treatment effect by subtracting the weighted pooled-control mean from the corresponding weighted intervention mean.
- **0.5 Coverage ★:** Complete coverage of all 16 interventions and 13 outcomes, yielding 208 unique, nonmissing treatment-effect predictions.

## A · Scope of LLM use

- **A.1 Purpose:** GPT-5.6 Luna generated the twins' pre-study responses, benchmark-study responses, and separate Wave-4 anchor responses. All eligibility checks, outcome construction, calibration, weighting, aggregation, and treatment-effect calculations were performed deterministically without an LLM.
- **A.2 Degree of automation ★:** The workflow was fully automated at prediction time. No simulated response, calibrated prediction, condition mean, or treatment-effect estimate was manually selected or edited.

## B · Model / system details

- **B.1 Model name:** OpenAI `gpt-5.6-luna`, using the exact provider model identifier configured by the platform.
- **B.2 Access and context mode:** Hosted API. Pre-study and Wave-4 calls used the synchronous Responses API; benchmark-study calls used `chat/completions` through the Batch API on August 26–28, 2026. Each condition was a fresh session.
- **B.3 Configuration:** Medium reasoning effort and one completion per request. The study batch requests set `max_completion_tokens = 8000`. Temperature, top-p, top-k, frequency and presence penalties, stop sequences, and provider generation seeds were not supplied. Survey-side randomization used global seed 42 and deterministic session seeds.
- **B.4 Customization:** No fine-tuning, retrieval-augmented generation, prompt optimization against benchmark outcomes, tool use, web search, or agentic model orchestration.
- **B.5 Persistent memory:** None. The model retained no state across conditions. When earlier answers were required by the survey flow, they were inserted explicitly into the next prompt.
- **B.6 Inference stack:** Not applicable; the model was accessed through a hosted API rather than served locally.
- **B.7 Ensembles:** None; all simulated responses came from one model and one completion per request.

## C · Prompts

- **C.1 Exact prompts:** The system and user messages were generated from the supplied QSF by the frozen SurveyTwin pipeline. The public repository includes the QSF and prompt-construction code, while the restricted escrow preserves the individual prediction matrices and supporting reproduction materials. Prompts were not revised in response to benchmark predictions.
- **C.2 System-wide instructions:** The model was instructed to answer as the supplied persona, follow the displayed survey instructions and response scales, and return one structured response for every requested answer unit.
- **C.3 Prompt-design rationale:** The prompts preserve the wording, stimuli, response options, order, and logic of the supplied instrument while requiring machine-readable output that can be aligned with Qualtrics variables and validated automatically.

## D · Persona / profile construction

- **D.1 Profile source:** The estimates were derived from Twin-2K-500, which contains more than 500 survey answers from a representative panel of 2,058 U.S. respondents (Toubia et al. 2025). Of these, 2,007 passed the prespecified pre-study eligibility checks. Every eligible twin was simulated in all 17 conditions.
- **D.2 Profile verbalization:** Pre-study eligibility responses were generated from the released full representation. Benchmark and Wave-4 simulations used the released paragraph-length summary representation. Neither representation was edited.
- **D.3 Assignment and weighting:** We retained the 1,921 twins with exactly complete sessions in every condition, so the same respondents underlie every contrast. We assigned one fixed weight to each twin using a 40-cell gender × age × race target distribution reconstructed by iterative proportional fitting of a Census-seeded table to the benchmark's released gender × age and gender × race margins. Each weight equals the cell's target proportion divided by its proportion in the clean pool and is reused across conditions and outcomes.

## E · Stimulus and survey administration

- **E.1 Stimulus presentation:** Intervention and control texts were taken verbatim from the QSF. Each session received exactly one intervention or one of the three texts pooled as the control condition. For the state-contingent “Extreme weather predictions” condition, the applicable follow-up text was selected after the model answered the state question.
- **E.2 Survey walk-through:** Sixteen conditions were administered in one model call. The state-contingent condition used two sequential calls, with the first-stage answer and prior survey context supplied to the second stage. QSF block order, outcome order, choice order, and condition-specific randomization were resolved with deterministic session seeds. Questions, response scales, and attention or comprehension items were displayed as specified by the QSF.
- **E.3 Response elicitation:** The model returned constrained, structured JSON containing the question identifier, answer value, and answer label for each requested answer unit. No token log-probabilities were used.

## F · Stochasticity and aggregation

- **F.1 Runs and seeds:** We generated one response per twin–condition combination. Survey randomization used global seed 42 and deterministic session-level seeds; the provider did not expose deterministic generation for this model. Calibration and treatment-effect calculations are exactly reproducible from the deposited matrices.
- **F.2 Aggregation rule:** For each intervention and outcome, we calculated the weighted mean across all 1,921 clean-pool twins and subtracted the weighted mean for the pooled control condition, using the same respondents and fixed demographic weights on both sides. Thus, `ATE = weighted intervention mean - weighted pooled-control mean`, the organizers' specified Tier 3 estimator.

## G · Validation and post-processing

- **G.1 Human validation:** None. Humans did not review, rate, select, or edit the simulated responses or submitted treatment effects.
- **G.2 Post-processing:** Returned responses were parsed by question identifier and converted according to each item's declared choice or numeric scale. We retained only sessions whose returned question set exactly matched the requested set and only twins with exact completion in all 17 conditions. The resulting effective sample is 1,921 per condition. Reconstruction reproduced all 117,000 Tier 1 outcome cells. The demographic weights range from 0.309 to 9.792, have an effective sample size of 1,442.7, and recover the 40 target-cell proportions to numerical precision. All 208 ATEs are finite and equal their source weighted-mean differences.
- **G.3 Calibration corrections:** We applied SYN-DIGITS (Fan et al. 2026) separately to each of the 221 condition–outcome columns. The calibration inputs were respondent-aligned human and GPT-5.6 Luna responses to 123 Twin-2K Wave-4 questions; the Wave-4 synthetic answers were generated separately and were not included in the benchmark personas. For each target, rank-5 hard-SVD imputation filled structurally missing anchor values separately in the human and synthetic matrices; each matrix was standardized using its own column statistics with `min_col_std = 1.0`. An elastic net with regularization multiplier `0.01` and L1 ratio `0.3` learned the relationship from synthetic anchors to the synthetic benchmark target, and that relationship was then applied to the human anchors. No human benchmark outcome was used. Following the published transfer rule, we retained the calibrated column when the synthetic-side training MSE was at most `0.15` and otherwise retained the raw digital-twin column; 68 targets were calibrated and 153 used the raw fallback. This specification was fixed without reference to human Silicon Sample outcomes.

## H · Learning and conditioning components

- **H.1 Fine-tuning:** None.
- **H.2 Calibration inputs:** Respondent- and item-aligned human and GPT-5.6 Luna responses to 123 Twin-2K Wave-4 anchors. Silicon targets were never used as donors for one another.

## I · Data inputs, blinding, and competing interests

- **I.1 Competing interests ★:** API costs were paid by Columbia Business School; no relationship with OpenAI beyond a paid API account.
- **I.2 External human data †:** Twin-2K human responses supplied calibration anchors and contain no benchmark outcomes.
- **I.3 Blinding ★:** No team member accessed human benchmark outcomes before the lock. Olivier Toubia signed the no-exposure declaration on August 18, 2026.
- **I.4 Contamination †:** OpenAI had not publicly disclosed the training-data cutoff for `gpt-5.6-luna`; benchmark human outcomes were unpublished before the lock.

## J · Design-space search

- **J.1 Search †:** Eight estimator families were diagnostically compared on a prespecified half of the Wave-4 anchors. Production used the published Twin-2K elastic-net specification and its fixed `tau = 0.15` transfer rule. Gate behavior was examined on the Wave-4 tuning fold and separate MegaStudy outcomes; the 61-anchor holdout was opened once to evaluate the ungated calibration specification. No Silicon human outcome was used for model selection, calibration, or gating.

## K · Reproducibility and frozen artifacts

- **K.1 Code and materials:** `code/calib/reproduce_submission.py` regenerates the 221 elastic-net calibrations from the public matrices and verifies the 208 submitted ATEs; `code/calib/poststratification.py` constructs and audits the fixed 40-cell reweighting factors. The original response-to-target reconstruction remains in `code/calib/production.py` and uses the escrowed run archive. The SYN-DIGITS calibration implementation is vendored under its MIT license.
- **K.2 Reproduction data †:** The individual raw and calibrated target matrices, aligned Wave-4 anchors, demographic target grid, respondent weights, fit diagnostics, weighting audits, and reproduction code are under restricted access at [10.5281/zenodo.22168937](https://doi.org/10.5281/zenodo.22168937). The underlying adopted provider responses used to construct these matrices are preserved in the Tier 2 escrow at [10.5281/zenodo.22168892](https://doi.org/10.5281/zenodo.22168892). The public `artifacts/` directory contains the corresponding derived matrices and audit reports needed to verify the submitted effects.
- **K.3 Resources:** The 221 elastic-net fits used one workstation and no additional API calls.

## L · Disclosure class

**Class B (escrowed).** Predictions, calibration code, derived matrices, and audits are public; a tier-specific reproduction archive is escrowed at [10.5281/zenodo.22168937](https://doi.org/10.5281/zenodo.22168937).

## References

Fan, Grace Jiarui, Chengpiao Huang, Tianyi Peng, Kaizheng Wang, and Yuhang Wu (2026), “SYN-DIGITS: A Synthetic Control Framework for Calibrated Digital Twin Simulation,” arXiv:2604.07513. [https://doi.org/10.48550/arXiv.2604.07513](https://doi.org/10.48550/arXiv.2604.07513).

Toubia, Olivier, George Z. Gui, Tianyi Peng, Daniel J. Merlau, Ang Li, and Haozhe Chen (2025), “Database Report: Twin-2K-500,” *Marketing Science*, 44 (6), 1446–1455. [https://doi.org/10.1287/mksc.2025.0262](https://doi.org/10.1287/mksc.2025.0262).
