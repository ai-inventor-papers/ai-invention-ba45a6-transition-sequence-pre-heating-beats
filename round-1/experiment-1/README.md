# Room-Transition Heating Prediction vs PreHeat

`demo/` — Self-contained demo (Colab-ready notebook or markdown). Run without setup.  
`src/` — Full source code, data, and outputs from the experiment execution.

**Type:** experiment  
**ID:** `art_kHXLs4RwHRgU`

## Layman Summary

We tested whether guessing which room someone will move to next helps heat homes more efficiently than an older method that just looks for similar past days.

## Full Summary

This experiment implements and compares two occupancy-forecasting methods for anticipatory residential heating, both driven through a shared RC-network thermal simulator with PreHeat's HeatRate-based anticipatory heating rule and a reactive-fallback comfort guarantee. The baseline is a faithful reproduction of PreHeat's per-room K=5 Hamming-distance nearest-neighbor predictor (Scott et al. 2011), which forecasts future occupancy by averaging the same time-of-day-and-daytype outcome across the 5 historical days most similar (by Hamming distance on occupancy-so-far) to today. The proposed method is a current-room-conditioned Markov room-transition predictor: an order-1 slot-to-slot transition matrix (weekday/weekend split) augmented with an order-2 last-two-slots context model that backs off to order-1 via additive smoothing when context evidence is sparse, propagated to longer horizons via closed-form transition-matrix powers (avoiding Monte Carlo rollout noise, per the plan's fallback item 3). No real multi-room occupancy dataset with room-level adjacency and timestamps was available from the DATASET dependency (the dependency directory was empty), so per the plan's fallback_plan item (1) a physics/behavior-calibrated synthetic occupant-trajectory generator was used instead: a semi-Markov random walk over each topology's room graph with log-normal dwell-time jitter around a weekday/weekend anchor schedule, a regularity parameter blending scripted vs. uniformly-random room choices, and 5-10% independent per-slot sensor noise, run across 3 distinct synthetic household topologies (3, 4, and 5 rooms; linear, star, and hallway-connected graphs; regularity 0.85/0.55/0.25) for 60 days each with a 60/40 weekday-stratified train/test split. Both predictors' outputs are converted to per-room ROC curves at lookaheads of 15/30/45/60 minutes via threshold sweeps, and threshold-matched at target false-positive rates of 0.05/0.10/0.20. At each matched FPR, all four heating policies (PreHeat baseline, transition-predictor, a fixed 06:00-22:00 schedule, and a no-lookahead purely-reactive policy) are run through an RC-network thermal simulator (dT/dt = (Q_heater - U*(T-T_out) - sum_j K_ij*(T-T_j)) / C, per-room C/U/Q_max sampled from published UK residential ranges, inter-room coupling K_ij=30 W/C, empirically-fit per-room HeatRate calibration run, and a reactive-fallback heating rule that heats whenever a room is truly occupied and below setpoint, exactly mirroring PreHeat's own comfort guarantee) to produce a gas-use proxy (Wh) and MissTime (minutes occupied and >1C below the 20C setpoint). Cross-topology bootstrap 95% confidence intervals (500 resamples) are computed on the percentage energy-savings of transition vs. PreHeat baseline at each matched FPR, together with a miss-time non-inferiority check. An i.i.d.-shuffled-trajectory self-test (the plan's disconfirmation probe) confirms the transition model's row-wise transition probabilities collapse toward near-uniform when no sequence structure exists. Results: the transition predictor dominates the PreHeat baseline's ROC AUC at every lookahead in all 3 topologies (e.g. 15-min AUC 0.860 vs 0.753, 0.813 vs 0.693, 0.824 vs 0.700), confirming the prediction-stage advantage cleanly. However, this ROC advantage does not reliably translate into thermal-simulation energy savings: per-topology savings at matched FPR range from -14.5% to +74.7% and are inconsistent in sign across topologies and FPR levels, so the cross-topology bootstrap 95% CI on mean savings includes zero at all three matched-FPR operating points (FPR=0.05: CI [-14.5%, 46.4%]; FPR=0.10: CI [-3.4%, 13.5%]; FPR=0.20: CI [-9.2%, 18.4%]), while MissTime is non-inferior (comfort is preserved, driven by the shared reactive-fallback rule which saturates comfort identically for both predictors regardless of prediction quality). The recorded verdict is DISCONFIRMED_TRANSLATION_STAGE: the prediction-quality gain is real and robust, but the reactive-fallback comfort guarantee and the discrete on/off heater dynamics wash out the anticipatory-heating benefit at the energy-use level, at least under this synthetic calibration. Full per-topology ROC curves, per-FPR-per-policy gas/miss-time tables, bootstrap CI breakdowns, and the explicit verdict are written to method_out.json (schema-validated against exp_gen_sol_out.json), with full/mini/preview variants generated. The script runs on 4 CPUs via ProcessPoolExecutor (one worker per topology, spawn context) in under 30 seconds, well within a synthetic-data time budget, and used $0 of OpenRouter spend (no LLM calls were needed for this fully self-contained simulation).

## Output Files

- `method.py`
- `full_method_out.json`
- `mini_method_out.json`
- `preview_method_out.json`

## Demo Files

- **method.py** — Research methodology implementation

---
*Generated by AI Inventor Pipeline*
