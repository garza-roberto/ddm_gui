# DDM Explorer (Garza et al. 2026)

An interactive browser-based GUI for building intuition about the **Drift-Diffusion Model (DDM)** — a canonical computational model of perceptual decision-making. Parameters are tuned in real time via sliders, and four visualisation panels update immediately to reflect the simulation outcome.

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Model](#model)
5. [Dependencies](#dependencies)
6. [How to Cite](#how-to-cite)
7. [AI Assistance](#ai-assistance)
8. [User Manual](#user-manual)
9. [License](#license)

---

## Features

### Parameter Controls
Six sliders (with synchronised numeric inputs and per-parameter reset buttons) expose the full model parameter space:

| Label | Model Variable | Default | Range | Description |
|-------|---------------|---------|-------|-------------|
| Diffusion | `noise_sigma` | 1.0 | 0.01 – 3.0 | Standard deviation of the Gaussian noise term |
| Drift | `scaling_factor` | 1.0 | 0.0 – 5.0 | Scales the evidence accumulation rate |
| Leak | `leak` | 0.5 | 0.0 – 2.0 | Leaky-integrator decay constant |
| Reset | `residual_after_bout` | 0.0 | 0.0 – 1.0 | Fraction of threshold retained after a decision |
| Delay | `inactive_time` | 0.1 | 0.0 – 1.0 | Refractory period (s) between consecutive decisions |
| Boundary | `threshold` | 1.0 | 0.1 – 3.0 | Decision threshold (±) |

### Visualisation Panels

1. **Psychometric curve** — proportion correct vs. coherence with SEM/SD error bars
2. **Coherence vs IBI** — mean inter-bout interval vs. coherence with SEM/SD error bars
3. **Chronometric curve** — accuracy binned by trial-time quartile per coherence level (speed-accuracy signatures)
4. **IBI distributions** — violin + box plot of inter-bout intervals, split by correct/incorrect decisions, per coherence level
5. **Decision-variable trajectory** — full x(t) trace for a single trial at the selected coherence; threshold bands shaded; crossing events colour-coded by decision direction; refractory epochs visible

### Additional Controls
- **Update mode** — *Manual* (click ▶ Run) or *Auto* (re-runs on every slider change)
- **Number of trials** — Monte Carlo trials per coherence level (default: 30)
- **Random seed** — toggle between truly random and reproducible simulations
- **Input law** — square-root, Fechner (log), or linear mapping from coherence to drift
- **Error bars** — switch between SEM and SD across all relevant plots
- **Trajectory displayed** — selects which coherence level is shown in the DV trajectory panel
- **Cite** — top-right button that opens the APA citation for the associated paper with a one-click copy button

---

## Installation

```bash
git clone https://github.com/your-username/ddm-explorer.git
cd ddm-explorer
pip install -r requirements.txt
python app.py
```

Then open [http://127.0.0.1:8050](http://127.0.0.1:8050) in your browser.

> **Note:** Python ≥ 3.9 is recommended. A conda environment is encouraged:
> ```bash
> conda create -n ddm python=3.11
> conda activate ddm
> pip install -r requirements.txt
> ```

---

## Project Structure

```
ddm-explorer/
├── app.py                  # Main Dash application
├── requirements.txt        # Python dependencies
├── assets/
│   └── slider_colors.css   # Per-parameter slider colour overrides
└── README.md
```

---

## Model

The simulated decision variable follows a discrete-time leaky accumulator with absorbing boundaries:

$$x_{t+1} = x_t + \left( \alpha \sqrt{c} - \lambda x_t \right) \Delta t + \mathcal{N}(0,\, \sigma^2 \Delta t)$$

where:
- $c$ — stimulus coherence (input signal strength)
- $\alpha$ — drift scaling factor (`scaling_factor`)
- $\lambda$ — leak (`leak`)
- $\sigma$ — diffusion coefficient (`noise_sigma`)

A decision (bout) is triggered when $|x_t| \geq \theta$ (`threshold`) and the time since the last decision exceeds the refractory period (`inactive_time`). After a decision, the accumulator resets to $\text{sign}(x_t) \cdot \theta \cdot r$ where $r$ is `residual_after_bout`.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `dash` | Web application framework |
| `dash-bootstrap-components` | Bootstrap layout and theme |
| `plotly` | Interactive figures |
| `numpy` | Numerical simulation |
| `scipy` | Statistics (SEM, KDE distributions) |
| `numba` | JIT-compiled simulation kernel |

---

## How to Cite

If you use DDM Explorer in your research, please cite the associated paper:

> Garza, R., El Hady, A., & Bahl, A. (2026). Developmental and genetic modulation of evidence integration dynamics in zebrafish sensorimotor decision-making. *bioRxiv*. https://doi.org/10.64898/2026.03.01.708829

**BibTeX:**
```bibtex
@article{garza2026developmental,
  title   = {Developmental and genetic modulation of evidence integration dynamics
             in zebrafish sensorimotor decision-making},
  author  = {Garza, Roberto and El Hady, Ahmed and Bahl, Armin},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.03.01.708829},
  url     = {https://doi.org/10.64898/2026.03.01.708829}
}
```

The in-app **Cite** button (top-right of the interface) generates and copies this citation automatically.

---

## AI Assistance

DDM Explorer was developed with the assistance of **Perplexity AI (Sonar model)**, which was used to generate, iterate, and debug the application code under the direction and review of the authors. All scientific content, model design, and research decisions were made by the authors.

---

## User Manual

This section describes every interactive element in the application and its effect on the model and visualisations.

---

### Layout Overview

The interface is divided into two main areas:

- **Left panel (sidebar)** — all controls for configuring the model and simulation
- **Right area** — five live-updating visualisation panels arranged in two rows

The sidebar is organised into five sections: **Parameters**, **Simulation**, **Input Law**, **Coherence Levels**, **Trial Structure**, and **Visualization**.

---

### Parameters Section

Each parameter is represented by a coloured row containing three elements: a **slider**, a **numeric input box** (directly editable), and a **↺ reset button** that restores the default value. The slider and input box are synchronised — moving one updates the other.

#### Diffusion (`noise_sigma`) — *cyan*

Controls the standard deviation of the Gaussian noise added at each time step:

$$\eta_t \sim \mathcal{N}(0,\, \sigma^2 \Delta t)$$

- **Low values** (e.g. 0.1): the decision variable integrates evidence cleanly, producing steep psychometric curves and short, consistent IBIs at high coherence.
- **High values** (e.g. 2.5): the accumulator is dominated by noise, flattening the psychometric curve and widening IBI distributions even at high coherence.
- **Default:** 1.0 &nbsp;|&nbsp; **Range:** 0.01 – 3.0

#### Drift (`scaling_factor`) — *pink*

Scales the rate at which coherence evidence is accumulated:

$$\text{drift} = \alpha \cdot f(c)$$

where $f(c)$ is the input law applied to coherence $c$.

- **High values** (e.g. 3.0): evidence accumulates rapidly, threshold is crossed sooner, IBIs are shorter and decisions more accurate.
- **Low values** (e.g. 0.2): the model integrates slowly; even high coherence may not consistently reach threshold, producing flat psychometric curves and long IBIs.
- **Default:** 1.0 &nbsp;|&nbsp; **Range:** 0.0 – 5.0

#### Leak (`leak`) — *orange*

Implements a leaky integrator that causes the decision variable to decay toward zero over time:

$$x_{t+1} = x_t + (\alpha f(c) - \lambda x_t)\Delta t + \eta_t$$

- **Zero leak** ($\lambda = 0$): perfect integration — evidence is accumulated without any decay (standard DDM).
- **High leak** (e.g. 1.5): the accumulator forgets recent evidence rapidly. Only sustained, high-coherence stimuli can push $x$ to threshold, producing long IBIs at low coherence and compressing the dynamic range of the psychometric curve.
- **Default:** 0.5 &nbsp;|&nbsp; **Range:** 0.0 – 2.0

#### Reset (`residual_after_bout`) — *green*

Controls how much of the threshold value is retained in the accumulator immediately after a decision:

$$x \leftarrow \text{sign}(x) \cdot \theta \cdot r$$

- **Reset = 0**: accumulator is fully reset to zero after each decision.
- **Reset = 1**: accumulator stays at the threshold — the model immediately re-crosses it on the next step if no refractory period is enforced, generating very high bout rates.
- **Intermediate values** (e.g. 0.5): the accumulator retains momentum in the same direction, shortening the time to the next same-direction decision.
- **Default:** 0.0 &nbsp;|&nbsp; **Range:** 0.0 – 1.0

#### Delay (`inactive_time`) — *red*

Sets the minimum time (in seconds) that must elapse between two consecutive decisions (refractory period). During this window the accumulator is frozen at its last value:

- **Short delay** (e.g. 0.05 s): decisions can occur in rapid succession; very high bout rates are possible with high Reset values.
- **Long delay** (e.g. 0.8 s): even if threshold is crossed early, the model must wait before registering another decision. This compresses IBI distributions from below and reduces total bout count.
- **Default:** 0.1 s &nbsp;|&nbsp; **Range:** 0.0 – 1.0 s

#### Boundary (`threshold`) — *white*

The absolute value of the decision threshold. A decision is triggered when $|x_t| \geq \theta$:

- **Low threshold** (e.g. 0.3): threshold is crossed frequently even under high noise; accuracy is poor and IBIs are short.
- **High threshold** (e.g. 2.5): the model requires strong, sustained evidence accumulation before committing — accuracy is high but decisions are slow (long IBIs).
- The threshold value is displayed as labelled dashed lines (±θ) on the **Trajectory** panel.
- **Default:** 1.0 &nbsp;|&nbsp; **Range:** 0.1 – 3.0

---

### Simulation Section

#### Update Mode

Determines when the simulation is re-run:

- **Manual** *(default)*: the simulation only runs when you click the **⟳ Run Simulation** button. Recommended for slow machines or large trial counts.
- **Auto**: the simulation re-runs automatically each time any slider or input changes. Useful for real-time exploration with small trial counts (≤ 30).

#### Number of Trials

Integer input specifying how many Monte Carlo trials are simulated per coherence level. Total simulations = *N trials × number of coherence levels*.

- **Low values** (10 – 20): fast but noisy — psychometric and chronometric curves will show high variance.
- **High values** (100 – 500): smooth curves with tight error bars, but slower to compute.
- **Default:** 30 &nbsp;|&nbsp; **Range:** 10 – 500 (step 10)

#### Random Seed

A toggle with two options:

- **No seed** *(default)*: every run uses a different random seed — results vary across runs, reflecting genuine stochastic variability.
- **Seed**: enables the integer input field to the right. Enter any non-negative integer. The simulation becomes fully reproducible: the same parameter set and seed will always produce identical results. Useful for comparisons between conditions.

*Example:* Set seed = 42, run with Diffusion = 0.5, then change Diffusion to 1.5 and run again. The noise sequences differ only in amplitude, isolating the effect of the parameter change from random variability.

#### ⟳ Run Simulation

Executes the batch simulation with the current parameters and updates all five plots. The status bar below the button shows:
- Update mode tag
- Number of trials and coherence levels
- Total bout count across all trials
- Trial duration
- Active seed (or "seed=random")

---

### Input Law Section

Controls the mapping from stimulus coherence $c$ to the drift input $f(c)$:

| Option | Formula | Effect |
|--------|---------|--------|
| **Square root √c** *(default)* | $f(c) = \sqrt{c}$ | Compresses high-coherence differences; matches Weber–Fechner-like scaling |
| **Fechner log(1+c)** | $f(c) = \ln(1 + c)$ | Even stronger compression at high coherence |
| **Linear c** | $f(c) = c$ | Equal spacing; psychometric curve spread is uniform across coherence levels |

The choice primarily affects the **steepness and shape** of the psychometric and chronometric curves. Square-root scaling is the default used in the associated paper.

---

### Coherence Levels Section

Defines the set of coherence values $c \in [0, 1]$ used as stimulus strengths. Each row shows a coherence value as a percentage and a delete button. The **+ Add level** button appends a new editable entry.

- Coherence 0% corresponds to a fully ambiguous stimulus — the model relies entirely on noise to reach threshold.
- Coherence 100% corresponds to the maximum available evidence.
- All plots will include one trace or data point per coherence level. Adding many levels increases simulation time linearly.

---

### Trial Structure Section

Defines the temporal structure of each simulated trial as a sequence of **segments**. Each segment has:

- **Duration (s)**: how long this segment lasts
- **Input scale**: multiplier applied to the coherence input during this segment (0 = no stimulus, 1 = full stimulus)

**Default structure:**

| Segment | Duration | Input Scale | Description |
|---------|----------|-------------|-------------|
| 1 | 5 s | 0.0 | Pre-stimulus baseline — accumulator evolves under noise only |
| 2 | 20 s | 1.0 | Stimulus on — drift active |
| 3 | 5 s | 0.0 | Post-stimulus — accumulator decays or drifts freely |

You can add segments with **+ Add segment** or delete individual rows. This allows simulation of more complex protocols such as stimulus ramps, gaps, or two-interval forced-choice designs.

---

### Visualization Section

#### Error Bars

Selects the error measure shown on the **Psychometric**, **Coherence vs IBI**, and **Chronometric** plots:

- **SEM** *(default)*: standard error of the mean — shrinks with more trials, appropriate for displaying uncertainty on the mean estimate.
- **SD**: standard deviation — reflects the spread of the distribution regardless of trial count. Use SD to assess trial-to-trial variability rather than estimation precision.

#### Trajectory Displayed (coherence)

Dropdown that selects which coherence level is rendered in the **Decision Variable Trajectory** panel. Only one trial (at the active seed) is shown at a time. Switching coherence updates the trajectory immediately without re-running the full batch.

---

### Visualisation Panels

All panels are interactive Plotly figures — you can zoom, pan, and hover over data points for exact values. Each panel has a camera icon in the top-right corner to download it as a PNG.

#### Psychometric Curve (top-left)

Plots **P(correct)** against coherence level (%). Error bars show SEM or SD across trials.

- The dashed horizontal line at 0.5 marks chance level.
- A steep sigmoid indicates the model reliably distinguishes high from low coherence. A flat curve indicates near-chance performance across all levels.
- **Tip:** increase Boundary or decrease Diffusion to steepen the curve.

#### Coherence vs IBI (top-centre)

Plots the **mean inter-bout interval** (in seconds) against coherence level (%). Error bars show SEM or SD.

- Shorter IBIs at high coherence indicate that strong evidence drives the accumulator to threshold rapidly.
- If IBIs do not decrease with coherence, the model is operating in a noise-dominated regime (try reducing Diffusion or increasing Drift).

#### Chronometric Curve (top-right, wider)

Plots **P(correct)** for each coherence level as a function of trial-time bin (quartile of total trial duration). Each line is one coherence level.

- Rising lines indicate that accuracy improves as more evidence is integrated — the hallmark of an accumulator model.
- Flat lines indicate that decisions are independent of the time elapsed (noise-dominated or very fast threshold crossing).

#### IBI Distributions (middle row)

Violin plots showing the full distribution of inter-bout intervals, split by **Correct** (upper row) and **Incorrect** (lower row) decisions, with one violin per coherence level.

- Wide, spread-out violins indicate high variability in decision timing.
- Bimodal shapes may emerge when Reset > 0, since post-decision residual momentum can create a short-IBI cluster.

#### Decision Variable Trajectory (bottom row)

Shows the full time course of the decision variable $x(t)$ for a single trial at the selected coherence level.

- The **dashed horizontal lines** (labelled ±θ) mark the decision boundaries. Their positions update automatically when Boundary changes.
- **Coloured circles** mark threshold-crossing events: teal circles indicate upward crossings (correct decisions when evidence is positive), yellow circles indicate downward crossings.
- **Shaded regions** mark segments during which the accumulator is frozen due to the refractory period (Delay).
- The grey segment bands in the background reflect the trial structure (pre-stimulus, stimulus-on, post-stimulus).

---

## License

MIT
