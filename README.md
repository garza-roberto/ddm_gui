# DDM Explorer

An interactive browser-based GUI for building intuition about the **Drift-Diffusion Model (DDM)** — a canonical computational model of perceptual decision-making. Parameters are tuned in real time via sliders, and four visualisation panels update immediately to reflect the simulation outcome.

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

1. **Psychometric curve** — proportion correct vs. coherence with SEM error bars
2. **Chronometric curve** — accuracy binned by trial-time quartile per coherence level (speed-accuracy signatures)
3. **Inter-bout interval (IBI) distributions** — violin + box plot per coherence level
4. **Decision-variable trajectory** — full x(t) trace for a single seed-42 trial; threshold bands shaded; crossing events colour-coded by decision direction; refractory epochs visible

### Additional Controls
- **Update mode** — *Manual* (click ▶ Run) or *Auto* (re-runs on every slider change)
- **N trials** — number of Monte Carlo trials per coherence level
- **Trajectory coherence** — selects which coherence level the DV trajectory panel displays
- **Input law** — square-root or linear scaling of coherence into drift

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

## Project Structure

```
ddm-explorer/
├── app.py                  # Main Dash application
├── requirements.txt        # Python dependencies
├── assets/
│   └── slider_colors.css   # Per-parameter slider colour overrides
└── README.md
```

## Model

The simulated decision variable follows a discrete-time leaky accumulator with absorbing boundaries:

$$x_{t+1} = x_t + \left( \alpha \sqrt{c} - \lambda x_t \right) \Delta t + \mathcal{N}(0,\, \sigma^2 \Delta t)$$

where:
- $c$ — stimulus coherence (input signal strength)
- $\alpha$ — drift scaling factor (`scaling_factor`)
- $\lambda$ — leak (`leak`)
- $\sigma$ — diffusion coefficient (`noise_sigma`)

A decision (bout) is triggered when $|x_t| \geq \theta$ (`threshold`) and the time since the last decision exceeds the refractory period (`inactive_time`). After a decision, the accumulator resets to $\text{sign}(x_t) \cdot \theta \cdot r$ where $r$ is `residual_after_bout`.

## Dependencies

| Package | Purpose |
|---------|---------|
| `dash` | Web application framework |
| `dash-bootstrap-components` | Bootstrap layout and theme |
| `plotly` | Interactive figures |
| `numpy` | Numerical simulation |
| `scipy` | Statistics (SEM, distributions) |
| `pandas` | Data aggregation for plots |

## License

MIT
