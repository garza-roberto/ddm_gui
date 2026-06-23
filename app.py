"""
Drift-Diffusion Model (DDM) Interactive Explorer
=================================================
Visualizations:
  1. Psychometric curve        – accuracy vs coherence  (white line)
  2. Chronometric curve        – P(correct) vs time (green palette, light=high coh)
  3. IBI distributions         – KDE per coherence, correct (blue) / incorrect (yellow)
  4. DV trajectory             – x(t); lower threshold in IBI-incorrect yellow

Features:
  - Numba-JIT simulation core (compiled at startup via warm-up call)
  - Input-law selector: sqrt, Fechner log, linear
  - Trial time-structure editor (segments: duration + input-scale)
  - Per-plot lock/unlock: transparent pointer-events overlay, toggled client-side
  - Manual / Auto update mode (auto fires when slider is released)
  - SEM / SD toggle for error bars (instant, no re-simulation)
  - batch-store caches raw simulation results for error-metric redraws

Run: python app.py  →  open http://127.0.0.1:8050
"""

import json
import colorsys
import numpy as np
from numba import njit
from scipy.stats import gaussian_kde
import dash
from dash import dcc, html, Input, Output, State, ctx, ALL, MATCH, clientside_callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# Numba-JIT simulation kernel
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _simulate_kernel(signal_array, scaling_factor, threshold, noise_sigma,
                     leak, inactive_time, residual_after_bout, dt, noise):
    n_steps   = len(signal_array)
    max_bouts = n_steps
    rt_arr  = np.full(max_bouts, np.nan)
    dec_arr = np.full(max_bouts, np.nan)
    t_arr   = np.full(max_bouts, np.nan)
    dv      = np.zeros(n_steps)
    xs_old = 0.0; ts = 0.0; tlb = 0.0; bc = 0
    for t_i in range(n_steps):
        eff = signal_array[t_i]
        dx  = scaling_factor * eff - leak * xs_old
        xs  = xs_old + dx * dt + noise[t_i]
        ts += dt
        tsb = ts - tlb
        if tsb > inactive_time:
            if xs >= threshold:
                rt_arr[bc]=tsb; dec_arr[bc]=1.0; t_arr[bc]=ts
                tlb=ts; bc+=1; xs=threshold*residual_after_bout
            elif xs <= -threshold:
                rt_arr[bc]=tsb; dec_arr[bc]=0.0; t_arr[bc]=ts
                tlb=ts; bc+=1; xs=-threshold*residual_after_bout
        else:
            xs = xs_old
        dv[t_i]=xs; xs_old=xs
    return rt_arr, dec_arr, t_arr, dv, bc


def simulate_trial(scaling_factor, threshold, noise_sigma, leak,
                   inactive_time, residual_after_bout,
                   signal_array, dt=0.002, seed=None):
    rng   = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_sigma * np.sqrt(dt), len(signal_array))
    rt, dec, t, dv, nb = _simulate_kernel(
        signal_array, scaling_factor, threshold, noise_sigma,
        leak, inactive_time, abs(residual_after_bout), dt, noise,
    )
    return {
        "rt":        rt[:nb].tolist(),
        "decisions": dec[:nb].tolist(),
        "times":     t[:nb].tolist(),
        "dv":        dv.tolist(),
        "t_axis":    (np.arange(len(signal_array)) * dt).tolist(),
        "n_bouts":   nb,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Input-law transforms
# ─────────────────────────────────────────────────────────────────────────────

INPUT_LAWS = {
    "sqrt":   ("Square root  √c",   lambda c: np.sqrt(max(c, 0))),
    "log":    ("Fechner  log(1+c)",  lambda c: np.log1p(max(c, 0))),
    "linear": ("Linear  c",          lambda c: float(c)),
}

# ─────────────────────────────────────────────────────────────────────────────
# Trial time-structure helpers
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SEGMENTS = [
    {"duration": 5.0,  "scale": 0.0},
    {"duration": 20.0, "scale": 1.0},
    {"duration": 5.0,  "scale": 0.0},
]


def segments_to_signal(coherence, segments, input_law_fn, dt):
    total_dur = sum(s["duration"] for s in segments)
    n_steps   = int(round(total_dur / dt))
    signal    = np.zeros(n_steps)
    idx       = 0
    for seg in segments:
        seg_steps = int(round(seg["duration"] / dt))
        end_idx   = min(idx + seg_steps, n_steps)
        signal[idx:end_idx] = input_law_fn(coherence * float(seg["scale"]))
        idx = end_idx
        if idx >= n_steps:
            break
    return signal, total_dur


def run_batch(scaling_factor, threshold, noise_sigma, leak,
              inactive_time, residual_after_bout,
              coherences, segments, input_law_fn,
              n_trials=30, dt=0.002, base_seed=None):
    """If base_seed is None, trials are truly random (seed=None each).
    If base_seed is an int, trial seeds are base_seed + offset for reproducibility."""
    results = {}
    for coh in coherences:
        signal, _ = segments_to_signal(coh, segments, input_law_fn, dt)
        results[coh] = [
            simulate_trial(scaling_factor, threshold, noise_sigma, leak,
                           inactive_time, residual_after_bout,
                           signal, dt=dt,
                           seed=(base_seed + k) if base_seed is not None else None)
            for k in range(n_trials)
        ]
    return results


def batch_to_store(batch, coherences):
    """Serialize batch (dict keyed by float) to JSON-safe list of [coh, trials]."""
    return [[coh, batch[coh]] for coh in coherences]


def store_to_batch(store_data):
    """Deserialize back to {coh: [trials]}."""
    if not store_data:
        return {}, []
    batch = {}
    coherences = []
    for coh, trials in store_data:
        # trials[i] lists need numpy for math; keep as plain lists – callers cast
        batch[coh] = [
            {k: np.array(v) if isinstance(v, list) else v
             for k, v in trial.items()}
            for trial in trials
        ]
        coherences.append(coh)
    return batch, sorted(coherences)


# ─────────────────────────────────────────────────────────────────────────────
# Numba warm-up
# ─────────────────────────────────────────────────────────────────────────────

def _warmup():
    print("⏳  Compiling Numba kernel …", end=" ", flush=True)
    _sig, _ = segments_to_signal(0.5, DEFAULT_SEGMENTS, INPUT_LAWS["sqrt"][1], 0.01)
    simulate_trial(1.0, 1.0, 0.5, 0.1, 0.3, 0.0, _sig, dt=0.01, seed=0)
    print("done ✓", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG    = "#1c1b19"
SURFACE    = "#201f1d"
SURFACE2   = "#262523"
TEXT_COLOR = "#cdccca"
MUTED      = "#797876"
DIVIDER    = "#393836"
ACCENT     = "#4f98a3"
YELLOW     = "#e8af34"   # IBI incorrect colour — also lower threshold
RED        = "#dd6974"   # kept only for upper crossing markers (contrast with teal)
GRID_COLOR = "#2d2c2a"

IBI_COLOR_CORRECT   = "#5591c7"
IBI_COLOR_INCORRECT = YELLOW

PLOT_LAYOUT = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=SURFACE,
    font=dict(family="'JetBrains Mono','Consolas',monospace", color=TEXT_COLOR, size=11),
    margin=dict(l=50, r=20, t=36, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=DIVIDER, font=dict(size=10)),
)
AXIS_STYLE = dict(
    gridcolor=GRID_COLOR, linecolor=DIVIDER,
    tickcolor=DIVIDER, zerolinecolor=DIVIDER,
    title_font=dict(size=11),
)

DEFAULT_COHERENCES = [0.0, 0.25, 0.5, 1.0]
PLOT_IDS = ["psychometric", "coh-ibi", "chronometric", "ibi", "trajectory"]


def coh_label(c):
    return f"{c * 100:.0f}%"


def green_palette(n):
    """Return n greens: index 0 = lightest (100% coh), index n-1 = darkest (0% coh).
    HSL hue=130°, S=60%, L spans 0.72→0.28 so both ends are visibly green."""
    if n == 1:
        return ["#4caf50"]
    L_vals = np.linspace(0.72, 0.28, n)
    colors = []
    for L in L_vals:
        r, g, b = colorsys.hls_to_rgb(130 / 360, float(L), 0.60)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors


def _error_array(values, metric):
    """Compute SEM or SD for a 1-D array of per-trial values (floats)."""
    arr = np.asarray(values)
    if len(arr) == 0:
        return 0.0
    if metric == "sd":
        return float(np.std(arr))
    return float(np.std(arr) / np.sqrt(len(arr)))


# ─────────────────────────────────────────────────────────────────────────────
# Plot builders
# ─────────────────────────────────────────────────────────────────────────────

def build_psychometric(batch, coherences, err_metric="sem"):
    accs, errs = [], []
    for coh in coherences:
        dec = np.concatenate([np.asarray(t["decisions"]) for t in batch[coh]])
        if len(dec) == 0:
            accs.append(0.5); errs.append(0.0)
        else:
            accs.append(float(np.mean(dec)))
            errs.append(_error_array(dec, err_metric))
    x_pct = [c * 100 for c in coherences]
    err_label = "SD" if err_metric == "sd" else "SEM"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_pct, y=accs,
        error_y=dict(array=errs, color="rgba(255,255,255,0.55)",
                     thickness=1.5, width=5),
        mode="lines+markers",
        line=dict(color="#ffffff", width=2),
        marker=dict(size=8, color="#ffffff",
                    line=dict(color=DARK_BG, width=1.5)),
        name=f"Accuracy ± {err_label}",
    ))
    fig.add_hline(y=0.5, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text=f"Psychometric Curve  (± {err_label})",
                   font=dict(size=13), x=0.5),
        xaxis=dict(**AXIS_STYLE, title="Coherence (%)",
                   tickvals=x_pct, ticktext=[coh_label(c) for c in coherences]),
        yaxis=dict(**AXIS_STYLE, title="P(correct)", range=[0, 1.05]),
    )
    return fig



def build_coh_ibi(batch, coherences, err_metric="sem"):
    """Mean IBI (across all bouts, both decisions) vs coherence."""
    means, errs = [], []
    for coh in coherences:
        all_rt = np.concatenate([np.asarray(t["rt"]) for t in batch[coh]])
        all_rt = all_rt[np.isfinite(all_rt)]
        if len(all_rt) == 0:
            means.append(np.nan); errs.append(0.0)
        else:
            means.append(float(np.mean(all_rt)))
            errs.append(_error_array(all_rt, err_metric))
    x_pct = [c * 100 for c in coherences]
    err_label = "SD" if err_metric == "sd" else "SEM"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_pct, y=means,
        error_y=dict(array=errs, color="rgba(255,255,255,0.55)",
                     thickness=1.5, width=5),
        mode="lines+markers",
        line=dict(color="#ffffff", width=2),
        marker=dict(size=8, color="#ffffff", line=dict(color=DARK_BG, width=1.5)),
        name=f"Mean IBI ± {err_label}",
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text=f"Coherence vs IBI  (± {err_label})",
                   font=dict(size=13), x=0.5),
        xaxis=dict(**AXIS_STYLE, title="Coherence (%)",
                   tickvals=x_pct, ticktext=[coh_label(c) for c in coherences]),
        yaxis=dict(**AXIS_STYLE, title="Mean IBI (s)"),
    )
    return fig

def build_chronometric(batch, coherences, bin_width_s=1.0, err_metric="sem"):
    """Green palette: lightest = highest coherence, darkest = lowest.
    Error bars show SEM or SD across trials within each bin."""
    n_coh  = len(coherences)
    # sort coherences descending so greens[0] (lightest) = highest coh
    sorted_cohs = sorted(coherences, reverse=True)
    greens = green_palette(n_coh)
    color_map = {coh: greens[i] for i, coh in enumerate(sorted_cohs)}

    err_label = "SD" if err_metric == "sd" else "SEM"
    fig = go.Figure()
    for coh in sorted_cohs:
        color = color_map[coh]
        trials = batch[coh]
        all_times = np.concatenate([np.asarray(t["times"]) for t in trials])
        all_dec   = np.concatenate([np.asarray(t["decisions"]) for t in trials])
        if len(all_times) < 4:
            continue
        t_max = all_times.max()
        edges = np.arange(0.0, t_max + bin_width_s, bin_width_s)
        if len(edges) < 2:
            continue
        mids, accs, errs = [], [], []
        for j in range(len(edges) - 1):
            mask = (all_times >= edges[j]) & (all_times < edges[j + 1])
            if mask.sum() > 0:
                bin_dec = all_dec[mask]
                mids.append((edges[j] + edges[j + 1]) / 2)
                accs.append(float(np.mean(bin_dec)))
                errs.append(_error_array(bin_dec, err_metric))
        if not mids:
            continue
        # parse hex color for error bar rgba
        hx = color.lstrip("#")
        r_c, g_c, b_c = int(hx[0:2],16), int(hx[2:4],16), int(hx[4:6],16)
        fig.add_trace(go.Scatter(
            x=mids, y=accs,
            error_y=dict(array=errs,
                         color=f"rgba({r_c},{g_c},{b_c},0.55)",
                         thickness=1.2, width=3),
            mode="lines+markers",
            line=dict(color=color, width=1.8),
            marker=dict(size=4, color=color),
            name=coh_label(coh),
        ))
    fig.add_hline(y=0.5, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text=f"Accuracy vs Time in Trial  (± {err_label})",
                   font=dict(size=13), x=0.5),
        xaxis=dict(**AXIS_STYLE, title="Time in trial (s)"),
        yaxis=dict(**AXIS_STYLE, title="P(correct)", range=[0, 1.05]),
    )
    return fig


def _kde_trace(data, color, name, n_points=300, x_range=None):
    data = data[np.isfinite(data)]
    if len(data) < 3:
        return None
    kde = gaussian_kde(data, bw_method="silverman")
    if x_range is not None:
        lo, hi = x_range
    else:
        lo, hi = data.min(), data.max()
        pad = (hi - lo) * 0.15 + 1e-9
        lo = max(0, lo - pad); hi = hi + pad
    xs = np.linspace(lo, hi, n_points)
    ys = kde(xs)
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    return go.Scatter(
        x=xs, y=ys, mode="lines", fill="tozeroy",
        fillcolor=f"rgba({r},{g},{b},0.18)",
        line=dict(color=f"rgba({r},{g},{b},0.85)", width=1.8),
        name=name,
    )


def build_ibi_distributions(batch, coherences):
    n_coh = len(coherences)
    if n_coh == 0:
        return go.Figure()
    all_correct   = [np.concatenate([np.asarray(t["rt"])[np.asarray(t["decisions"]) == 1]
                                     for t in batch[coh]])
                     for coh in coherences]
    all_incorrect = [np.concatenate([np.asarray(t["rt"])[np.asarray(t["decisions"]) == 0]
                                     for t in batch[coh]])
                     for coh in coherences]
    combined = np.concatenate(all_correct + all_incorrect)
    valid = combined[np.isfinite(combined)]
    if len(valid) >= 2:
        pad   = (valid.max() - valid.min()) * 0.08 + 1e-6
        x_min = max(0.0, valid.min() - pad)
        x_max = valid.max() + pad
    else:
        x_min, x_max = 0.0, 1.0

    kde_traces = {}; y_max_global = 0.0
    for col_i, coh in enumerate(coherences):
        for row_i, (data, color, sfx) in enumerate([
            (all_correct[col_i],   IBI_COLOR_CORRECT,   "correct"),
            (all_incorrect[col_i], IBI_COLOR_INCORRECT, "incorrect"),
        ], start=1):
            tr = _kde_trace(data, color, f"{coh_label(coh)} {sfx}", x_range=(x_min, x_max))
            kde_traces[(row_i, col_i)] = tr
            if tr is not None:
                y_max_global = max(y_max_global, float(np.max(tr.y)))
    y_max_global = y_max_global * 1.12 if y_max_global > 0 else 1.0

    fig = make_subplots(
        rows=2, cols=n_coh,
        shared_xaxes=True, shared_yaxes=True,
        row_titles=["Correct", "Incorrect"],
        column_titles=[coh_label(c) for c in coherences],
        vertical_spacing=0.14, horizontal_spacing=0.06,
    )
    for col_i in range(n_coh):
        for row_i in range(1, 3):
            tr = kde_traces[(row_i, col_i)]
            if tr is not None:
                fig.add_trace(tr, row=row_i, col=col_i + 1)
            else:
                fig.add_annotation(text="no data", xref="paper", yref="paper",
                                   showarrow=False, font=dict(color=MUTED, size=10),
                                   row=row_i, col=col_i + 1)
    ax = {k: v for k, v in AXIS_STYLE.items() if k != "title_font"}
    fig.update_xaxes(range=[x_min, x_max], title_text="IBI (s)", title_font=dict(size=10),
                     **ax, row=2)
    fig.update_xaxes(range=[x_min, x_max], showticklabels=False, **ax, row=1)
    fig.update_yaxes(range=[0, y_max_global], title_text="Density", title_font=dict(size=10),
                     **ax, col=1)
    fig.update_yaxes(range=[0, y_max_global], **ax)
    ibi_layout = {k: v for k, v in PLOT_LAYOUT.items() if k != "margin"}
    fig.update_layout(**ibi_layout,
                      title=dict(text="IBI Distributions  ·  Correct (blue) vs Incorrect (yellow)",
                                 font=dict(size=13), x=0.5),
                      showlegend=False, margin=dict(l=55, r=20, t=60, b=50))
    return fig


def build_trajectory(scaling_factor, threshold, noise_sigma, leak,
                     inactive_time, residual_after_bout,
                     coherence, segments, input_law_fn, dt=0.002, seed=None):
    signal, total_dur = segments_to_signal(coherence, segments, input_law_fn, dt)
    trial = simulate_trial(scaling_factor, threshold, noise_sigma, leak,
                           inactive_time, residual_after_bout, signal, dt=dt, seed=seed)
    t   = np.asarray(trial["t_axis"])
    dv  = np.asarray(trial["dv"])
    decisions = np.asarray(trial["decisions"])
    times_arr = np.asarray(trial["times"])

    fig = go.Figure()
    t_cursor = 0.0
    for seg in segments:
        t_end = t_cursor + seg["duration"]
        if seg["scale"] > 0:
            fig.add_vrect(x0=t_cursor, x1=t_end,
                          fillcolor="rgba(79,152,163,0.05)", line_width=0)
        t_cursor = t_end

    # Upper threshold band (teal)
    fig.add_hrect(y0=threshold,         y1=threshold * 1.18,
                  fillcolor="rgba(79,152,163,0.07)", line_width=0)
    # Lower threshold band (yellow — same as IBI incorrect)
    yh = int(YELLOW[1:3], 16); yg = int(YELLOW[3:5], 16); yb_c = int(YELLOW[5:7], 16)
    fig.add_hrect(y0=-threshold * 1.18, y1=-threshold,
                  fillcolor=f"rgba({yh},{yg},{yb_c},0.07)", line_width=0)

    thr_label = f"{threshold:.2f}".rstrip("0").rstrip(".")
    fig.add_hline(y= threshold, line=dict(color=ACCENT, width=1.5, dash="dash"),
                  annotation_text=f"+{thr_label}", annotation_position="right",
                  annotation_font=dict(color=ACCENT, size=11))
    fig.add_hline(y=-threshold, line=dict(color=YELLOW, width=1.5, dash="dash"),
                  annotation_text=f"−{thr_label}", annotation_position="right",
                  annotation_font=dict(color=YELLOW, size=11))
    fig.add_hline(y=0, line=dict(color=MUTED, width=0.8, dash="dot"))
    fig.add_trace(go.Scatter(x=t, y=dv, mode="lines",
                             line=dict(color=TEXT_COLOR, width=1.0), name="DV"))

    if len(times_arr) > 0:
        cross_y = [threshold if d == 1 else -threshold for d in decisions]
        colors  = [ACCENT    if d == 1 else YELLOW      for d in decisions]
        fig.add_trace(go.Scatter(
            x=times_arr, y=cross_y, mode="markers",
            marker=dict(size=9, color=colors, symbol="circle",
                        line=dict(color=DARK_BG, width=1.5)),
            name="Crossing",
        ))
        for ct, cd in zip(times_arr, decisions):
            fig.add_vline(x=ct,
                          line=dict(color=ACCENT if cd == 1 else YELLOW,
                                    width=0.6, dash="dot"),
                          opacity=0.35)

    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text=(f"DV Trajectory  ·  {coh_label(coherence)}  ·  "
                         f"{trial['n_bouts']} bouts  ·  {total_dur:.0f} s"),
                   font=dict(size=13), x=0.5),
        xaxis=dict(**AXIS_STYLE, title="Time (s)"),
        yaxis=dict(**AXIS_STYLE, title="x(t)"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Dash app
# ─────────────────────────────────────────────────────────────────────────────

import os as _os
app = dash.Dash(
    __name__,
    assets_folder=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "assets"),
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500"
        "&family=Inter:wght@300;400;500;600&display=swap",
    ],
    title="DDM Explorer",
)

DEFAULTS = dict(
    noise_sigma=1.0,
    scaling_factor=1.0,
    leak=0.5,
    residual_after_bout=0.0,
    inactive_time=0.1,
    threshold=1.0,
    traj_coh=0.5,
    n_trials=30,
    input_law="sqrt",
)

# Order: diffusion, drift, leak, reset, delay, boundary
# pid↔label: noise_sigma=Diffusion, scaling_factor=Drift, leak=Leak,
#             residual_after_bout=Reset, inactive_time=Delay, threshold=Boundary
# Colors (matplotlib exact): deepskyblue, deeppink, orange, springgreen, tomato, white
PARAM_SPECS = [
    ("noise_sigma",         "Diffusion",  0.0,  3.0,  0.01),
    ("scaling_factor",      "Drift",     -3.0,  3.0,  0.01),
    ("leak",                "Leak",      -3.0,  3.0,  0.01),
    ("residual_after_bout", "Reset",      0.0,  1.0,  0.01),
    ("inactive_time",       "Delay",      0.0,  1.0,  0.01),
    ("threshold",           "Boundary",   0.0,  3.0,  0.05),
]

PARAM_COLORS = {
    "noise_sigma":         "#00bfff",  # deepskyblue
    "scaling_factor":      "#ff1493",  # deeppink
    "leak":                "#ffa500",  # orange
    "residual_after_bout": "#00ff7f",  # springgreen
    "inactive_time":       "#ff6347",  # tomato
    "threshold":           "#ffffff",  # white
}


def make_slider_row(pid, label, mn, mx, step, default):
    color = PARAM_COLORS.get(pid, "#cdccca")
    return dbc.Row([
        dbc.Col(html.Label(label,
                           className="param-label",
                           style={"color": color, "fontWeight": "500"}), width=3),
        dbc.Col(dcc.Slider(
            id=f"slider-{pid}", min=mn, max=mx, step=step, value=default,
            marks=None,
            tooltip={"placement": "bottom", "always_visible": False},
            className=f"param-slider param-slider-{pid}",
            updatemode="mouseup",
        ), width=6),
        dbc.Col(dbc.Input(
            id=f"val-{pid}", type="number", value=default,
            min=mn, max=mx, step=step, size="sm",
            className="val-input", debounce=True,
            style={"color": color, "borderColor": f"{color}66"},
        ), width=2),
        dbc.Col(dbc.Button(
            "↺", id=f"reset-{pid}", size="sm", color="secondary", outline=True,
            className="reset-btn", title=f"Reset ({default})",
            style={"borderColor": f"{color}66", "color": color},
        ), width=1),
    ], className="param-row g-2")


def plot_panel(graph_id, height="340px"):
    is_traj = (graph_id == "trajectory")
    cfg = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "select2d", "lasso2d", "autoScale2d",
            "hoverClosestCartesian", "hoverCompareCartesian",
            "toggleSpikelines", "toImage",
        ],
        **({"scrollZoom": True} if is_traj else {}),
    }
    return html.Div([
        html.Div(
            id={"type": "plot-overlay", "plot": graph_id},
            className="plot-overlay locked",
            title="Click to unlock plot interaction",
        ),
        dcc.Graph(
            id=f"fig-{graph_id}",
            config=cfg,
            style={"height": height},
        ),
    ], className="plot-wrapper", id=f"wrap-{graph_id}")


# ── Coherence editor ──────────────────────────────────────────────────────────

def _coh_rows(coherences):
    rows = []
    for i, c in enumerate(coherences):
        rows.append(dbc.Row([
            dbc.Col(dbc.Input(
                id={"type": "coh-input", "index": i},
                type="number", value=round(c, 6),
                min=0.0, max=1.0, step=0.001, size="sm",
                className="val-input", debounce=True,
            ), width=7),
            dbc.Col(html.Span(coh_label(c),
                              id={"type": "coh-pct-label", "index": i},
                              className="coh-pct-label"), width=3),
            dbc.Col(dbc.Button(
                "×", id={"type": "remove-coh-btn", "index": i},
                size="sm", color="danger", outline=True, className="reset-btn",
                disabled=(len(coherences) <= 1),
            ), width=2),
        ], className="g-1 mb-1"))
    return rows


def coherence_editor():
    return [
        dbc.Row([dbc.Col(html.Label("Coherence levels", className="param-label"))],
                className="mb-1"),
        html.Div(id="coh-rows-container", children=_coh_rows(DEFAULT_COHERENCES)),
        dbc.Row([
            dbc.Col(dbc.Button("+ Add level", id="add-coh-btn", size="sm",
                               color="secondary", outline=True,
                               className="add-coh-btn mt-1 w-100"), width=6),
            dbc.Col(html.Small("(0–1 decimal)", className="param-label",
                               style={"lineHeight": "28px"}), width=6),
        ], className="g-1 mt-1"),
    ]


# ── Trial time-structure editor ───────────────────────────────────────────────

def _seg_rows(segments):
    rows = []
    for i, seg in enumerate(segments):
        rows.append(dbc.Row([
            dbc.Col(html.Small(f"#{i+1}", className="param-label",
                               style={"lineHeight": "28px", "textAlign": "center"}), width=1),
            dbc.Col(dbc.Input(
                id={"type": "seg-dur",   "index": i},
                type="number", value=round(seg["duration"], 2),
                min=0.1, max=300.0, step=0.5, size="sm",
                className="val-input", debounce=True, placeholder="dur (s)",
            ), width=4),
            dbc.Col(dbc.Input(
                id={"type": "seg-scale", "index": i},
                type="number", value=round(seg["scale"], 3),
                min=0.0, max=10.0, step=0.1, size="sm",
                className="val-input", debounce=True, placeholder="scale",
            ), width=4),
            dbc.Col(dbc.Button(
                "×", id={"type": "remove-seg-btn", "index": i},
                size="sm", color="danger", outline=True, className="reset-btn",
                disabled=(len(segments) <= 1),
            ), width=2),
        ], className="g-1 mb-1"))
    return rows


def segment_editor():
    return [
        dbc.Row([dbc.Col(html.Label("Time structure", className="param-label"))],
                className="mb-1"),
        dbc.Row([
            dbc.Col(html.Small("#",           className="param-label",
                               style={"textAlign": "center"}), width=1),
            dbc.Col(html.Small("Dur (s)",     className="param-label"), width=4),
            dbc.Col(html.Small("Input scale", className="param-label"), width=4),
        ], className="g-1 mb-1"),
        html.Div(id="seg-rows-container", children=_seg_rows(DEFAULT_SEGMENTS)),
        dbc.Row([
            dbc.Col(dbc.Button("+ Add segment", id="add-seg-btn", size="sm",
                               color="secondary", outline=True,
                               className="add-coh-btn mt-1 w-100"), width=7),
            dbc.Col(html.Div(id="seg-total-label", className="coh-pct-label",
                             style={"lineHeight": "26px"}), width=5),
        ], className="g-1 mt-1"),
    ]


# ── Layout ────────────────────────────────────────────────────────────────────

sidebar = dbc.Col([
    html.Div([
        html.H6("PARAMETERS", className="section-title"),
        *[make_slider_row(pid, lbl, mn, mx, step, DEFAULTS[pid])
          for pid, lbl, mn, mx, step in PARAM_SPECS],

        html.Hr(className="divider"),
        html.H6("SIMULATION", className="section-title"),
        dbc.Row([
            dbc.Col(html.Label("Update mode", className="param-label"), width=5),
            dbc.Col(dbc.RadioItems(
                id="update-mode",
                options=[{"label": "Manual", "value": "manual"},
                         {"label": "Auto",   "value": "auto"}],
                value="manual", inline=True,
                className="update-mode-radio",
            ), width=7),
        ], className="param-row g-2"),
        dbc.Row([
            dbc.Col(html.Label("Number of trials", className="param-label"), width=6),
            dbc.Col(dbc.Input(id="n-trials", type="number", value=DEFAULTS["n_trials"],
                              min=10, max=500, step=10, size="sm",
                              className="val-input", debounce=True), width=4),
        ], className="param-row g-2"),
        dbc.Row([
            dbc.Col(dbc.RadioItems(
                id="seed-mode",
                options=[{"label": "No seed", "value": "none"},
                         {"label": "Seed",    "value": "fixed"}],
                value="none", inline=True,
                className="update-mode-radio",
            ), width=6),
            dbc.Col(dbc.Input(id="rng-seed", type="number", value=None,
                              min=0, max=2**31-1, step=1, size="sm",
                              placeholder="integer…",
                              className="val-input", debounce=True,
                              disabled=True), width=5),
        ], className="param-row g-2 align-items-center"),
        html.Hr(className="divider"),
        html.H6("INPUT LAW", className="section-title"),
        dbc.Row([dbc.Col(dcc.Dropdown(
            id="input-law",
            options=[{"label": v[0], "value": k} for k, v in INPUT_LAWS.items()],
            value=DEFAULTS["input_law"], clearable=False, className="coh-dropdown",
        ), width=12)], className="param-row g-2"),

        html.Hr(className="divider"),
        html.H6("COHERENCE LEVELS", className="section-title"),
        *coherence_editor(),
        dcc.Store(id="coh-store", data=DEFAULT_COHERENCES),

        html.Hr(className="divider"),
        html.H6("TRIAL STRUCTURE", className="section-title"),
        *segment_editor(),
        dcc.Store(id="seg-store", data=DEFAULT_SEGMENTS),

        html.Hr(className="divider"),
        html.H6("VISUALIZATION", className="section-title"),
        dbc.Row([
            dbc.Col(html.Label("Error bars", className="param-label"), width=4),
            dbc.Col(dbc.RadioItems(
                id="err-metric",
                options=[{"label": "SEM", "value": "sem"},
                         {"label": "SD",  "value": "sd"}],
                value="sem", inline=True,
                className="update-mode-radio",
            ), width=8),
        ], className="param-row g-2"),
        dbc.Row([
            dbc.Col(html.Label("Trajectory displayed (coherence)", className="param-label"), width=7),
            dbc.Col(dcc.Dropdown(
                id="traj-coh",
                options=[{"label": coh_label(c), "value": c} for c in DEFAULT_COHERENCES],
                value=DEFAULTS["traj_coh"], clearable=False, className="coh-dropdown",
            ), width=5),
        ], className="param-row g-2 align-items-center"),

        html.Hr(className="divider"),
        dbc.Button("⟳  Run Simulation", id="run-btn", color="primary",
                   className="run-btn w-100", n_clicks=0),
        html.Div(id="status-msg", className="status-msg"),
    ], className="sidebar-inner"),
], width=3, className="sidebar-col")


plots_area = dbc.Col([
    dcc.Loading(
        id="plots-loading",
        type="circle",
        color="#4f98a3",
        overlay_style={"visibility": "visible", "opacity": 0.4,
                       "backgroundColor": "transparent"},
        children=[
            dbc.Row([
                dbc.Col(plot_panel("psychometric"), width=3),
                dbc.Col(plot_panel("coh-ibi"),      width=3),
                dbc.Col(plot_panel("chronometric"), width=6),
            ], className="plot-row"),
            dbc.Row([
                dbc.Col(plot_panel("ibi", height="380px"), width=12),
            ], className="plot-row"),
            dbc.Row([
                dbc.Col(plot_panel("trajectory", height="340px"), width=12),
            ], className="plot-row"),
        ],
    ),
], width=9, className="plots-col")


LOGO_SVG = (
    '<svg viewBox="0 0 40 28" width="40" height="28" '
    'xmlns="http://www.w3.org/2000/svg" aria-label="DDM Explorer logo">'
    f'<line x1="0" y1="4"  x2="40" y2="4"  stroke="{ACCENT}" stroke-width="2"/>'
    f'<line x1="0" y1="24" x2="40" y2="24" stroke="{YELLOW}"  stroke-width="2"/>'
    f'<polyline points="0,14 5,13 8,16 13,8 17,19 21,6 25,20 30,12 35,15 40,14" '
    f'stroke="{TEXT_COLOR}" stroke-width="1.5" fill="none"/>'
    f'<circle cx="13" cy="8"  r="2.5" fill="{ACCENT}"/>'
    f'<circle cx="21" cy="6"  r="2.5" fill="{ACCENT}"/>'
    "</svg>"
)

app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Markdown(LOGO_SVG, dangerously_allow_html=True, className="logo-svg"),
            html.H1("DDM Explorer", className="app-title"),
        ], className="header-brand"),
        html.Span("Drift-Diffusion Model · Interactive Parameter Explorer",
                  className="header-subtitle"),
    ], className="app-header"),
    dbc.Container(
        dbc.Row([sidebar, plots_area], className="main-row"),
        fluid=True, className="main-container",
    ),
    # Per-param stores (unused now but kept for future use)
    *[dcc.Store(id=f"store-{pid}", data=DEFAULTS[pid]) for pid, *_ in PARAM_SPECS],
    # Cached simulation results — allows error-metric toggle without re-simulation
    dcc.Store(id="batch-store", data=None),
    dcc.Store(id="coh-order-store", data=DEFAULT_COHERENCES),
    dcc.Store(id="seg-dur-store",   data=sum(s["duration"] for s in DEFAULT_SEGMENTS)),
], className="app-root")


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

# ── Slider ↔ input sync ───────────────────────────────────────────────────────
for pid, _lbl, mn, mx, step in PARAM_SPECS:
    @app.callback(
        Output(f"slider-{pid}", "value"),
        Output(f"val-{pid}",    "value"),
        Input(f"slider-{pid}",  "value"),
        Input(f"val-{pid}",     "value"),
        Input(f"reset-{pid}",   "n_clicks"),
        prevent_initial_call=True,
    )
    def sync_param(sv, iv, _nc,
                   _pid=pid, _mn=mn, _mx=mx, _def=DEFAULTS[pid]):
        t = ctx.triggered_id
        if t == f"reset-{_pid}":  return _def, _def
        if t == f"slider-{_pid}": return sv, round(sv, 4)
        if t == f"val-{_pid}":
            v = min(_mx, max(_mn, float(iv or _mn)))
            return v, round(v, 4)
        return sv, iv


# Lock overlay: single MATCH-based server callback.
# clientside_callback with same JS string across a loop causes a hash-collision
# bug in Dash's JS runtime. MATCH pattern avoids all duplicate registrations.
@app.callback(
    Output({"type": "plot-overlay", "plot": MATCH}, "className"),
    Output({"type": "plot-overlay", "plot": MATCH}, "title"),
    Input({"type": "plot-overlay",  "plot": MATCH}, "n_clicks"),
    State({"type": "plot-overlay",  "plot": MATCH}, "className"),
    prevent_initial_call=True,
)
def toggle_lock(_n, current_class):
    if not current_class:
        current_class = "plot-overlay locked"
    locked = "locked" in current_class
    new_class = "plot-overlay unlocked" if locked else "plot-overlay locked"
    tip = "Click to lock plot" if locked else "Click to unlock plot interaction"
    return new_class, tip



# ── Coherence store ───────────────────────────────────────────────────────────
@app.callback(
    Output("coh-store",          "data"),
    Output("coh-rows-container", "children"),
    Output("traj-coh",           "options"),
    Output("traj-coh",           "value"),
    Input("add-coh-btn",                             "n_clicks"),
    Input({"type": "remove-coh-btn", "index": ALL}, "n_clicks"),
    Input({"type": "coh-input",      "index": ALL}, "value"),
    State("coh-store", "data"),
    State("traj-coh",  "value"),
    prevent_initial_call=True,
)
def update_coherences(_, remove_clicks, input_values, current_cohs, traj_val):
    triggered = ctx.triggered_id
    cohs = list(current_cohs)
    if triggered == "add-coh-btn":
        new_c = round((cohs[-1] + cohs[-2]) / 2, 3) if len(cohs) >= 2 \
                else round(min(cohs[-1] + 0.1, 1.0), 3) if cohs else 0.5
        cohs.append(new_c)
    elif isinstance(triggered, dict) and triggered.get("type") == "remove-coh-btn":
        idx = triggered["index"]
        if len(cohs) > 1:
            cohs.pop(idx)
    elif isinstance(triggered, dict) and triggered.get("type") == "coh-input":
        for i, v in enumerate(input_values):
            if v is not None and i < len(cohs):
                cohs[i] = round(float(v), 6)
    cohs = sorted(set(round(c, 6) for c in cohs if c is not None))
    if not cohs:
        cohs = [0.0]
    opts = [{"label": coh_label(c), "value": c} for c in cohs]
    new_traj = traj_val if traj_val in cohs else cohs[-1]
    return cohs, _coh_rows(cohs), opts, new_traj


# ── Segment store ─────────────────────────────────────────────────────────────
@app.callback(
    Output("seg-store",          "data"),
    Output("seg-rows-container", "children"),
    Output("seg-total-label",    "children"),
    Input("add-seg-btn",                             "n_clicks"),
    Input({"type": "remove-seg-btn", "index": ALL}, "n_clicks"),
    Input({"type": "seg-dur",        "index": ALL}, "value"),
    Input({"type": "seg-scale",      "index": ALL}, "value"),
    State("seg-store", "data"),
    prevent_initial_call=True,
)
def update_segments(_, remove_clicks, dur_values, scale_values, current_segs):
    triggered = ctx.triggered_id
    segs = [dict(s) for s in current_segs]
    if triggered == "add-seg-btn":
        segs.append({"duration": 5.0, "scale": 1.0})
    elif isinstance(triggered, dict) and triggered.get("type") == "remove-seg-btn":
        idx = triggered["index"]
        if len(segs) > 1:
            segs.pop(idx)
    elif isinstance(triggered, dict) and triggered.get("type") in ("seg-dur", "seg-scale"):
        for i in range(min(len(segs), len(dur_values))):
            if dur_values[i] is not None:
                segs[i]["duration"] = max(0.1, float(dur_values[i]))
            if i < len(scale_values) and scale_values[i] is not None:
                segs[i]["scale"] = float(scale_values[i])
    total = sum(s["duration"] for s in segs)
    return segs, _seg_rows(segs), f"total: {total:.1f} s"


# ── Main simulation callback → writes batch-store ─────────────────────────────
@app.callback(
    Output("rng-seed", "disabled"),
    Input("seed-mode", "value"),
)
def toggle_seed_input(mode):
    return mode != "fixed"


@app.callback(
    Output("batch-store",     "data"),
    Output("coh-order-store", "data"),
    Output("seg-dur-store",   "data"),
    Output("fig-ibi",         "figure"),
    Output("fig-trajectory",  "figure"),
    Output("status-msg",      "children"),
    Input("run-btn",   "n_clicks"),
    *[Input(f"slider-{pid}", "value") for pid, *_ in PARAM_SPECS],
    State("update-mode", "value"),
    State("n-trials",    "value"),
    State("seed-mode",   "value"),
    State("rng-seed",    "value"),
    State("traj-coh",    "value"),
    State("coh-store",   "data"),
    State("seg-store",   "data"),
    State("input-law",   "value"),
    *[State(f"slider-{pid}", "value") for pid, *_ in PARAM_SPECS],
    prevent_initial_call=False,
)
def run_simulation(run_clicks,
                   ns_in, sf_in, lk_in, ra_in, it_in, th_in,
                   update_mode, n_trials, seed_mode, rng_seed, traj_coh, coherences, segments, input_law_key,
                   ns_st, sf_st, lk_st, ra_st, it_st, th_st):
    # Input/State order matches PARAM_SPECS:
    #   noise_sigma, scaling_factor, leak, residual_after_bout, inactive_time, threshold
    from dash.exceptions import PreventUpdate
    triggered = ctx.triggered_id
    slider_ids = {f"slider-{pid}" for pid, *_ in PARAM_SPECS}
    if triggered in slider_ids and update_mode != "auto":
        raise PreventUpdate

    def _f(v, d): return float(v) if v is not None else float(d)
    noise_sigma         = _f(ns_st, DEFAULTS["noise_sigma"])
    scaling_factor      = _f(sf_st, DEFAULTS["scaling_factor"])
    leak                = _f(lk_st, DEFAULTS["leak"])
    residual_after_bout = _f(ra_st, DEFAULTS["residual_after_bout"])
    inactive_time       = _f(it_st, DEFAULTS["inactive_time"])
    threshold           = _f(th_st, DEFAULTS["threshold"])
    n_trials       = int(n_trials or DEFAULTS["n_trials"])
    base_seed      = int(rng_seed) if (seed_mode == "fixed" and rng_seed is not None) else None
    coherences     = sorted(coherences or DEFAULT_COHERENCES)
    segments       = segments or DEFAULT_SEGMENTS
    input_law_key  = input_law_key or DEFAULTS["input_law"]
    input_law_fn   = INPUT_LAWS[input_law_key][1]
    traj_coh       = _f(traj_coh, coherences[-1])
    if traj_coh not in coherences:
        traj_coh = coherences[-1]

    total_dur = sum(s["duration"] for s in segments)

    batch = run_batch(
        scaling_factor, threshold, noise_sigma, leak,
        inactive_time, residual_after_bout,
        coherences, segments, input_law_fn,
        n_trials=n_trials, dt=0.002, base_seed=base_seed,
    )
    fig_coh_ibi = build_coh_ibi(batch, coherences)
    fig_ibi  = build_ibi_distributions(batch, coherences)
    fig_traj = build_trajectory(
        scaling_factor, threshold, noise_sigma, leak,
        inactive_time, residual_after_bout,
        traj_coh, segments, input_law_fn, dt=0.002,
        seed=base_seed,
    )
    total_bouts = sum(sum(t["n_bouts"] for t in batch[c]) for c in coherences)
    mode_tag = "⚡ auto" if update_mode == "auto" else "⟳ manual"
    seed_tag = f"seed={base_seed}" if base_seed is not None else "seed=random"
    msg = (f"[{mode_tag}]  {n_trials} trials × {len(coherences)} levels  ·  "
           f"{total_bouts} bouts  ·  {total_dur:.0f} s/trial  ·  {seed_tag}")

    store_data = batch_to_store(batch, coherences)
    return store_data, coherences, total_dur, fig_ibi, fig_traj, msg


# ── Error-metric callback → redraws psychometric + chronometric only ──────────
# Triggered by: err-metric toggle OR batch-store update (new simulation).
# Never calls run_batch → instant redraw.
@app.callback(
    Output("fig-psychometric", "figure"),
    Output("fig-coh-ibi",      "figure"),
    Output("fig-chronometric", "figure"),
    Input("err-metric",     "value"),
    Input("batch-store",    "data"),
    State("coh-order-store","data"),
    State("seg-dur-store",  "data"),
    prevent_initial_call=False,
)
def redraw_err_plots(err_metric, store_data, coherences, total_dur):
    from dash.exceptions import PreventUpdate
    if not store_data:
        raise PreventUpdate
    batch, coherences = store_to_batch(store_data)
    coherences = sorted(coherences)
    total_dur  = float(total_dur or 30.0)
    bin_w      = max(1.0, total_dur / 60)
    err = err_metric or "sem"
    fig_psy    = build_psychometric(batch, coherences, err_metric=err)
    fig_cohibi = build_coh_ibi(batch, coherences, err_metric=err)
    fig_chro   = build_chronometric(batch, coherences, bin_width_s=bin_w, err_metric=err)
    return fig_psy, fig_cohibi, fig_chro


# ─────────────────────────────────────────────────────────────────────────────
# Inline CSS
# ─────────────────────────────────────────────────────────────────────────────

app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
    :root {
        --bg: #171614; --surface: #1c1b19; --surface2: #201f1d;
        --surface3: #262523; --border: #393836; --divider: #2d2c2a;
        --text: #cdccca; --muted: #797876; --faint: #5a5957;
        --accent: #4f98a3; --accent-hover: #227f8b; --yellow: #e8af34;
        --radius: 6px;
        --font-body: 'Inter', system-ui, sans-serif;
        --font-mono: 'JetBrains Mono', 'Consolas', monospace;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { background: var(--bg); color: var(--text); font-family: var(--font-body); font-size: 13px; min-height: 100vh; }
    .app-root { min-height: 100vh; display: flex; flex-direction: column; background: var(--bg); }

    .app-header { display: flex; align-items: center; justify-content: space-between; padding: 10px 20px; background: var(--surface); border-bottom: 1px solid var(--border); flex-shrink: 0; }
    .header-brand { display: flex; align-items: center; gap: 12px; }
    .logo-svg p { margin: 0; line-height: 0; }
    .logo-svg svg { width: 40px; height: 28px; display: block; }
    .app-title { font-family: var(--font-mono); font-size: 16px; font-weight: 500; color: var(--text); letter-spacing: -0.2px; }
    .header-subtitle { font-size: 11px; color: var(--muted); font-family: var(--font-mono); }

    .main-container { flex: 1; padding: 0 !important; }
    .main-row { margin: 0 !important; }
    .sidebar-col { background: var(--surface); border-right: 1px solid var(--border); padding: 0 !important; overflow-y: auto; min-height: calc(100vh - 50px); }
    .sidebar-inner { padding: 16px 14px; }
    .plots-col { padding: 0 !important; background: var(--bg); overflow-y: auto; }

    .section-title { font-family: var(--font-mono); font-size: 10px; font-weight: 500; color: var(--faint); letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 10px; margin-top: 4px; }
    .param-row { margin-bottom: 10px !important; align-items: center; }
    .param-label { font-size: 12px; color: var(--muted); line-height: 1.3; cursor: default; }

    .val-input input, .val-input { background: var(--surface3) !important; border: 1px solid var(--border) !important; color: var(--text) !important; font-family: var(--font-mono) !important; font-size: 12px !important; text-align: right; border-radius: var(--radius) !important; padding: 3px 6px !important; height: 28px !important; }
    .val-input input:focus { border-color: var(--accent) !important; outline: none !important; box-shadow: 0 0 0 2px rgba(79,152,163,0.2) !important; }

    .reset-btn { width: 28px !important; height: 28px !important; padding: 0 !important; font-size: 14px !important; border-color: var(--border) !important; color: var(--muted) !important; line-height: 1 !important; }
    .reset-btn:hover { border-color: var(--yellow) !important; color: var(--yellow) !important; }

    .add-coh-btn { font-size: 11px !important; height: 26px !important; border-color: var(--border) !important; color: var(--muted) !important; }
    .add-coh-btn:hover { border-color: var(--accent) !important; color: var(--accent) !important; }
    .coh-pct-label { font-family: var(--font-mono); font-size: 11px; color: var(--accent); line-height: 28px; text-align: center; display: block; }

    .run-btn { background: var(--accent) !important; border-color: var(--accent) !important; color: #0f3638 !important; font-weight: 600 !important; font-size: 13px !important; height: 36px !important; border-radius: var(--radius) !important; transition: background 180ms ease; }
    .run-btn:hover { background: var(--accent-hover) !important; border-color: var(--accent-hover) !important; }

    .coh-dropdown .Select-control, .coh-dropdown .Select-menu-outer { background: var(--surface3) !important; border-color: var(--border) !important; color: var(--text) !important; font-family: var(--font-mono) !important; font-size: 12px !important; }
    .coh-dropdown .Select-value-label, .coh-dropdown .Select-placeholder { color: var(--text) !important; }
    .coh-dropdown .Select-option { background: var(--surface3) !important; color: #e8e7e4 !important; }
    .coh-dropdown .Select-option.is-focused { background: #313130 !important; color: #ffffff !important; }
    .coh-dropdown .Select-option.is-selected { background: var(--accent) !important; color: #0f3638 !important; }
    .coh-dropdown .VirtualizedSelectOption { color: #e8e7e4 !important; }
    /* Dash 2.x uses react-select v5 */
    .coh-dropdown [class*="-option"] { background: var(--surface3) !important; color: #e8e7e4 !important; }
    .coh-dropdown [class*="-option"]:hover { background: #3a3937 !important; color: #ffffff !important; }
    .coh-dropdown [class*="-singleValue"] { color: #e8e7e4 !important; }
    .coh-dropdown [class*="-placeholder"] { color: var(--muted) !important; }
    .coh-dropdown [class*="-menu"] { background: var(--surface3) !important; border: 1px solid var(--border) !important; }

    .divider { border-color: var(--border) !important; margin: 14px 0 !important; }
    .status-msg { font-family: var(--font-mono); font-size: 11px; color: var(--accent); margin-top: 8px; min-height: 16px; text-align: center; }

    .update-mode-radio .form-check { display: inline-flex; align-items: center; gap: 4px; margin-right: 10px; }
    .update-mode-radio .form-check-label { font-size: 12px; color: var(--muted); }
    .update-mode-radio .form-check-input:checked { background-color: var(--accent) !important; border-color: var(--accent) !important; }
    .update-mode-radio .form-check-input { background-color: var(--surface3) !important; border-color: var(--border) !important; }

    /* Plot wrapper + overlay lock system */
    .plot-wrapper { position: relative; background: var(--bg); }
    .plot-overlay {
        position: absolute; inset: 0; z-index: 5;
        display: flex; align-items: flex-start; justify-content: flex-end;
        padding: 6px 8px;
        cursor: pointer;
        transition: background 120ms ease;
    }
    .plot-overlay.locked {
        pointer-events: all;
        background: transparent;
    }
    .plot-overlay.locked::after {
        content: '🔒';
        font-size: 13px;
        opacity: 0.40;
        line-height: 1;
        pointer-events: none;
    }
    .plot-overlay.locked:hover { background: rgba(79,152,163,0.04); }
    .plot-overlay.locked:hover::after { opacity: 0.85; }
    .plot-overlay.unlocked {
        pointer-events: none;
        background: transparent;
    }
    .plot-overlay.unlocked::after {
        content: '🔓';
        font-size: 13px;
        opacity: 0.30;
        line-height: 1;
        pointer-events: none;
    }

    .plot-row { margin: 0 !important; }
    .plot-row .col { padding: 0 !important; }
    .plot-row .js-plotly-plot { width: 100% !important; }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--surface3); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--muted); }

    /* Per-parameter slider colours — coloured labels, white sliders fallback
       Full Dash 4.x / Radix UI colours are in assets/slider_colors.css */
    .param-slider-noise_sigma         [data-radix-slider-thumb] { background: #00bfff !important; border-color: #00bfff !important; }
    .param-slider-noise_sigma         .dash-slider-range { background: #00bfff !important; }
    .param-slider-scaling_factor      [data-radix-slider-thumb] { background: #ff1493 !important; border-color: #ff1493 !important; }
    .param-slider-scaling_factor      .dash-slider-range { background: #ff1493 !important; }
    .param-slider-leak                [data-radix-slider-thumb] { background: #ffa500 !important; border-color: #ffa500 !important; }
    .param-slider-leak                .dash-slider-range { background: #ffa500 !important; }
    .param-slider-residual_after_bout [data-radix-slider-thumb] { background: #00ff7f !important; border-color: #00ff7f !important; }
    .param-slider-residual_after_bout .dash-slider-range { background: #00ff7f !important; }
    .param-slider-inactive_time       [data-radix-slider-thumb] { background: #ff6347 !important; border-color: #ff6347 !important; }
    .param-slider-inactive_time       .dash-slider-range { background: #ff6347 !important; }
    .param-slider-threshold           [data-radix-slider-thumb] { background: #ffffff !important; border-color: #ffffff !important; }
    .param-slider-threshold           .dash-slider-range { background: #ffffff !important; }

    </style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>
"""

if __name__ == "__main__":
    _warmup()
    print("\n" + "═" * 60)
    print("  DDM Explorer  ·  http://127.0.0.1:8050")
    print("═" * 60 + "\n")
    app.run(debug=True, port=8050)
