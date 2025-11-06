import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Pitch Grids — In/Out of Possession", layout="wide")
plt.rcParams['figure.dpi'] = 80

# -----------------------
# Helpers (geometry/plots)
# -----------------------
def compute_grid_geometry(length, width, cell_size):
    """Snap length/width to an integer number of cells to keep imshow & grid aligned."""
    ncols = max(1, int(round(length / cell_size)))
    nrows = max(1, int(round(width  / cell_size)))
    eff_length = ncols * cell_size
    eff_width  = nrows * cell_size
    x_centers = (np.arange(ncols) + 0.5) * cell_size
    y_centers = (np.arange(nrows) + 0.5) * cell_size
    return eff_length, eff_width, cell_size, x_centers, y_centers, ncols, nrows

def compute_fields(
    length, width, cell_size,
    max_value, min_value, dist_decay, angle_drop, goal_radius, power_angle, power_dist,
    oop_scale, oop_offset, sigmoid_weight, sigmoid_center
):
    # Snap geometry
    eff_length, eff_width, cell_size, x_centers, y_centers, ncols, nrows = compute_grid_geometry(length, width, cell_size)

    ny, nx = len(y_centers), len(x_centers)
    values_in  = np.zeros((ny, nx), dtype=float)
    values_out = np.zeros_like(values_in)

    goal_x, goal_y = eff_length, eff_width/2

    for ix, x in enumerate(x_centers):
        for iy, y in enumerate(y_centers):
            dx = goal_x - x
            dy = y - goal_y
            dist  = np.hypot(dx, dy)
            angle = np.degrees(np.abs(np.arctan2(dy, dx)))

            # In-possession
            if dist <= goal_radius:
                val_in = max_value
            else:
                val_in = max_value * np.exp(-((dist / max(dist_decay, 1e-6))**power_dist +
                                              (angle / max(angle_drop, 1e-6))**power_angle))
                val_in = max(val_in, min_value)

            # Width boost in own half
            if x < eff_length/2:
                width_factor = 1 + 0.9 * abs((y - eff_width/2) / (eff_width/2))**1.7
                val_in *= width_factor

            values_in[iy, ix] = val_in

            # Out-of-possession transform
            norm_x  = x / eff_length
            sigmoid = 1.0 / (1.0 + np.exp(-sigmoid_weight * (norm_x - sigmoid_center)))
            val_out = oop_offset - oop_scale * sigmoid * val_in
            values_out[iy, ix] = val_out

    return eff_length, eff_width, x_centers, y_centers, values_in, values_out

def draw_pitch_grid(ax, eff_length, eff_width, cell_size):
    ax.set_xlim(0, eff_length)
    ax.set_ylim(0, eff_width)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor('#228B22')
    xs = np.arange(0, eff_length + 1e-9, cell_size)
    ys = np.arange(0, eff_width  + 1e-9, cell_size)
    for x in xs:
        ax.axvline(x, color='black', lw=1, alpha=0.8, zorder=3)
    for y in ys:
        ax.axhline(y, color='black', lw=1, alpha=0.8, zorder=3)
    ax.set_aspect('equal', adjustable='box')

def annotate_cells(ax, x_centers, y_centers, arr, fmt="{:.2f}", color='black', fs=6, weight='semibold'):
    for ix, x in enumerate(x_centers):
        for iy, y in enumerate(y_centers):
            ax.text(
                x, y, fmt.format(arr[iy, ix]),
                ha='center', va='center', fontsize=fs,
                color=color, fontweight=weight, zorder=4, clip_on=True
            )

def plot_grid(arr, x_centers, y_centers, eff_length, eff_width, cell_size,
              cmap, vmin, vmax, title, text_color='black', text_fs=6):
    fig, ax = plt.subplots(figsize=(16, 9))
    im = ax.imshow(
        arr,
        extent=[0, eff_length, 0, eff_width],
        origin='lower',
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        interpolation='nearest',
        alpha=0.97, zorder=1
    )
    draw_pitch_grid(ax, eff_length, eff_width, cell_size)
    annotate_cells(ax, x_centers, y_centers, arr, fmt="{:.2f}", color=text_color, fs=text_fs, weight='semibold')
    fig.patch.set_facecolor('#228B22')
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    ax.set_title(title, fontsize=16, color='white')
    fig.tight_layout()
    return fig

def df_from_grid(arr, x_centers, y_centers):
    df = pd.DataFrame(arr, index=[f"y={y:.1f}" for y in y_centers],
                           columns=[f"x={x:.1f}" for x in x_centers])
    return df

def grid_from_df(df):
    return df.values.astype(float)

def download_png(fig, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches='tight')
    st.download_button("Download PNG", data=buf.getvalue(), file_name=filename, mime="image/png")

# -----------------------
# Defaults (session state)
# -----------------------
def ensure_state():
    defaults = {
        "length": 105.0, "width": 68.0, "cell_size": 2.0,
        "max_value": 1.0, "min_value": 0.001, "dist_decay": 45.0, "angle_drop": 120.0,
        "goal_radius": 7.0, "power_angle": 2.9, "power_dist": 1.7,
        "oop_scale": 0.95, "oop_offset": 0.05, "sigmoid_weight": 15.0, "sigmoid_center": 0.3,
        "oop_darker_negative": True
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if "x_centers" not in st.session_state:
        eff_length, eff_width, x_c, y_c, vin, vout = compute_fields(
            st.session_state.length, st.session_state.width, st.session_state.cell_size,
            st.session_state.max_value, st.session_state.min_value,
            st.session_state.dist_decay, st.session_state.angle_drop, st.session_state.goal_radius,
            st.session_state.power_angle, st.session_state.power_dist,
            st.session_state.oop_scale, st.session_state.oop_offset,
            st.session_state.sigmoid_weight, st.session_state.sigmoid_center
        )
        st.session_state.eff_length = eff_length
        st.session_state.eff_width = eff_width
        st.session_state.x_centers = x_c
        st.session_state.y_centers = y_c
        st.session_state.values_in = vin
        st.session_state.values_out = vout

ensure_state()

# -----------------------
# UI — Sidebar Controls
# -----------------------
with st.sidebar:
    st.header("Parameters")
    length = st.number_input("Pitch length (m)", value=st.session_state.length, step=1.0)
    width  = st.number_input("Pitch width (m)",  value=st.session_state.width, step=1.0)
    cell_size = st.number_input("Cell size (m)", value=st.session_state.cell_size, step=0.5, min_value=0.5)

    st.subheader("In-possession decay")
    max_value = st.number_input("max_value", value=st.session_state.max_value, step=0.1, format="%.3f")
    min_value = st.number_input("min_value", value=st.session_state.min_value, step=0.001, format="%.3f")
    dist_decay = st.number_input("dist_decay", value=st.session_state.dist_decay, step=1.0)
    angle_drop = st.number_input("angle_drop", value=st.session_state.angle_drop, step=5.0)
    goal_radius = st.number_input("goal_radius", value=st.session_state.goal_radius, step=0.5)
    power_angle = st.number_input("power_angle", value=st.session_state.power_angle, step=0.1)
    power_dist  = st.number_input("power_dist",  value=st.session_state.power_dist, step=0.1)

    st.subheader("Out-of-possession transform")
    oop_scale  = st.number_input("oop_scale",  value=st.session_state.oop_scale, step=0.05)
    oop_offset = st.number_input("oop_offset", value=st.session_state.oop_offset, step=0.01)
    sigmoid_weight = st.number_input("sigmoid_weight", value=st.session_state.sigmoid_weight, step=1.0)
    sigmoid_center = st.number_input("sigmoid_center", value=st.session_state.sigmoid_center, step=0.05)

    st.subheader("Display options")
    oop_darker_negative = st.checkbox("Out-of-possession: darker = more negative", value=st.session_state.oop_darker_negative)

    regen = st.button("Recompute grids from parameters", use_container_width=True)

# Recompute on demand
if regen:
    eff_length, eff_width, x_centers, y_centers, vin, vout = compute_fields(
        length, width, cell_size,
        max_value, min_value, dist_decay, angle_drop, goal_radius, power_angle, power_dist,
        oop_scale, oop_offset, sigmoid_weight, sigmoid_center
    )
    st.session_state.length = length
    st.session_state.width = width
    st.session_state.cell_size = cell_size
    st.session_state.max_value = max_value
    st.session_state.min_value = min_value
    st.session_state.dist_decay = dist_decay
    st.session_state.angle_drop = angle_drop
    st.session_state.goal_radius = goal_radius
    st.session_state.power_angle = power_angle
    st.session_state.power_dist = power_dist
    st.session_state.oop_scale = oop_scale
    st.session_state.oop_offset = oop_offset
    st.session_state.sigmoid_weight = sigmoid_weight
    st.session_state.sigmoid_center = sigmoid_center
    st.session_state.oop_darker_negative = oop_darker_negative

    st.session_state.eff_length = eff_length
    st.session_state.eff_width = eff_width
    st.session_state.x_centers = x_centers
    st.session_state.y_centers = y_centers
    st.session_state.values_in = vin
    st.session_state.values_out = vout

# -----------------------
# Editing tools
# -----------------------
st.header("Edit Grids")

colA, colB = st.columns(2, gap="large")

with colA:
    st.subheader("In-possession (table editor)")
    df_in = df_from_grid(st.session_state.values_in, st.session_state.x_centers, st.session_state.y_centers)
    edited_df_in = st.data_editor(df_in, height=400, use_container_width=True, num_rows="fixed")
    if st.button("Apply table edits to In-possession grid"):
        new_vals = grid_from_df(edited_df_in)
        if new_vals.shape == st.session_state.values_in.shape:
            st.session_state.values_in = new_vals
            st.success("In-possession grid updated")
        else:
            st.error(f"Shape mismatch: got {new_vals.shape}, expected {st.session_state.values_in.shape}")

with colB:
    st.subheader("Out-of-possession (table editor)")
    df_out = df_from_grid(st.session_state.values_out, st.session_state.x_centers, st.session_state.y_centers)
    edited_df_out = st.data_editor(df_out, height=400, use_container_width=True, num_rows="fixed")
    if st.button("Apply table edits to Out-of-possession grid"):
        new_vals = grid_from_df(edited_df_out)
        if new_vals.shape == st.session_state.values_out.shape:
            st.session_state.values_out = new_vals
            st.success("Out-of-possession grid updated")
        else:
            st.error(f"Shape mismatch: got {new_vals.shape}, expected {st.session_state.values_out.shape}")

st.subheader("Single-cell editor")
with st.form("single_cell_edit"):
    edit_grid = st.selectbox("Grid", ["In-possession", "Out-of-possession"])
    ix = st.number_input("x index (0-based)", min_value=0, max_value=len(st.session_state.x_centers)-1, value=0)
    iy = st.number_input("y index (0-based)", min_value=0, max_value=len(st.session_state.y_centers)-1, value=0)
    new_val = st.number_input("New value", value=0.0, format="%.6f")
    submitted = st.form_submit_button("Apply single-cell edit")
    if submitted:
        if edit_grid == "In-possession":
            st.session_state.values_in[iy, ix] = float(new_val)
        else:
            st.session_state.values_out[iy, ix] = float(new_val)
        st.success(f"Updated {edit_grid} at (iy={iy}, ix={ix})")

# CSV import/export
st.subheader("Import/Export CSV")
colC, colD, colE, colF = st.columns(4)
with colC:
    csv_in = io.StringIO()
    pd.DataFrame(st.session_state.values_in).to_csv(csv_in, index=False)
    st.download_button("Download In-pos CSV", csv_in.getvalue(), file_name="values_in.csv", mime="text/csv")
with colD:
    csv_out = io.StringIO()
    pd.DataFrame(st.session_state.values_out).to_csv(csv_out, index=False)
    st.download_button("Download Out-of-pos CSV", csv_out.getvalue(), file_name="values_out.csv", mime="text/csv")
with colE:
    uploaded_in = st.file_uploader("Upload In-pos CSV", type=["csv"], key="upload_in")
    if uploaded_in:
        arr = pd.read_csv(uploaded_in).values
        if arr.shape == st.session_state.values_in.shape:
            st.session_state.values_in = arr
            st.success("Loaded In-possession grid from CSV")
        else:
            st.error(f"Uploaded shape {arr.shape} does not match expected {st.session_state.values_in.shape}")
with colF:
    uploaded_out = st.file_uploader("Upload Out-of-pos CSV", type=["csv"], key="upload_out")
    if uploaded_out:
        arr = pd.read_csv(uploaded_out).values
        if arr.shape == st.session_state.values_out.shape:
            st.session_state.values_out = arr
            st.success("Loaded Out-of-possession grid from CSV")
        else:
            st.error(f"Uploaded shape {arr.shape} does not match expected {st.session_state.values_out.shape}")

# -----------------------
# Plots
# -----------------------
st.header("Plots")

cmap_out = "Blues_r" if st.session_state.oop_darker_negative else "Blues"

vmin_in, vmax_in = st.session_state.min_value, st.session_state.max_value
vmin_out, vmax_out = float(np.min(st.session_state.values_out)), float(np.max(st.session_state.values_out))

left, right = st.columns(2, gap="large")
with left:
    fig_in = plot_grid(
        st.session_state.values_in,
        st.session_state.x_centers, st.session_state.y_centers,
        st.session_state.eff_length, st.session_state.eff_width, st.session_state.cell_size,
        cmap="YlOrRd", vmin=vmin_in, vmax=vmax_in,
        title="Pitch Grid — In Possession",
        text_color='black', text_fs=6
    )
    st.pyplot(fig_in, use_container_width=True)
    download_png(fig_in, "pitch_in_possession.png")

with right:
    fig_out = plot_grid(
        st.session_state.values_out,
        st.session_state.x_centers, st.session_state.y_centers,
        st.session_state.eff_length, st.session_state.eff_width, st.session_state.cell_size,
        cmap=cmap_out, vmin=vmin_out, vmax=vmax_out,
        title="Pitch Grid — Out of Possession",
        text_color='black', text_fs=6
    )
    st.pyplot(fig_out, use_container_width=True)
    download_png(fig_out, "pitch_out_of_possession.png")
