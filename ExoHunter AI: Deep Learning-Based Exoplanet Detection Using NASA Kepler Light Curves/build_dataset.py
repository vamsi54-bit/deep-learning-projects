# ─────────────────────────────────────────────
# BUILD DATASET
# ─────────────────────────────────────────────

# Load the NASA Exoplanet Archive CSV.
# comment="#"   → skip header rows that start with '#'
# header=None   → don't auto-assign column names (we do it manually below)
# low_memory=False → read full file before inferring dtypes (avoids mixed-type warnings)
df = pd.read_csv(
    r"C:\Users\DELL\PycharmProjects\PythonProject\PS_2026.05.31_23.07.24.csv",
    comment="#",
    header=None,
    low_memory=False
)

# Row 1 (index 1) contains the actual column names in this CSV format.
# Assign them, then drop the first two metadata rows and reset the index.
df.columns = df.iloc[1]
df = df.iloc[2:].reset_index(drop=True)

# Keep only the "default" row per planet (one row per planet, not per paper)
# and filter to planets discovered by the Kepler mission only.
df = df[df["default_flag"].astype(str) == "1"]
df = df[df["disc_facility"] == "Kepler"]

# Get unique host star names — each star may have multiple planets,
# but we only need to download its light curve once.
star_names = df["hostname"].dropna().unique()

print(f"Found {len(star_names)} Kepler stars")

X = []          # Will collect one normalised flux array per star
MAX_STARS = 50  # Cap downloads during development / testing

for i, star in enumerate(star_names[:MAX_STARS]):

    try:
        print(f"{i+1}/{MAX_STARS} : {star}")

        # Query the MAST archive for all available Kepler light curves for this star.
        search = lk.search_lightcurve(
            star,
            mission="Kepler"
        )

        # Skip stars with no available light curve data.
        if len(search) == 0:
            continue

        # Download the first (or best) result and remove cadences with NaN flux values.
        lc = search.download().remove_nans()

        flux = lc.flux.value  # Extract raw flux as a plain NumPy array

        # Skip stars with very short light curves — too little data to be useful.
        if len(flux) < 300:
            continue

        # ── Resample to a fixed length of 300 points ──────────────────────────
        # Light curves from different stars have different cadence counts.
        # We map both the original and target grids onto [0, 1] and interpolate
        # so every sample fed to the model has the same shape.
        original = np.linspace(0, 1, len(flux))  # Original time axis (normalised)
        target   = np.linspace(0, 1, 300)         # Desired fixed-length time axis

        flux = interpolate.interp1d(
            original,
            flux,
            kind="linear"   # Linear interpolation between cadence points
        )(target)

        # ── Z-score normalisation ──────────────────────────────────────────────
        # Subtract the mean and divide by the standard deviation so every light
        # curve has mean ≈ 0 and std ≈ 1. This removes brightness differences
        # between stars and keeps transit dips on a comparable scale.
        flux = (flux - np.mean(flux)) / np.std(flux)

        X.append(flux)

    except Exception:
        # Silently skip any star that fails (download error, bad data, etc.)
        continue

# Stack the list of 1-D arrays into a 2-D NumPy array: shape (n_stars, 300)
X = np.array(X)

print("Dataset Shape:", X.shape)  # Expected: (≤50, 300)

# ─────────────────────────────────────────────
# SAVE & RELOAD
# ─────────────────────────────────────────────

# Persist the processed dataset to disk so we don't re-download every run.
np.save("exo_dataset.npy", X)
print("Dataset saved successfully!")

# Reload from disk (useful as a sanity check, or as the entry point in later scripts).
X = np.load("exo_dataset.npy")

# ─────────────────────────────────────────────
# CONVERT TO PYTORCH TENSOR
# ─────────────────────────────────────────────

# Convert to a float32 tensor — PyTorch's default precision for model weights.
# unsqueeze(-1) adds a channel dimension at the end:
#   (n_stars, 300)  →  (n_stars, 300, 1)
# This makes the shape compatible with 1-D convolutional layers (Conv1d)
# which expect input as (batch, sequence_length, channels).
X_tensor = torch.tensor(
    X,
    dtype=torch.float32
).unsqueeze(-1)

print(X_tensor.shape)  # Expected: torch.Size([≤50, 300, 1])
