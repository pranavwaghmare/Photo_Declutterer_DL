"""
app/review_ui.py
────────────────
Streamlit review UI for the Photo De-Clutterer.

Features
--------
- Load results.parquet produced by the pipeline.
- Browse photos grouped by Duplicate_Cluster_ID or Content_Category.
- Highlight the recommended "Keep" photo per duplicate group.
- Checkbox overrides: toggle any individual recommendation.
- Confirm button → move files marked "Delete" to data/trash/ (soft-delete only;
  no os.remove is ever called).

Run:
    streamlit run app/review_ui.py -- --results data/processed/results.parquet
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

# ── resolve project root ──────────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent
_ROOT = _APP_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_RESULTS = str(_ROOT / "data" / "processed" / "results.parquet")
DEFAULT_TRASH = str(_ROOT / "data" / "trash")
THUMB_WIDTH = 200  # pixels for thumbnail display


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_results(parquet_path: str) -> pd.DataFrame:
    """Load and cache the results parquet."""
    return pd.read_parquet(parquet_path)


def _open_thumb(file_path: str, width: int = THUMB_WIDTH) -> Image.Image | None:
    """Open an image and resize to thumbnail width; return None on failure."""
    try:
        img = Image.open(file_path).convert("RGB")
        ratio = width / img.width
        height = int(img.height * ratio)
        return img.resize((width, height), Image.LANCZOS)
    except (UnidentifiedImageError, FileNotFoundError, OSError):
        return None


def _soft_delete(file_path: str, trash_dir: str) -> bool:
    """Move *file_path* to *trash_dir*.  Returns True on success."""
    src = Path(file_path)
    dst_dir = Path(trash_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    # Avoid name collisions in trash
    counter = 1
    while dst.exists():
        dst = dst_dir / f"{src.stem}_{counter}{src.suffix}"
        counter += 1

    try:
        shutil.move(str(src), str(dst))
        return True
    except OSError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# UI sections
# ─────────────────────────────────────────────────────────────────────────────

def _render_image_card(
    row: pd.Series,
    key_prefix: str,
    show_cluster_info: bool = True,
) -> tuple[str, str]:
    """Render a single image card and return (Image_ID, recommendation)."""
    image_id = str(row["Image_ID"])
    file_path = str(row["File_Path"])
    is_keep = row.get("Keep_Recommendation", "Keep") == "Keep"
    lap_var = float(row.get("Laplacian_Variance", 0.0))
    blur_label = str(row.get("Blur_Quality_Label", "unknown"))
    sim_score = float(row.get("Similarity_Score", 1.0))
    category = str(row.get("Content_Category", "General"))

    thumb = _open_thumb(file_path)
    if thumb:
        st.image(thumb, use_container_width=True)
    else:
        st.info("🖼 No Preview Available")

    status_icon = "🟢" if is_keep else "🔴"
    status_text = "Keep" if is_keep else "Delete"
    st.markdown(f"**{status_icon} Recommendation: {status_text}**")

    checked = st.checkbox(
        "Keep this photo",
        value=is_keep,
        key=f"{key_prefix}_{image_id}",
    )
    user_decision = "Keep" if checked else "Delete"

    fname = Path(file_path).name
    details = f"📄 `{fname}`\n\n" f"• **Quality**: `{blur_label}` (Laplacian: `{lap_var:.1f}`)"
    if show_cluster_info and row.get("Duplicate_Cluster_ID", -1) != -1:
        details += f"\n• **Similarity**: `{sim_score:.3f}`"
    if category:
        details += f"\n• **Category**: `{category}`"

    with st.expander("ℹ Details", expanded=False):
        st.caption(details)

    return image_id, user_decision


def render_duplicate_groups(df: pd.DataFrame) -> dict[str, str]:
    """
    Render each duplicate cluster as a clean responsive card grid.

    Returns a dict of image_id → user-chosen recommendation ("Keep"/"Delete").
    """
    overrides: dict[str, str] = {}

    clustered = df[df["Duplicate_Cluster_ID"] != -1].copy()
    singletons = df[df["Duplicate_Cluster_ID"] == -1].copy()

    if not clustered.empty:
        st.subheader("📸 Near-Duplicate Clusters")
        st.caption("The highest quality image in each group is automatically recommended to **Keep**.")

        for cluster_id, group in clustered.groupby("Duplicate_Cluster_ID"):
            group = group.sort_values(
                by=["Keep_Recommendation", "Similarity_Score"],
                ascending=[False, False],
            )
            n_items = len(group)
            with st.expander(f"📁 Cluster #{cluster_id}  —  {n_items} similar photos", expanded=True):
                n_cols = min(4, max(1, n_items))
                chunks = [group.iloc[i : i + n_cols] for i in range(0, len(group), n_cols)]
                for chunk in chunks:
                    cols = st.columns(len(chunk))
                    for col, (_, row) in zip(cols, chunk.iterrows()):
                        with col:
                            iid, dec = _render_image_card(row, key_prefix=f"dup_{cluster_id}")
                            overrides[iid] = dec

    if not singletons.empty:
        st.subheader("🖼 Singleton Photos")
        st.caption("Unique images without near-duplicates. Blurry singletons are flagged for review.")

        n_cols = 4
        chunks = [singletons.iloc[i : i + n_cols] for i in range(0, len(singletons), n_cols)]
        for chunk in chunks:
            cols = st.columns(len(chunk))
            for col, (_, row) in zip(cols, chunk.iterrows()):
                with col:
                    iid, dec = _render_image_card(row, key_prefix="single", show_cluster_info=False)
                    overrides[iid] = dec

    return overrides


def render_category_view(df: pd.DataFrame) -> dict[str, str]:
    """Render photos grouped by Content_Category."""
    overrides: dict[str, str] = {}

    categories = sorted(df["Content_Category"].dropna().unique())
    if not categories:
        st.info("No content categories found in dataset.")
        return overrides

    selected_cat = st.selectbox("Select Content Category", categories)
    subset = df[df["Content_Category"] == selected_cat].copy()

    st.write(f"Showing **{len(subset)}** photo(s) in `{selected_cat}`")

    n_cols = 4
    chunks = [subset.iloc[i : i + n_cols] for i in range(0, len(subset), n_cols)]
    for chunk in chunks:
        cols = st.columns(len(chunk))
        for col, (_, row) in zip(cols, chunk.iterrows()):
            with col:
                iid, dec = _render_image_card(row, key_prefix="cat")
                overrides[iid] = dec

    return overrides


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Photo De-Clutterer",
        page_icon="📷",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📷 Photo De-Clutterer — Review & Clean Studio")
    st.markdown(
        "Review AI recommendations, inspect near-duplicate clusters, and declutter your photo library safely."
    )

    # ── Sidebar controls ──────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙ Configuration")
        results_path = st.text_input("Results Parquet Path", value=DEFAULT_RESULTS)
        trash_dir = st.text_input("Trash Destination Directory", value=DEFAULT_TRASH)

        st.markdown("---")
        st.header("🧭 Navigation")
        view_mode = st.radio(
            "View Mode",
            ["Duplicate Groups", "Content Categories", "Cluster Visualisation & Analytics"],
            index=0,
        )

        st.markdown("---")
        filter_status = st.selectbox(
            "Filter Recommendation",
            ["Show All", "Candidates for Trash (Delete)", "Retained (Keep)"],
        )

        st.markdown("---")
        st.info(
            "🛡 **Safe Deletion Guarantee**:\n"
            "Files are moved to the trash folder upon confirmation. No files are permanently destroyed."
        )

    # ── Load data ─────────────────────────────────────────────────────────
    if not Path(results_path).exists():
        st.warning(
            f"Results file not found: `{results_path}`\n\n"
            "Run the pipeline first to analyze your photo collection:\n"
            "```bash\npython -m src.pipeline --input data/raw\n```"
        )
        return

    with st.spinner("Loading decluttering results …"):
        df = load_results(results_path)

    if df.empty:
        st.info("The results parquet file is empty.")
        return

    # Apply filter if requested
    filtered_df = df.copy()
    if filter_status == "Candidates for Trash (Delete)":
        filtered_df = filtered_df[filtered_df["Keep_Recommendation"] == "Delete"]
    elif filter_status == "Retained (Keep)":
        filtered_df = filtered_df[filtered_df["Keep_Recommendation"] == "Keep"]

    # ── Summary metrics ───────────────────────────────────────────────────
    n_total = len(df)
    n_keep = int((df["Keep_Recommendation"] == "Keep").sum())
    n_delete = int((df["Keep_Recommendation"] == "Delete").sum())
    n_clusters = df[df["Duplicate_Cluster_ID"] != -1]["Duplicate_Cluster_ID"].nunique()
    pct_reduction = (100.0 * n_delete / max(1, n_total))

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Images", n_total)
    col2.metric("Recommended Keep", n_keep)
    col3.metric("Recommended Delete", n_delete)
    col4.metric("Duplicate Clusters", n_clusters)
    col5.metric("Storage Reduction", f"{pct_reduction:.1f}%")

    st.markdown("---")

    # ── Main view ─────────────────────────────────────────────────────────
    overrides: dict[str, str] = {}

    if view_mode == "Duplicate Groups":
        overrides = render_duplicate_groups(filtered_df)
    elif view_mode == "Content Categories":
        overrides = render_category_view(filtered_df)
    else:
        st.subheader("📊 Library Visualisation & Analytics")
        viz_path = _ROOT / "reports" / "content_clusters.png"
        if viz_path.exists():
            st.image(str(viz_path), caption="2-D Semantic Embedding Projection (UMAP / t-SNE)", use_container_width=True)
        else:
            st.info("Cluster plot not generated yet. Run the pipeline with cluster visualization enabled.")

        # Breakdown charts
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("#### 🏷 Content Category Distribution")
            cat_counts = df["Content_Category"].value_counts()
            st.bar_chart(cat_counts)
        with col_b:
            st.write("#### 🔍 Blur Quality Breakdown")
            blur_counts = df["Blur_Quality_Label"].value_counts()
            st.bar_chart(blur_counts)

    # ── Confirmation & soft-delete ────────────────────────────────────────
    if view_mode in ("Duplicate Groups", "Content Categories"):
        st.markdown("---")
        n_to_delete = sum(1 for v in overrides.values() if v == "Delete")
        st.markdown(f"### 🗑 Batch Action: Move Marked Photos to Trash")
        st.write(f"**{n_to_delete}** photo(s) currently selected for deletion.")

        if st.button("🗑 Move Selected to Trash", type="primary", disabled=n_to_delete == 0):
            id_to_path = df.set_index("Image_ID")["File_Path"].to_dict()
            moved, failed = 0, 0
            progress_bar = st.progress(0)
            items = [(iid, rec) for iid, rec in overrides.items() if rec == "Delete"]
            for step, (iid, _) in enumerate(items):
                fp = id_to_path.get(iid)
                if fp and _soft_delete(fp, trash_dir):
                    moved += 1
                else:
                    failed += 1
                progress_bar.progress((step + 1) / max(1, len(items)))

            if moved:
                st.success(f"Successfully moved {moved} photo(s) to `{trash_dir}`.")
                load_results.clear()
            if failed:
                st.warning(f"{failed} photo(s) could not be moved (already moved or missing).")


if __name__ == "__main__":
    main()
