import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from scipy import ndimage
import sys

# ─────────────────────────────────────────────
# 1. DATA LOADING & VISUALISATION
# ─────────────────────────────────────────────

def load_example_data(path: str, example_num: int):

    mat_contents = scio.loadmat(path)
    
    amp = f'amplitudes{example_num}'
    dist = f'distances{example_num}'
    clo = f'cloud{example_num}'
    
    amplitudes = mat_contents.get(amp)
    distances = mat_contents.get(dist)
    cloud = mat_contents.get(clo)
    
    if amplitudes is None or distances is None or cloud is None:
        raise KeyError(f"Example {example_num} data not found in {path}")
    
    return amplitudes, distances, cloud

def visualise_data(A, D, PC, example_num: int, subsample: int = 4):
    """Visualise amplitude image, distance image, and point cloud."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Amplitude image (take first channel if multi-channel)
    axes[0].imshow(A[..., 0] if A.ndim == 3 else A, cmap="gray")
    axes[0].set_title("Amplitude Image")
    axes[0].axis("off")

    # Distance image
    axes[1].imshow(D[..., 0] if D.ndim == 3 else D, cmap="jet")
    axes[1].set_title("Distance Image")
    axes[1].axis("off")

    # 3-D point cloud (subsampled)
    ax3d = fig.add_subplot(1, 3, 3, projection="3d")
    pc = PC[::subsample, ::subsample, :]    # subsample rows & cols
    valid = pc[..., 2] != 0            # ignore z == 0 points
    xs, ys, zs = pc[valid, 0], pc[valid, 1], pc[valid, 2]
    ax3d.scatter(xs, ys, zs, c=zs, cmap="jet", s=1)
    ax3d.set_title("Point Cloud")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")


    save_path = f"Results/step1_visualisation_example{example_num}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.show()
    print("Saved: step1_visualisation.png")


# =============================================================================
# STEP 2 – MEDIAN FILTER 
# =============================================================================

def apply_median_filter(D, example_num, kernel_size=5):
    """
    Apply a median filter (window = kernel_size x kernel_size) to distance image D.
    """
    print("Applying Median Filter")
    D_2d = D[..., 0] if D.ndim == 3 else D
    D_filtered = ndimage.median_filter(D_2d, size=kernel_size)


    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Median Filter", fontsize=13)

    axes[0].imshow(D_2d, cmap="jet")           # always show original 2D
    axes[0].set_title("Original Distance Image")
    axes[0].axis("off")
    axes[1].imshow(D_filtered, cmap="jet")     # always show filtered result
    axes[1].set_title("After Median Filter")
    axes[1].axis("off")

    plt.tight_layout()
    save_path = f"Results/step2_median_filter_example{example_num}.png"
    plt.savefig(save_path, dpi=120)
    plt.show()
    print("Saved: step2_median_filter.png")
    return D_filtered

# ─────────────────────────────────────────────
# 3. RANSAC for floor detection
# ─────────────────────────────────────────────

def fit_plane_to_3_points(p1, p2, p3):
    """
    Compute the plane through exactly 3 points in 3D.
    Returns (normal, d) or (None, None) if points are nearly in a line.
    """
    v1 = p2 - p1 #vector lying IN the plane
    v2 = p3 - p1 # another vector lying IN the plane
    normal = np.cross(v1, v2)       # numpy.cross - v1 x v2 perpendicular to both = the plane's normal
    length = np.linalg.norm(normal)

    if length < 1e-10:              # collinear points – cannot define a plane
        return None, None

    normal = normal / length        # make it a unit vector  (length = 1)
    d = np.dot(normal, p1)     # the plane's offset 
    return normal, d

def ransac_plane(points, threshold=0.01, max_iterations=1000):
    """
    RANSAC plane fitting

    points: array of valid 3-D points
    threshold: inlier distance threshold from the plane
    max_iterations: maximum number of RANSAC iterations

    The floor is the biggest flat surface → gets the most inliers.
    z == 0 points are invalid measurements (exercise sheet note) and skipped.

    """
    N = len(points)
    best_normal = None
    best_d = None
    best_inliers = np.zeros(N, dtype=bool)
    best_count = 0

    # Fixed random seed so we get the same result every run
    rng = np.random.default_rng(seed=42)

    for i in range(max_iterations):   # while (i < N) in pseudocode
        
        # minimal_sample: pick 3 random points (k=3 for a plane)
        idx = rng.choice(N, size=3, replace=False)
        p1, p2, p3 = points[idx[0]], points[idx[1]], points[idx[2]]

         # estimate_model: fit a plane to those 3 points
        normal, d = fit_plane_to_3_points(p1, p2, p3)
        if normal is None:
            continue  # degenerate – try again

         # computeInliers: count how many points are close to this plane
        inliers = count_inliers(points, normal, d, threshold)

        count = inliers.sum()

         # keep best model
        if count > best_count:
            best_count = count
            best_normal = normal
            best_d = d
            best_inliers = inliers

             # "return if all candidate points are within the inlier set"
            # (exercise sheet requirement)
            if best_count == N:
                break
     
    # ── Refinement: refit using ALL inliers for higher accuracy ───────────────
    # full_matrices=False is REQUIRED – without it numpy tries to allocate
    # a 133000x133000 matrix and crashes with MemoryError!
    if best_count >= 3:
        pts_in = points[best_inliers]
        centroid = pts_in.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts_in - centroid, full_matrices=False)
        best_normal = Vt[-1]
        best_d = np.dot(best_normal, centroid)
        best_inliers = count_inliers(points, best_normal, best_d, threshold)

    return best_normal, best_d, best_inliers





def count_inliers(points, normal, d, threshold):

    # distances = np.abs(points @ normal - d)
    # print(distances)

    distances = np.abs(np.dot(points, normal) - d)     

    # vectors = d - points
    # distances = np.abs(np.dot(normal, np.transpose(vectors)) / np.linalg.norm(normal))

    return distances < threshold          # returns a True/False array



def detect_floor(PC, threshold=0.01, max_iter=1000):

    # Firstly Getting valid cloud
    H, W, _ = PC.shape
    cloud = PC.reshape((-1, 3)) # flatten (H, W, 3) → (N, 3)

    # Ignore invalid measurements
    valid_mask = cloud[:, 2] != 0       # ignore z == 0
    valid_pts = cloud[valid_mask]

    print(f"Total pixels : {H * W}")
    print(f"Valid points : {valid_mask.sum()}")

    normal, d, inliers_valid = ransac_plane(valid_pts, threshold, max_iter)


    # Map back from valid-only array to the full (H * W) grid
    inliers_full = np.zeros(H * W, dtype=bool)
    inliers_full[valid_mask] = inliers_valid

    floor_mask = inliers_full.reshape(H, W)
    return normal, d, floor_mask


def find_box_top(PC, floor_mask, threshold=0.01, max_iterations=1000):
    """
    Remove floor points, then run RANSAC on what remains.
    the SECOND largest plane = box top.
    """
    print("\n[Step 5] Finding BOX TOP with RANSAC ...")

    H, W, _ = PC.shape
    cloud = PC.reshape((-1, 3)) # flatten (H, W, 3) → (N, 3)
    floor_flat = floor_mask.reshape(-1)

    # Non-floor, valid points only
    keep = (~floor_flat) & (cloud[:, 2] != 0)
    kept_pts = cloud[keep]

    print(f"Non-floor valid points: {kept_pts.shape[0]}")

    normal, d, inliers_valid = ransac_plane(kept_pts, threshold, max_iterations)

    print(f"Box top normal  : {np.round(normal, 3)}")
    print(f"Box top inliers : {inliers_valid.sum()} px")

    inliers_full = np.zeros(H * W, dtype=bool)
    inliers_full[keep] = inliers_valid

    box_mask = inliers_full.reshape(H, W)
    return normal, d, box_mask



def box_top(PC, floor_mask, floor_n, floor_d, threshold=0.01, max_iterations=1000):

    H, W, _ = PC.shape
    cloud = PC.reshape((-1, 3)) # flatten (H, W, 3) → (N, 3)
    floor_flat = floor_mask.reshape(-1)

    # Non-floor, valid points only
    keep = (cloud[:, 2] != 0)
    kept_pts = cloud[keep]

    n_floor_u = floor_n / np.linalg.norm(floor_n)
    signed_valid = np.dot(kept_pts, n_floor_u) - floor_d

    if np.median(signed_valid) < 0:
        floor_n, floor_d = -floor_n, -floor_d
        n_floor_u = -n_floor_u
        signed_valid = -signed_valid  # keep consistent

    # (1) Remove floor geometrically using distance to the plane; then keep only "above".
    signed_all = np.dot(cloud, n_floor_u) - floor_d
    floor_remove_eps = max(threshold * 1.5, threshold + 1e-9)

    keep_mask = keep & (np.abs(signed_all) > floor_remove_eps) & (signed_all > 0)
    pts_keep = cloud[keep_mask]

    normal, d, inliers_valid = ransac_plane(pts_keep, threshold, max_iterations)

    box_mask_all = np.zeros(H*W, dtype=bool)
    keep_idx = np.where(keep_mask)[0]
    box_mask_all[keep_idx[inliers_valid]] = True
    box_mask = box_mask_all.reshape(H, W)

    return normal, d, box_mask


# =============================================================================
# STEP 4 – MORPHOLOGICAL FILTERING
# =============================================================================
# The raw floor mask has two problems:
#   1. Small holes (black dots) inside the floor region
#   2. Stray white pixels outside the floor
#
# We fix them using the lecture's morphological operators:
#
#   CLOSING  = dilate then erode  → fills holes in the foreground
#   OPENING  = erode then dilate  → removes small stray blobs

def clean_mask(floor_mask, label=""):
    print(f"Cleaning {label} mask: Closing (fill holes) then Opening (remove noise) ...")

    # struct = ndimage.generate_binary_structure(2, 2)   # 8-connectivity
    #mask = ndimage.binary_closing(floor_mask,  structure=struct, iterations=5)
   # mask = ndimage.binary_opening(mask,structure=struct, iterations=3)
    
    floor_clean = ndimage.binary_closing(floor_mask, structure=np.ones((3,3)))
    floor_clean = ndimage.binary_opening(floor_clean, structure=np.ones((5,5)))
    floor_clean = ndimage.binary_fill_holes(floor_clean)

    return floor_clean


def show_floor_masks(raw_mask, clean_mask_result, example_num):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Morphological Filtering", fontsize=11)
    for ax, mask, title in zip(
            axes,
            [raw_mask, clean_mask_result],
            ["Floor Mask – Raw RANSAC", "Floor Mask – After Closing + Opening"]):
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        rgb[mask]  = [0,   0, 200]    # blue = floor
        rgb[~mask] = [120, 0,   0]    # dark red = not floor
        ax.imshow(rgb); ax.set_title(title); ax.axis("off")
    
    plt.tight_layout()
    save_path = f"Results/step4_floor_mask_example{example_num}.png"
    plt.savefig(save_path, dpi=120)
    plt.show()
    print("Saved: step4_floor_mask.png")

# =============================================================================
# STEP 5 – LARGEST CONNECTED COMPONENT 
# =============================================================================
# The box top mask still contains noise from walls or background.
def keep_largest_component(box_mask):
    """Label all blobs, keep only the biggest one."""
     # largest connected component on box mask
    labeled, num = ndimage.label(box_mask)
    if num > 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        keep_label = int(sizes.argmax())
        box_top_cc = (labeled == keep_label)
    else:
        box_top_cc = box_mask
        
    return box_top_cc

def visualise_box_mask(box_mask, box_top_mask , example_num):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, m, title in zip(axes,
                             [box_mask, box_top_mask],
                             ["Box Top (all inliers)", "Box Top (largest component)"]):
        display = np.zeros((*m.shape, 3), dtype=np.uint8)
        display[m]  = [0,   0, 200]
        display[~m] = [150, 0,   0]
        ax.imshow(display)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    save_path = f"Results/step5_box_top_mask_example{example_num}.png"
    plt.savefig(save_path, dpi=120)
    plt.show()
    print("Saved: box_top_mask.png")

# ─────────────────────────────────────────────
# 6. DIMENSION ESTIMATION
# ─────────────────────────────────────────────

def plane_to_plane_distance(floor_n, floor_d, n_box, d_box):
    """
    Distance between two parallel planes sharing the same normal:
       n·x = floor_d  and  n·x = d_box
    Distance = |d_box - floor_d| / ||n||   (normal is already unit length)
    """
    # Ensure normals point roughly the same way
    if np.dot(floor_n, n_box) < 0:
        n_box, d_box = -n_box, -d_box
    # distance between parallel planes with unit normals
    return abs(d_box - floor_d)


# =============================================================================
# MEASURE DIMENSIONS
# =============================================================================
# HEIGHT:  distance between two (near-parallel) planes = |d_top − floor_d|
#          (works because both normals are unit vectors after refinement)
#
# LENGTH / WIDTH:
#   Project box top 3D points onto the fitted plane, then fit a PCA-oriented
#   rectangle.  This gives the true dimensions regardless of camera tilt.

def _plane_uv_frame(normal):
    """
    Build two unit vectors (u, v) that lie IN the plane with the given normal.
    Together (u, v, normal) form a complete orthonormal frame.
    """
    normal = normal / np.linalg.norm(normal)
    # Choose a helper vector not parallel to normal
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v  = np.cross(normal, u)
    return u, v


def _pca_rectangle_dims(uv_coords):
    """
    Given 2D projected coordinates (N, 2), fit a PCA-oriented bounding rectangle.
    Returns (length, width) where length >= width.
    """
    centre = uv_coords.mean(axis=0)
    centred = uv_coords - centre
    cov = (centred.T @ centred) / len(centred)
    _, eigvecs = np.linalg.eigh(cov)           # ascending eigenvalues
    ax_long  = eigvecs[:, 1]   # principal axis  (longest dimension)
    ax_short = eigvecs[:, 0]   # secondary axis  (shortest dimension)

    # Project all points onto each axis
    proj_long  = centred @ ax_long
    proj_short = centred @ ax_short

    length = float(proj_long.max()  - proj_long.min())
    width  = float(proj_short.max() - proj_short.min())

    # Build 4 corners in axis-aligned space then rotate back to UV frame
    corners_local = np.array([
        [proj_long.min(), proj_short.min()],
        [proj_long.max(), proj_short.min()],
        [proj_long.max(), proj_short.max()],
        [proj_long.min(), proj_short.max()],
    ])
    corners_uv = np.array([
        centre + c[0] * ax_long + c[1] * ax_short
        for c in corners_local
    ])

    # Sort corners clockwise using arctan2 so the polygon never self-intersects
    centroid = corners_uv.mean(axis=0)
    angles   = np.arctan2(corners_uv[:, 1] - centroid[1],
                           corners_uv[:, 0] - centroid[0])
    corners_uv = corners_uv[np.argsort(angles)]

    return corners_uv, length, width


# =============================================================================
# STEP 7 – MEASURE THE BOX
# =============================================================================
def box_height(floor_normal, floor_d, box_normal, box_d):
    # Ensure normals point roughly the same way
    if np.dot(floor_normal, box_normal) < 0:
        box_normal, box_d = -box_normal, -box_d
    # distance between parallel planes with unit normals
    return abs(box_d - floor_d)

def measure(PC, box_mask, box_normal, box_d):
    H, W, _ = PC.shape
    flat = PC.reshape(-1, 3)   # (H*W, 3)

    # Gather 3D points on the box top
    rows, cols = np.where(box_mask)
    pts_3d = PC[rows, cols, :]
    pts_3d = pts_3d[pts_3d[:, 2] != 0]        # remove invalid

    # Anchor on the plane (closest point to origin)
    normal_u = box_normal / np.linalg.norm(box_normal)
    anchor = normal_u * box_d

    # Project onto in-plane UV frame
    u, v = _plane_uv_frame(box_normal)
    offset = pts_3d - anchor
    uv = np.column_stack([offset @ u, offset @ v])   # (N, 2)

    # uv = np.c_[offset @ u, offset @ v]   # (N, 2) in-plane coords
    
    # Remove 1% outliers before rectangle fitting
    lo, hi  = np.percentile(uv, [1, 99], axis=0)
    #keep = (uv[:,0] >= lo[0]) & (uv[:,0] <= hi[0]) & (uv[:,1] >= lo[1]) & (uv[:,1] <= hi[1])
    keep = np.all((uv >= lo) & (uv <= hi), axis=1)

    UVc = uv[keep]

    corners_uv, length, width = _pca_rectangle_dims(UVc)
    print(f"Length  : {length*100:.1f} cm \nWidth : {width*100:.1f} cm")

    
    # Convert UV corners back to 3D world coordinates
    corners_3d = np.array([anchor + c[0]*u + c[1]*v for c in corners_uv])

    pixel_idxs = []
    for corner in corners_3d:
        dist_sq = np.sum((flat - corner) ** 2, axis=1)
        pixel_idxs.append(int(np.argmin(dist_sq)))

    pixel_idxs = np.array(pixel_idxs)
    rows = pixel_idxs // W
    cols = pixel_idxs  % W
    corners_rc = np.column_stack([rows, cols])   # (4, 2)

    return length, width, corners_rc

# =============================================================================
# STEP 8 – FINAL VISUALISATION
# =============================================================================
# Matches Figure 4 in the exercise sheet: floor, box and box corners shown
# as a colour-coded 2D image with measurements printed on it.

def show_final(A, PC, floor_mask, box_mask, height, length, width, corners_rc, example_num):

    H, W, _ = PC.shape
    # Build colour map image
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[floor_mask]                = [30,  180,  30]   # green  = floor
    rgb[box_mask]                  = [30,   30, 220]   # blue   = box top
    rgb[~floor_mask & ~box_mask]   = [100,   0,   0]   # dark red = other

    # Corner rows and cols – close the polygon by repeating first corner
    r_corners = corners_rc[:, 0]
    c_corners = corners_rc[:, 1]
    r_closed  = np.append(r_corners, r_corners[0])
    c_closed  = np.append(c_corners, c_corners[0])

    # Measurement text
    meas_text = (f"Height : {height * 100:.1f} cm\n"
                 f"Length : {length * 100:.1f} cm\n"
                 f"Width  : {width  * 100:.1f} cm")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Final Result – Box Detection", fontsize=13)

    # ── Left panel: colour-coded mask ─────────────────────────────────────────
    axes[0].imshow(rgb)
    axes[0].set_title("Green=floor Blue=box top  Red=other", fontsize=10)

    # Draw closed rectangle connecting the 4 corners
    axes[0].plot(c_closed, r_closed, "w-", linewidth=2, label="box corners")
    # Mark each corner with a cross
    for r, c in zip(r_corners, c_corners):
        axes[0].plot(c, r, "w+", markersize=14, markeredgewidth=2.5)

    # Measurement box in top-left corner
    axes[0].text(6, 16, meas_text, color="white", fontsize=10, verticalalignment="top", bbox=dict(facecolor="black", alpha=0.65, pad=4))
    axes[0].axis("off")

    # ── Right panel: amplitude + contour + rectangle ──────────────────────────
    axes[1].imshow(A, cmap="gray")
    axes[1].set_title("Amplitude image with box top boundary", fontsize=10)

    # Cyan contour: edge pixels of the box top mask
    contour = ndimage.binary_dilation(box_mask) ^ box_mask
    cy, cx  = np.where(contour)
    # axes[1].scatter(cx, cy, s=1, c="cyan", linewidths=0)
    # axes[1].plot(np.r_corners_[c_corners, c_corners[0]], np.r_corners_[r_corners, r_corners[0]], '-')


    # Yellow rectangle through the 4 3D-refined corners
    axes[1].plot(c_closed, r_closed, "y-", linewidth=2, label="corners")
    for r, c in zip(r_corners, c_corners):
        axes[1].plot(c, r, "y+", markersize=12, markeredgewidth=2)

    axes[1].scatter(c_corners, r_corners, s=30)
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].axis("off")

    plt.tight_layout()
    save_path = f"Results/step8_final_result_example{example_num}.png"
    plt.savefig(save_path, dpi=120)
    plt.show()
    print("Saved: step8_final_result.png")


def corners_pixels_from_3d(PC, corners_3d):
    """
    For each 3D corner, find the nearest pixel in the point cloud by vectorised nearest-neighbour.
    Returns (4, 2) array of [row, col] pixel positions.
    """
    H, W, _ = PC.shape
    flat = PC.reshape(-1, 3)   # (H*W, 3)

    pixel_idxs = []
    for corner in corners_3d:
        dist_sq = np.sum((flat - corner) ** 2, axis=1)
        pixel_idxs.append(int(np.argmin(dist_sq)))

    pixel_idxs = np.array(pixel_idxs)
    rows = pixel_idxs // W
    cols = pixel_idxs  % W
    return np.column_stack([rows, cols])   # (4, 2)

def main():
    # Trigger : python ex_01.py example1kinect.mat 1
    filepath = sys.argv[1] if len(sys.argv) > 1 else "./data/example1kinect.mat"
    number = sys.argv[2] if len(sys.argv) > 1 else 1
    print("\n" + "=" * 60)
    print(f"Loading: {filepath} number {number}")
   
    # 1. 
    # Load the Data
    Amp, Dist, PC = load_example_data(filepath, number)
    print(f"Amp shape: {Amp.shape}")
    print(f"Dist shape: {Dist.shape}")
    print(f"PC shape: {PC.shape}")
    #visualise the data
    visualise_data(Amp, Dist, PC, number, 5)

    # 2.Apply median filter
    Dist_filtered = apply_median_filter(Dist, number, kernel_size=5)

    # 3. RANSAC
    threshold = 0.015
    max_iter = 1000

    #3.1 floor detection
    floor_normal, floor_d, floor_mask_raw = detect_floor(PC, threshold, max_iter)
    print(f"Floor normal: {floor_normal}")
    print(f"Floor d: {floor_d:.4f}")
    print(f"Floor inliers: {floor_mask_raw.sum()} px")
    floor_mask = clean_mask(floor_mask_raw, label="floor")
    show_floor_masks(floor_mask_raw, floor_mask, number)

    ## 1.2 box top detection
    box_normal, box_d, box_mask_raw = box_top(PC, floor_mask, floor_normal, floor_d,threshold=0.010, max_iterations=500)
    print(f"Box normal: {box_normal}")
    print(f"Box d: {box_d:.4f}")
    print(f"Box inliers (raw): {box_mask_raw.sum()} px")

    # Keep only largest connected component
    box_top_mask = keep_largest_component(box_mask_raw)
    print(f"Box top (largest CC): {box_top_mask.sum()} px")
    visualise_box_mask(box_mask_raw, box_top_mask, number)


    # Dimension estimation
    print("\nEstimating box dimensions …")
    height = box_height(floor_normal, floor_d, box_normal, box_d)
    print(f"Height  : {height*100:.1f} cm")


    length, width, corners_rc = measure(PC, box_top_mask, box_normal, box_d)

     # Show final result
    show_final(Amp, PC, floor_mask, box_top_mask, height, length, width, corners_rc, number)


    print("\n" + "=" * 60)
    print(" DONE")
    print("=" * 60)

    exit(0)
    exit(0)

if __name__ == "__main__":
    main()
