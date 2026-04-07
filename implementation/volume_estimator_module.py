import os
import numpy as np
import cv2
from pathlib import Path
from scipy.optimize import linear_sum_assignment


# ============================================================
# Shape correction factors (food-type aware)
# ============================================================
SHAPE_CORRECTION = {
    "rice"     : 1.00,
    "soup"     : 1.00,
    "salad"    : 1.00,
    "apple"    : 0.52,
    "orange"   : 0.52,
    "egg"      : 0.50,
    "meatball" : 0.52,
    "pizza"    : 0.25,
    "flatbread": 0.20,
    "naan"     : 0.20,
    "burger"   : 0.65,
    "sandwich" : 0.60,
    "cake"     : 0.70,
    "default"  : 1.00,
}


class VolumeEstimator:
    """
    A module for estimating the volume of segmented food items
    using a two-view voxel-grid pipeline.

    Inputs required per estimation:
      - Two RGB images of the same food from different angles
        (e.g. side view + top view)
      - Two corresponding .npy segmentation masks (one per image)
      - pixel_length_mm: real-world size of one pixel in mm
        (calibration value from Module 3)

    Typical usage:
        estimator = VolumeEstimator(pixel_length_mm=0.5)
        result = estimator.estimate_by_paths(
            image1_path  = "images/food_side.jpg",
            image2_path  = "images/food_top.jpg",
            mask1_path   = "masks/food_side_mask.npy",
            mask2_path   = "masks/food_top_mask.npy",
            food_type    = "rice",
        )
        print(result['volume_ml'], "mL")
    """

    def __init__(
        self,
        pixel_length_mm: float = 0.5,
        voxel_size_mm: float = 5.0,
    ):
        """
        Initialize the VolumeEstimator.

        Args:
            pixel_length_mm : Real-world length of one pixel in mm.
                              Get this value from Module 3 (thumb calibration).
            voxel_size_mm   : Side length of each voxel in mm.
                              Smaller = more precise but slower. Default 5.0 mm.
        """
        self.pixel_length_mm = pixel_length_mm
        self.voxel_size_mm   = voxel_size_mm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_by_paths(
        self,
        image1_path: str,
        image2_path: str,
        mask1_path: str,
        mask2_path: str,
        food_type: str = "",
    ) -> dict:
        """
        Estimate volume for ONE food item given file paths.

        Args:
            image1_path : Path to the first image  (e.g. side view)
            image2_path : Path to the second image (e.g. top view)
            mask1_path  : Path to the .npy mask for image 1
            mask2_path  : Path to the .npy mask for image 2
            food_type   : Food class name for shape correction
                          (e.g. "rice", "burger"). Leave "" for default.

        Returns:
            dict with keys:
              volume_mm3, volume_cm3, volume_ml, volume_l,
              pixel_count, num_points, num_voxels,
              voxel_size_mm, shape_correction, food_type
        """
        img1  = self._load_image(image1_path)
        img2  = self._load_image(image2_path)
        mask1 = self._load_npy_mask(mask1_path)
        mask2 = self._load_npy_mask(mask2_path)
        return self.estimate(img1, img2, mask1, mask2, food_type)

    def estimate(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask1: np.ndarray,
        mask2: np.ndarray,
        food_type: str = "",
    ) -> dict:
        """
        Estimate volume for ONE food item given already-loaded arrays.

        Args:
            img1, img2   : RGB images as numpy arrays (H×W×3 uint8)
            mask1, mask2 : Segmentation masks as numpy arrays (H×W uint8)
            food_type    : Food class name for shape correction

        Returns:
            dict with volume results and diagnostics (same as estimate_by_paths)
        """
        return self._calculate_volume_voxel(img1, img2, mask1, mask2, food_type)

    def estimate_multiple_items(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        segments1: list,
        segments2: list,
    ) -> dict:
        """
        Estimate volume for ALL food items in a pair of images.

        segments1 / segments2 are lists of dicts — each dict produced by
        parse_npy_segments() or parse_yolo_segments():
            {
                'class_id': str,
                'mask'    : np.ndarray (H×W uint8),
                'bbox'    : (x, y, w, h),
            }

        Returns:
            dict mapping food_label → result_dict
        """
        return self._calculate_volume_all_items(
            img1, img2, segments1, segments2,
            self.pixel_length_mm, self.voxel_size_mm,
        )

    # ------------------------------------------------------------------
    # Mask / image parsing helpers  (for use by integration code)
    # ------------------------------------------------------------------

    def load_image(self, path: str) -> np.ndarray:
        """Load an image from disk and return as RGB numpy array."""
        return self._load_image(path)

    def load_npy_mask(self, path: str) -> np.ndarray:
        """Load a .npy segmentation mask and return as binary uint8 mask."""
        return self._load_npy_mask(path)

    def parse_npy_segments(
        self,
        image: np.ndarray,
        npy_paths: list,
        class_ids: list = None,
    ) -> list:
        """
        Build the segment-dict list from a list of .npy mask file paths.

        Args:
            image     : Reference RGB image (used for H/W alignment)
            npy_paths : List of .npy file paths, one per food item
            class_ids : Optional list of class-id strings (same length as npy_paths).
                        If None, ids are auto-assigned '0', '1', ...

        Returns:
            List of segment dicts: [{'class_id', 'mask', 'bbox'}, ...]
        """
        h, w   = image.shape[:2]
        result = []
        for i, path in enumerate(npy_paths):
            cid = str(class_ids[i]) if class_ids and i < len(class_ids) else str(i)
            try:
                raw_mask = self._load_npy_mask(path)
            except FileNotFoundError as e:
                print(f"⚠️  Skipping mask {path}: {e}")
                continue
            if raw_mask.shape != (h, w):
                raw_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            clean = self._preprocess_mask(raw_mask)
            bbox  = self._bbox_from_mask(clean)
            result.append({"class_id": cid, "mask": clean, "bbox": bbox})
            print(f"   [{cid}] bbox={bbox}, food_pixels={np.count_nonzero(clean):,}")
        print(f"\n✅ parse_npy_segments: {len(result)} item(s) ready")
        return result

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_npy_mask(self, path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Mask file not found: {path}")
        data   = np.load(path, allow_pickle=True)
        mask   = np.asarray(data)
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        binary = (mask > 0).astype(np.uint8) * 255
        # Auto-flip inverted masks (SAM sometimes stores background=True)
        if np.count_nonzero(binary) / max(binary.size, 1) > 0.50:
            binary = cv2.bitwise_not(binary)
            print(f"   ⚠️  Mask auto-flipped (was mostly foreground)")
        return binary

    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        h, w   = closed.shape
        flood  = closed.copy()
        cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
        holes_filled = closed | cv2.bitwise_not(flood)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes_filled, 8)
        if num_labels <= 1:
            return holes_filled
        areas    = stats[1:, cv2.CC_STAT_AREA]
        min_area = max(areas.max() * 0.05, 100)
        clean    = np.zeros_like(holes_filled)
        for idx, area in enumerate(areas, start=1):
            if area >= min_area:
                clean[labels == idx] = 255
        return clean

    def _build_point_cloud(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask1: np.ndarray,
        mask2: np.ndarray,
        max_points: int = 5000,
    ) -> np.ndarray:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        h, w  = gray1.shape
        try:
            sift = cv2.SIFT_create(nfeatures=2000)
        except AttributeError:
            sift = cv2.xfeatures2d.SIFT_create(nfeatures=2000)
        kp1, des1 = sift.detectAndCompute(gray1, mask1)
        kp2, des2 = sift.detectAndCompute(gray2, mask2)
        MIN_GOOD  = 8
        good      = []
        if des1 is not None and des2 is not None and len(kp1) >= MIN_GOOD and len(kp2) >= MIN_GOOD:
            flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
            raw   = flann.knnMatch(des1, des2, k=2)
            good  = [m for m, n in raw if m.distance < 0.75 * n.distance]
        if len(good) >= MIN_GOOD:
            pts1   = np.float32([kp1[m.queryIdx].pt for m in good])
            pts2   = np.float32([kp2[m.trainIdx].pt for m in good])
            focal  = float(max(h, w))
            K      = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]], dtype=np.float64)
            E, em  = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, pm = cv2.recoverPose(E, pts1, pts2, K, mask=em)
            p1i    = pts1[pm.ravel() > 0]
            p2i    = pts2[pm.ravel() > 0]
            P1     = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P2     = K @ np.hstack([R, t])
            pts4d  = cv2.triangulatePoints(P1, P2, p1i.T, p2i.T)
            pts3d  = (pts4d[:3] / pts4d[3]).T
            valid  = pts3d[:, 2] > 0
            pts3d  = pts3d[valid]
            p1i    = p1i[valid]
            xi     = np.clip(p1i[:, 0].astype(int), 0, w - 1)
            yi     = np.clip(p1i[:, 1].astype(int), 0, h - 1)
            pts3d  = pts3d[mask1[yi, xi] > 0]
            print(f"   ✅ Triangulated {len(pts3d):,} pts from {len(good)} matches")
        else:
            print(f"   ⚠️  Only {len(good)} matches — single-view fallback")
            ys, xs = np.where(mask1 > 0)
            if len(xs) == 0:
                return np.zeros((0, 3), dtype=np.float32)
            depth_estimate = (ys.max() - ys.min()) * 0.5
            pts3d = np.column_stack([xs.astype(np.float32), ys.astype(np.float32),
                                     np.full(len(xs), depth_estimate, np.float32)])
        pts3d_mm = pts3d.astype(np.float32) * self.pixel_length_mm
        if len(pts3d_mm) > max_points:
            pts3d_mm = pts3d_mm[np.random.choice(len(pts3d_mm), max_points, replace=False)]
        return pts3d_mm

    def _remove_outliers(self, points: np.ndarray, k: int = 20, std_ratio: float = 2.0) -> np.ndarray:
        if len(points) < k + 1:
            return points
        diff        = points[:, None, :] - points[None, :, :]
        dists       = np.sqrt((diff ** 2).sum(axis=-1))
        dists.sort(axis=1)
        mean_k_dist = dists[:, 1:k + 1].mean(axis=1)
        threshold   = mean_k_dist.mean() + std_ratio * mean_k_dist.std()
        keep        = mean_k_dist <= threshold
        print(f"   Outlier removal: {len(points)} → {keep.sum()} pts kept")
        return points[keep]

    def _voxelize(self, points_mm: np.ndarray) -> int:
        if len(points_mm) == 0:
            return 0
        shifted  = points_mm - points_mm.min(axis=0)
        indices  = np.floor(shifted / self.voxel_size_mm).astype(np.int32)
        occupied = {(ix, iy, iz) for ix, iy, iz in indices}
        return len(occupied)

    def _get_shape_correction(self, food_type: str) -> float:
        ft = food_type.lower() if food_type else ""
        for keyword, factor in SHAPE_CORRECTION.items():
            if keyword in ft:
                return factor
        return SHAPE_CORRECTION["default"]

    def _calculate_volume_voxel(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask1: np.ndarray,
        mask2: np.ndarray,
        food_type: str = "",
    ) -> dict:
        print("\n" + "=" * 60)
        print(f"VOXEL-GRID VOLUME ESTIMATION  [{food_type or 'unknown'}]")
        print("=" * 60)

        print("\n[1/5] Preprocessing masks...")
        clean1      = self._preprocess_mask(mask1)
        clean2      = self._preprocess_mask(mask2)
        pixel_count = int(np.count_nonzero(clean1))
        print(f"   Mask1 food pixels: {pixel_count:,}")
        print(f"   Mask2 food pixels: {np.count_nonzero(clean2):,}")

        print("\n[2/5] Building 3-D point cloud...")
        points_mm = self._build_point_cloud(img1, img2, clean1, clean2)
        print(f"   Raw cloud: {len(points_mm):,} points")
        if len(points_mm) == 0:
            print("❌ Empty point cloud — check inputs")
            return {}

        print("\n[3/5] Removing outliers...")
        points_mm = self._remove_outliers(points_mm)

        print(f"\n[4/5] Voxelising (voxel size = {self.voxel_size_mm} mm)...")
        num_voxels = self._voxelize(points_mm)
        print(f"   Occupied voxels: {num_voxels:,}")

        print("\n[5/5] Computing volume...")
        shape_factor = self._get_shape_correction(food_type)
        volume_mm3   = num_voxels * (self.voxel_size_mm ** 3) * shape_factor

        result = {
            "pixel_count"     : pixel_count,
            "num_points"      : len(points_mm),
            "num_voxels"      : num_voxels,
            "voxel_size_mm"   : self.voxel_size_mm,
            "shape_correction": shape_factor,
            "volume_mm3"      : volume_mm3,
            "volume_cm3"      : volume_mm3 / 1_000,
            "volume_ml"       : volume_mm3 / 1_000,
            "volume_l"        : volume_mm3 / 1_000_000,
            "food_type"       : food_type or "unknown",
        }

        print(f"\n{'=' * 60}")
        print(f"  Food type        : {result['food_type']}")
        print(f"  Pixel count      : {result['pixel_count']:,}")
        print(f"  Point cloud size : {result['num_points']:,}")
        print(f"  Occupied voxels  : {result['num_voxels']:,}")
        print(f"  Voxel size       : {result['voxel_size_mm']} mm")
        print(f"  Shape correction : {result['shape_correction']}")
        print(f"  Volume           : {result['volume_mm3']:,.2f} mm³")
        print(f"                   : {result['volume_cm3']:,.4f} cm³  /  {result['volume_ml']:,.4f} mL")
        print("=" * 60)
        return result

    def _calculate_volume_all_items(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        segments1: list,
        segments2: list,
        pixel_length_mm: float,
        voxel_size_mm: float,
    ) -> dict:
        print("\n" + "=" * 60)
        print("MULTI-ITEM VOLUME ESTIMATION")
        print("=" * 60)
        print(f"  Segments in image 1 : {len(segments1)}")
        print(f"  Segments in image 2 : {len(segments2)}")

        def group_by_class(segs):
            groups = {}
            for s in segs:
                groups.setdefault(s["class_id"], []).append(s)
            return groups

        groups1    = group_by_class(segments1)
        groups2    = group_by_class(segments2)
        all_results = {}

        for class_id, segs1 in groups1.items():
            if class_id not in groups2:
                print(f"\n⚠️  '{class_id}' found in image 1 but NOT image 2 — skipping")
                continue
            pairs = self._match_by_position(segs1, groups2[class_id])
            for i, (s1, s2) in enumerate(pairs):
                label = f"{class_id}_{i}" if len(pairs) > 1 else class_id
                print(f"\n▶  Processing item: {label}")
                result = self._calculate_volume_voxel(img1, img2, s1["mask"], s2["mask"], class_id)
                all_results[label] = result

        print("\n" + "=" * 60)
        print("SUMMARY — all items")
        print("=" * 60)
        print(f"  {'Item':<20} {'Volume (mL)':>12}")
        print(f"  {'-'*20} {'-'*12}")
        for label, r in all_results.items():
            if r:
                print(f"  {label:<20} {r['volume_ml']:>12.2f}")
        print("=" * 60)
        return all_results

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> tuple:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return (0, 0, 0, 0)
        x, y = int(xs.min()), int(ys.min())
        return (x, y, int(xs.max()) - x, int(ys.max()) - y)

    @staticmethod
    def _bbox_center(bbox: tuple) -> np.ndarray:
        x, y, w, h = bbox
        return np.array([x + w / 2.0, y + h / 2.0])

    def _match_by_position(self, segs1: list, segs2: list) -> list:
        if not segs1 or not segs2:
            return []
        cost = np.array([
            [np.linalg.norm(self._bbox_center(s1["bbox"]) - self._bbox_center(s2["bbox"]))
             for s2 in segs2]
            for s1 in segs1
        ])
        row_idx, col_idx = linear_sum_assignment(cost)
        return [(segs1[r], segs2[c]) for r, c in zip(row_idx, col_idx)]
