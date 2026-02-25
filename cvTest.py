import cv2
import numpy as np
import argparse
from pathlib import Path

def classify_white_bgr(img_bgr, roi=None,
                       L_min=200,  # brightness threshold (0..255)
                       ab_tol=12,  # how close a,b must be to neutral (128)
                       white_pct_thresh=0.55,  # fraction of ROI pixels that must be white
                       blur_ksize=5):
    """
    Returns: (label:str, white_fraction:float, debug_masks:dict)
    label in {"WHITE", "NOT_WHITE"}
    """
    h, w = img_bgr.shape[:2]

    if roi is None:
        # Default ROI: centered box (good if you place item in the middle)
        x1, y1 = int(0.20 * w), int(0.20 * h)
        x2, y2 = int(0.80 * w), int(0.80 * h)
    else:
        x1, y1, x2, y2 = roi

    roi_bgr = img_bgr[y1:y2, x1:x2].copy()

    # Optional denoise to reduce speckle
    if blur_ksize and blur_ksize >= 3:
        roi_bgr = cv2.GaussianBlur(roi_bgr, (blur_ksize, blur_ksize), 0)

    # Convert to LAB
    roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(roi_lab)

    # Neutral in OpenCV LAB is around A=128, B=128
    near_neutral = (np.abs(A.astype(np.int16) - 128) <= ab_tol) & (np.abs(B.astype(np.int16) - 128) <= ab_tol)
    bright = (L >= L_min)

    white_mask = (near_neutral & bright).astype(np.uint8) * 255

    white_fraction = float(np.count_nonzero(white_mask)) / float(white_mask.size)
    label = "WHITE" if white_fraction >= white_pct_thresh else "NOT_WHITE"

    debug = {
        "roi": (x1, y1, x2, y2),
        "white_mask": white_mask
    }
    return label, white_fraction, debug


def run_webcam(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {index}")

    # Tunables (start here, then adjust)
    L_min = 200
    ab_tol = 12
    white_pct_thresh = 0.55

    print("Press q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        label, frac, dbg = classify_white_bgr(
            frame,
            roi=None,
            L_min=L_min,
            ab_tol=ab_tol,
            white_pct_thresh=white_pct_thresh,
            blur_ksize=5
        )

        x1, y1, x2, y2 = dbg["roi"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        txt = f"{label} | white_frac={frac:.2f} | L_min={L_min} ab_tol={ab_tol} thresh={white_pct_thresh}"
        cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # Show mask for debugging
        mask_vis = cv2.cvtColor(dbg["white_mask"], cv2.COLOR_GRAY2BGR)
        mask_vis = cv2.resize(mask_vis, (frame.shape[1], frame.shape[0]))

        cv2.imshow("Laundry Sorter - Frame", frame)
        cv2.imshow("Laundry Sorter - White Mask", mask_vis)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_images(input_path):
    p = Path(input_path)
    if p.is_dir():
        paths = sorted([*p.glob("*.jpg"), *p.glob("*.jpeg"), *p.glob("*.png"), *p.glob("*.bmp")])
    else:
        paths = [p]

    if not paths:
        raise RuntimeError("No images found.")

    for img_path in paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skipping unreadable: {img_path}")
            continue

        label, frac, dbg = classify_white_bgr(img)
        print(f"{img_path.name}: {label} (white_frac={frac:.3f})")

        x1, y1, x2, y2 = dbg["roi"]
        vis = img.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{label} ({frac:.2f})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        mask = dbg["white_mask"]
        mask_big = cv2.resize(mask, (vis.shape[1], vis.shape[0]))

        cv2.imshow("Image", vis)
        cv2.imshow("White Mask", mask_big)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--webcam", action="store_true", help="Run live webcam mode")
    ap.add_argument("--cam", type=int, default=0, help="Webcam index")
    ap.add_argument("--input", type=str, help="Image file or directory")
    args = ap.parse_args()

    if args.webcam:
        run_webcam(args.cam)
    elif args.input:
        run_images(args.input)
    else:
        print("Usage:")
        print("  python white_vs_not.py --webcam")
        print("  python white_vs_not.py --input path/to/image_or_folder")