import cv2
import numpy as np
import argparse
from pathlib import Path

# ---------- Fabric classifier (loads if fabric_model.pth exists) ----------
_fabric_model = None
_fabric_classes = None
_fabric_transform = None

def _load_fabric_model(model_path="fabric_model.pth"):
    global _fabric_model, _fabric_classes, _fabric_transform
    p = Path(model_path)
    if not p.exists():
        return False
    try:
        import torch
        import torchvision.models as models
        import torchvision.transforms as T
        import torch.nn as nn

        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        classes = ckpt["classes"]
        num_classes = ckpt["num_classes"]

        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        _fabric_model = model
        _fabric_classes = classes
        _fabric_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return True
    except Exception as e:
        print(f"[fabric] Could not load model: {e}")
        return False


def classify_fabric(img_bgr, roi):
    """Returns (label, confidence) or (None, 0) if model not loaded."""
    if _fabric_model is None:
        return None, 0.0

    import torch
    x1, y1, x2, y2 = roi
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, 0.0

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = _fabric_transform(crop_rgb).unsqueeze(0)

    with torch.no_grad():
        logits = _fabric_model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        idx = probs.argmax().item()
        conf = probs[idx].item()

    return _fabric_classes[idx], conf
# -------------------------------------------------------------------------

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


def get_motion_roi(bg_gray, frame_gray, min_area=4000, dilate_iters=4):
    """
    Returns (x1, y1, x2, y2) bounding box around the largest moving region,
    or None if nothing significant moved.
    """
    diff = cv2.absdiff(bg_gray, frame_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # Fatten up the blobs so scattered pixels merge
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=dilate_iters)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Pick the largest contour
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None  # Too small — probably noise
    
    x, y, w, h = cv2.boundingRect(largest)
    
    # Add a little padding
    pad = 20
    h_img, w_img = bg_gray.shape
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad)
    y2 = min(h_img, y + h + pad)
    
    return x1, y1, x2, y2


def run_webcam(index=0):
    fabric_ready = _load_fabric_model()
    if fabric_ready:
        print(f"Fabric model loaded: {_fabric_classes}")
    else:
        print("No fabric_model.pth found — fabric detection disabled. Run train_fabric.py first.")

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {index}")

    L_min = 200
    ab_tol = 12
    white_pct_thresh = 0.55

    # Capture background (empty sorter)
    print("Capturing background — keep the sorter EMPTY for 2 seconds...")
    bg_frame = None
    for _ in range(30):  # warm up camera
        ok, frame = cap.read()
    ok, bg_frame = cap.read()
    bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.GaussianBlur(bg_gray, (21, 21), 0)
    print("Background captured. Toss in clothing now!")
    print("Press 'r' to reset background | 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray_blur = cv2.GaussianBlur(frame_gray, (21, 21), 0)

        motion_roi = get_motion_roi(bg_gray, frame_gray_blur)

        if motion_roi:
            roi = motion_roi
            roi_source = "MOTION"
        else:
            # Fall back to center box if nothing is moving
            h, w = frame.shape[:2]
            roi = (int(0.1*w), int(0.1*h), int(0.9*w), int(0.9*h))
            roi_source = "DEFAULT"

        label, frac, dbg = classify_white_bgr(
            frame,
            roi=roi,
            L_min=L_min,
            ab_tol=ab_tol,
            white_pct_thresh=white_pct_thresh,
            blur_ksize=5
        )

        x1, y1, x2, y2 = roi
        color = (0, 255, 0) if roi_source == "MOTION" else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, roi_source, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        fabric_label, fabric_conf = classify_fabric(frame, roi)

        txt = f"{label} | white_frac={frac:.2f} | L_min={L_min} ab_tol={ab_tol} thresh={white_pct_thresh}"
        cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        if fabric_label:
            ftxt = f"FABRIC: {fabric_label.upper()}  ({fabric_conf*100:.0f}%)"
            cv2.putText(frame, ftxt, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(frame, ftxt, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)

        mask_vis = cv2.cvtColor(dbg["white_mask"], cv2.COLOR_GRAY2BGR)
        mask_vis = cv2.resize(mask_vis, (frame.shape[1], frame.shape[0]))

        cv2.imshow("Laundry Sorter - Frame", frame)
        cv2.imshow("Laundry Sorter - White Mask", mask_vis)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('r'):
            ok, bg_frame = cap.read()
            bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
            bg_gray = cv2.GaussianBlur(bg_gray, (21, 21), 0)
            print("Background reset!")

    cap.release()
    cv2.destroyAllWindows()

def run_images(input_path):
    _load_fabric_model()
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
        fabric_label, fabric_conf = classify_fabric(img, dbg["roi"])
        print(f"{img_path.name}: {label} (white_frac={frac:.3f})"
              + (f" | fabric={fabric_label} ({fabric_conf*100:.0f}%)" if fabric_label else ""))

        x1, y1, x2, y2 = dbg["roi"]
        vis = img.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{label} ({frac:.2f})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        if fabric_label:
            cv2.putText(vis, f"FABRIC: {fabric_label.upper()} ({fabric_conf*100:.0f}%)", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2, cv2.LINE_AA)

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