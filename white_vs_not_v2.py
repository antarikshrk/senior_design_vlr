import cv2
import numpy as np
import argparse
from pathlib import Path

# Saved preset (2026-04-24): L_min=68, ab_tol=27, thresh=0.45, exposure=7, bright=127
PRESET_L_MIN    = 68
PRESET_AB_TOL   = 27
PRESET_THRESH   = 0.45
PRESET_EXPOSURE = 7
PRESET_BRIGHT   = 127

# ---------- Fabric classifier ----------
_fabric_model = None
_fabric_classes = None
_fabric_transform = None

def _load_fabric_model(model_path="fabric_modelv2.pth"):
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
# ----------------------------------------


def classify_white_bgr(img_bgr, roi=None, L_min=PRESET_L_MIN, ab_tol=PRESET_AB_TOL,
                       white_pct_thresh=PRESET_THRESH, blur_ksize=5):
    h, w = img_bgr.shape[:2]

    if roi is None:
        x1, y1 = int(0.20 * w), int(0.20 * h)
        x2, y2 = int(0.80 * w), int(0.80 * h)
    else:
        x1, y1, x2, y2 = roi

    roi_bgr = img_bgr[y1:y2, x1:x2].copy()

    if blur_ksize and blur_ksize >= 3:
        roi_bgr = cv2.GaussianBlur(roi_bgr, (blur_ksize, blur_ksize), 0)

    roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(roi_lab)

    near_neutral = (np.abs(A.astype(np.int16) - 128) <= ab_tol) & \
                   (np.abs(B.astype(np.int16) - 128) <= ab_tol)
    bright = (L >= L_min)

    white_mask = (near_neutral & bright).astype(np.uint8) * 255
    white_fraction = float(np.count_nonzero(white_mask)) / float(white_mask.size)
    label = "WHITE" if white_fraction >= white_pct_thresh else "NOT_WHITE"

    debug = {"roi": (x1, y1, x2, y2), "white_mask": white_mask}
    return label, white_fraction, debug


def _setup_camera(cap):
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # 1 = manual
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, PRESET_EXPOSURE - 13)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, PRESET_BRIGHT)


def run_webcam_color(cam_index):
    print("Mode: COLOR (white / not-white)")
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {cam_index}")
    _setup_camera(cap)

    win = "Laundry Sorter - Color Mode"
    cv2.namedWindow(win)
    cv2.createTrackbar("L_min",    win, PRESET_L_MIN,             255, lambda _: None)
    cv2.createTrackbar("ab_tol",   win, PRESET_AB_TOL,             50, lambda _: None)
    cv2.createTrackbar("thresh%",  win, int(PRESET_THRESH * 100), 100, lambda _: None)
    cv2.createTrackbar("exposure", win, PRESET_EXPOSURE,           12,
                       lambda v: cap.set(cv2.CAP_PROP_EXPOSURE, v - 13))
    cv2.createTrackbar("bright",   win, PRESET_BRIGHT,            255,
                       lambda v: cap.set(cv2.CAP_PROP_BRIGHTNESS, v))

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        L_min            = cv2.getTrackbarPos("L_min",   win)
        ab_tol           = cv2.getTrackbarPos("ab_tol",  win)
        white_pct_thresh = cv2.getTrackbarPos("thresh%", win) / 100.0

        h, w = frame.shape[:2]
        roi = (int(0.1 * w), int(0.1 * h), int(0.9 * w), int(0.9 * h))

        label, frac, dbg = classify_white_bgr(
            frame, roi=roi, L_min=L_min, ab_tol=ab_tol,
            white_pct_thresh=white_pct_thresh, blur_ksize=5
        )

        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

        txt = f"{label} | frac={frac:.2f} | L_min={L_min} ab_tol={ab_tol} thresh={white_pct_thresh:.2f}"
        cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        mask_vis = cv2.cvtColor(dbg["white_mask"], cv2.COLOR_GRAY2BGR)
        mask_vis = cv2.resize(mask_vis, (frame.shape[1], frame.shape[0]))

        cv2.imshow(win, frame)
        cv2.imshow("Laundry Sorter - White Mask", mask_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_webcam_fabric(cam_index):
    if _fabric_classes:
        print(f"Mode: FABRIC — classes: {_fabric_classes}")
    else:
        print("Mode: FABRIC — WARNING: model not loaded, fabric labels unavailable.")

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {cam_index}")
    _setup_camera(cap)

    win = "Laundry Sorter - Fabric Mode"
    cv2.namedWindow(win)
    cv2.createTrackbar("L_min",    win, PRESET_L_MIN,             255, lambda _: None)
    cv2.createTrackbar("ab_tol",   win, PRESET_AB_TOL,             50, lambda _: None)
    cv2.createTrackbar("thresh%",  win, int(PRESET_THRESH * 100), 100, lambda _: None)
    cv2.createTrackbar("exposure", win, PRESET_EXPOSURE,           12,
                       lambda v: cap.set(cv2.CAP_PROP_EXPOSURE, v - 13))
    cv2.createTrackbar("bright",   win, PRESET_BRIGHT,            255,
                       lambda v: cap.set(cv2.CAP_PROP_BRIGHTNESS, v))

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        L_min            = cv2.getTrackbarPos("L_min",   win)
        ab_tol           = cv2.getTrackbarPos("ab_tol",  win)
        white_pct_thresh = cv2.getTrackbarPos("thresh%", win) / 100.0

        h, w = frame.shape[:2]
        roi = (int(0.1 * w), int(0.1 * h), int(0.9 * w), int(0.9 * h))

        label, frac, dbg = classify_white_bgr(
            frame, roi=roi, L_min=L_min, ab_tol=ab_tol,
            white_pct_thresh=white_pct_thresh, blur_ksize=5
        )

        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

        fabric_label, fabric_conf = classify_fabric(frame, roi)

        txt = f"{label} | frac={frac:.2f} | L_min={L_min} ab_tol={ab_tol} thresh={white_pct_thresh:.2f}"
        cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        if fabric_label:
            ftxt = f"FABRIC: {fabric_label.upper()}  ({fabric_conf*100:.0f}%)"
            cv2.putText(frame, ftxt, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(frame, ftxt, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)

        mask_vis = cv2.cvtColor(dbg["white_mask"], cv2.COLOR_GRAY2BGR)
        mask_vis = cv2.resize(mask_vis, (frame.shape[1], frame.shape[0]))

        cv2.imshow(win, frame)
        cv2.imshow("Laundry Sorter - White Mask", mask_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_images_color(input_path):
    p = Path(input_path)
    paths = sorted([*p.glob("*.jpg"), *p.glob("*.jpeg"), *p.glob("*.png"), *p.glob("*.bmp")]) \
            if p.is_dir() else [p]
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

        cv2.imshow("Image", vis)
        cv2.imshow("White Mask", cv2.resize(dbg["white_mask"], (vis.shape[1], vis.shape[0])))
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def run_images_fabric(input_path):
    p = Path(input_path)
    paths = sorted([*p.glob("*.jpg"), *p.glob("*.jpeg"), *p.glob("*.png"), *p.glob("*.bmp")]) \
            if p.is_dir() else [p]
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

        cv2.imshow("Image", vis)
        cv2.imshow("White Mask", cv2.resize(dbg["white_mask"], (vis.shape[1], vis.shape[0])))
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Laundry sorter — choose a mode")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--color",  action="store_true", help="White / not-white detection only")
    mode.add_argument("--fabric", action="store_true", help="Fabric classification (loads model)")

    ap.add_argument("--cam",   type=int, default=0,               help="Webcam index (default 0)")
    ap.add_argument("--input", type=str,                           help="Image file or directory")
    ap.add_argument("--model", type=str, default="fabric_modelv2.pth", help="Fabric model checkpoint")
    args = ap.parse_args()

    if args.fabric:
        _load_fabric_model(args.model)

    if args.input:
        if args.color:
            run_images_color(args.input)
        else:
            run_images_fabric(args.input)
    else:
        if args.color:
            run_webcam_color(args.cam)
        else:
            run_webcam_fabric(args.cam)
