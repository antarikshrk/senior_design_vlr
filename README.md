CV Parameter Description:
L_min
This is the biggest lever for “how bright does it need to be to count as white.”
Raise it if light gray, beige, or dim objects are being called white too often.
Lower it if actual white clothes in normal room lighting are getting missed.

A good tuning range is usually around 180–220.

ab_tol
This controls how color-neutral something must be.
Lower it if pale blue, cream, yellowish white, or colored fabric is being called white.
Raise it if real white fabric under warm/cool lighting is getting rejected because the camera tint shifts A/B a bit.

A good range is about 8–20.

white_pct_thresh
This decides how much of the ROI must look white before the frame is labeled white.
Raise it if small white patches are causing a full “WHITE” result.
Lower it if a white shirt with folds, shadows, logos, or wrinkles is being labeled “NOT WHITE.”

A good range is about 0.40–0.70.

roi_pct
This matters a lot if the background is affecting the result.
Increase it if the edges of the frame include table, floor, bin, or background clutter.
Decrease it if the clothing item is large and you are accidentally cropping out too much of it.

A good range is about 0.10–0.30.

blur_ksize
This is a cleanup parameter, not a primary classifier parameter.
Increase slightly if noise or texture causes flickering.
Reduce it if blur is washing out useful detail.
