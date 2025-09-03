# eng.py
import cv2
import pyttsx3
import time
import mediapipe as mp
import os
import math

# -------------------------
# Config
# -------------------------
PINCH_DIST_THRESHOLD = 0.06     # normalized distance threshold for pinch
OPEN_HAND_THRESHOLD = 0.18      # avg spread between finger tips to consider open
THUMBS_UP_COOLDOWN = 1.5        # seconds between photo captures
MAX_PHOTOS = 2
SNAP_TOLERANCE_PIX = 60         # pixel tolerance for stacking snap
SPHERE_MIN_RADIUS = 20
SPHERE_MAX_RADIUS = 140
ENLARGE_SPEED = 0.8             # multiplier for vertical movement -> radius change

# -------------------------
# Helpers
# -------------------------
def norm_dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def pixel_dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)

def clamp(x, a, b):
    return max(a, min(b, x))

# -------------------------
# TTS Initialization
# -------------------------
engine = pyttsx3.init()
engine.setProperty("rate", 165)
engine.setProperty("volume", 1.0)
voices = engine.getProperty("voices")
if len(voices) > 1:
    engine.setProperty("voice", voices[1].id)

# -------------------------
# Face Detector (Haar)
# -------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------------------------
# MediaPipe Hands
# -------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(max_num_hands=2,
                                min_detection_confidence=0.6,
                                min_tracking_confidence=0.5)

# -------------------------
# Video capture
# -------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# -------------------------
# Game objects (positions in pixels)
# -------------------------
# Start positions (you can tweak)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640

big_rect = {'center': [int(width*0.3), int(height*0.7)], 'size': [260, 80], 'color': (40, 120, 200)}
small_rect = {'center': [int(width*0.6), int(height*0.6)], 'size': [160, 60], 'color': (60, 200, 100)}
sphere = {'center': [int(width*0.5), int(height*0.3)], 'radius': 40, 'color': (200, 50, 50)}

# Original positions for reset
orig_positions = {
    'big_rect': big_rect['center'].copy(),
    'small_rect': small_rect['center'].copy(),
    'sphere': sphere['center'].copy(),
    'sphere_radius': sphere['radius']
}

# Track which hand (Left/Right) holds which object: values: None or 'big', 'small', 'sphere'
hand_holds = {
    'Left': None,
    'Right': None
}
# For each hand track last pinch state to prevent flicker
hand_prev_pinch = {'Left': False, 'Right': False}
# Track last positions of hand to allow relative movement
hand_prev_pos = {'Left': None, 'Right': None}

# Photo capture control
photo_count = 0
last_photo_time = 0.0

# Spoken message only once
spoken = False

# Success flag
stacked_correctly = False
stacked_message_time = 0

# Create photos folder
photos_dir = "photos"
os.makedirs(photos_dir, exist_ok=True)

# -------------------------
# Main loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face detection (keeps face_count accurate)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_count = len(faces)

    # Draw faces
    color_index = 0
    colors = [(0,255,0), (0,200,255), (255,0,100), (200,100,255)]
    for (x, y, w, h) in faces:
        color = colors[color_index % len(colors)]
        color_index += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, f"Faces detected: {face_count}", (width-260, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    if not spoken:
        engine.say("Hand stack demo active. Pinch to grab, open hand to drop.")
        engine.runAndWait()
        spoken = True

    # Draw objects
    bx, by = big_rect['center']
    bw, bh = big_rect['size']
    cv2.rectangle(frame, (bx - bw//2, by - bh//2), (bx + bw//2, by + bh//2), big_rect['color'], -1)
    cv2.putText(frame, "Big (base)", (bx - bw//2, by - bh//2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    sx, sy = small_rect['center']
    sw, sh = small_rect['size']
    cv2.rectangle(frame, (sx - sw//2, sy - sh//2), (sx + sw//2, sy + sh//2), small_rect['color'], -1)
    cv2.putText(frame, "Small (middle)", (sx - sw//2, sy - sh//2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cx, cy = sphere['center']
    cr = sphere['radius']
    cv2.circle(frame, (cx, cy), cr, sphere['color'], -1)
    cv2.putText(frame, "Sphere (top)", (cx - cr, cy - cr - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Process hands
    results = hands_detector.process(frame_rgb)

    # Reset ephemeral per-frame data
    hands_this_frame = {}  # key: label 'Left'/'Right' -> dict with landmarks and pixel coords

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hand_handedness.classification[0].label  # 'Left' / 'Right'
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark
            # fingertip landmarks we care about
            thumb_tip = lm[4]
            index_tip = lm[8]
            middle_tip = lm[12]
            ring_tip = lm[16]
            pinky_tip = lm[20]

            # convert to pixel coords
            tpx, tpy = int(thumb_tip.x * width), int(thumb_tip.y * height)
            ipx, ipy = int(index_tip.x * width), int(index_tip.y * height)
            mpx, mpy = int(middle_tip.x * width), int(middle_tip.y * height)
            rpx, rpy = int(ring_tip.x * width), int(ring_tip.y * height)
            ppx, ppy = int(pinky_tip.x * width), int(pinky_tip.y * height)

            hands_this_frame[label] = {
                'landmarks': lm,
                'thumb_tip': (tpx, tpy),
                'index_tip': (ipx, ipy),
                'middle_tip': (mpx, mpy),
                'ring_tip': (rpx, rpy),
                'pinky_tip': (ppx, ppy),
                'center_px': (int((tpx + ipx + mpx + rpx + ppx)/5), int((tpy + ipy + mpy + rpy + ppy)/5))
            }

    # For each hand present this frame, decide actions
    for label in ['Left', 'Right']:
        if label not in hands_this_frame:
            # If hand disappeared while holding something, drop it
            if hand_holds[label] is not None:
                hand_holds[label] = None
            hand_prev_pinch[label] = False
            hand_prev_pos[label] = None
            continue

        h = hands_this_frame[label]
        lm = h['landmarks']
        thumb_tip = lm[4]
        index_tip = lm[8]

        # Normalized pinch distance
        pinch_d = norm_dist(thumb_tip, index_tip)

        # Open hand detection: check avg distance between fingertips (index-middle, middle-ring, ring-pinky)
        d_im = norm_dist(lm[8], lm[12])
        d_mr = norm_dist(lm[12], lm[16])
        d_rp = norm_dist(lm[16], lm[20])
        avg_spread = (d_im + d_mr + d_rp) / 3.0

        # Pixel center for movement delta
        cx_px, cy_px = h['center_px']
        prev_pos = hand_prev_pos[label]

        # ---- PINCH START: grab if currently not holding and near an object ----
        currently_pinching = (pinch_d < PINCH_DIST_THRESHOLD)

        if currently_pinching and not hand_prev_pinch[label]:
            # pinch started this frame
            # find nearest object to pinch point (index tip pixel)
            ipx, ipy = h['index_tip']
            d_to_big = pixel_dist(ipx, ipy, big_rect['center'][0], big_rect['center'][1])
            d_to_small = pixel_dist(ipx, ipy, small_rect['center'][0], small_rect['center'][1])
            d_to_sphere = pixel_dist(ipx, ipy, sphere['center'][0], sphere['center'][1])

            nearest = min(('big', d_to_big), ('small', d_to_small), ('sphere', d_to_sphere), key=lambda x: x[1])
            obj, nearest_dist = nearest

            # require a reasonable pixel threshold to grab
            if nearest_dist < 120:  # you can tweak this
                # assign object to this hand
                hand_holds[label] = obj
                hand_prev_pos[label] = (cx_px, cy_px)
                # feedback
                engine.say(f"{label} hand grabbed {obj}")
                engine.runAndWait()

        # ---- PINCH RELEASE detected when open hand (spread) or pinch ended ----
        # Consider open if avg_spread > OPEN_HAND_THRESHOLD (normalized)
        is_open = (avg_spread > OPEN_HAND_THRESHOLD)
        if is_open and hand_holds[label] is not None:
            # drop object
            engine.say(f"{label} hand released {hand_holds[label]}")
            engine.runAndWait()
            hand_holds[label] = None
            hand_prev_pos[label] = None

        # ---- WHILE HOLDING: move object with hand movement ----
        if hand_holds[label] is not None and prev_pos is not None:
            # delta movement (pixels)
            dx = cx_px - prev_pos[0]
            dy = cy_px - prev_pos[1]

            obj = hand_holds[label]
            if obj == 'big':
                big_rect['center'][0] += int(dx)
                big_rect['center'][1] += int(dy)
            elif obj == 'small':
                small_rect['center'][0] += int(dx)
                small_rect['center'][1] += int(dy)
            elif obj == 'sphere':
                # If left hand is holding sphere -> allow resizing by vertical motion
                if label == 'Left':
                    # change radius by -dy (moving up increases radius)
                    sphere['radius'] = int(clamp(sphere['radius'] + int(-dy * ENLARGE_SPEED), SPHERE_MIN_RADIUS, SPHERE_MAX_RADIUS))
                    # also move the sphere in x,y a bit (so it follows)
                    sphere['center'][0] += int(dx)
                    sphere['center'][1] += int(dy)
                else:
                    sphere['center'][0] += int(dx)
                    sphere['center'][1] += int(dy)

        # update prev states
        hand_prev_pinch[label] = currently_pinching
        hand_prev_pos[label] = (cx_px, cy_px)

    # ---- Photo capture (thumbs-up) improved ----
    # Detect thumbs up: index tip is above thumb tip and the thumb direction is roughly downward (thumb below palm)
    now = time.time()
    thumbs_up_detected = False
    if 'Right' in hands_this_frame or 'Left' in hands_this_frame:
        # check each hand for thumbs-up
        for label, h in hands_this_frame.items():
            lm = h['landmarks']
            index_tip = lm[8]
            thumb_tip = lm[4]
            # A robust simple rule: index higher than thumb (index.y < thumb.y)
            # and thumb is relatively vertical (dist thumb to wrist small horizontally)
            wrist = lm[0]
            # normalized checks
            if index_tip.y < thumb_tip.y and abs(thumb_tip.x - wrist.x) < 0.12:
                thumbs_up_detected = True
                # You could also check that other fingers are folded (their tips below index y)
                folded_count = 0
                for fidx in [12, 16, 20]:
                    if lm[fidx].y > index_tip.y + 0.02:
                        folded_count += 1
                if folded_count < 2:
                    thumbs_up_detected = False
                if thumbs_up_detected:
                    break

    if thumbs_up_detected and (now - last_photo_time) > THUMBS_UP_COOLDOWN and photo_count < MAX_PHOTOS:
        photo_count += 1
        last_photo_time = now
        filename = os.path.join(photos_dir, f"photo_{photo_count}.jpg")
        cv2.imwrite(filename, frame)
        cv2.putText(frame, f"Photo Saved: {filename}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        engine.say(f"Photo {photo_count} saved")
        engine.runAndWait()

    # ---- Check stacking correctness ----
    # We want: big_rect bottom, small_rect on top of big_rect, sphere on top of small_rect
    # We'll check centers and vertical ordering
    def check_and_snap():
        global stacked_correctly, stacked_message_time
        bx, by = big_rect['center']
        sx, sy = small_rect['center']
        cx, cy = sphere['center']
        # small rect should be above big_rect and centered horizontally within tolerance
        within_x_small = abs(sx - bx) < SNAP_TOLERANCE_PIX
        small_above_big = (by - sy) > (big_rect['size'][1]//2 - 10) and (by - sy) < (big_rect['size'][1] + 200)
        # sphere should be above small and near center
        within_x_sphere = abs(cx - sx) < SNAP_TOLERANCE_PIX
        sphere_above_small = (sy - cy) > (small_rect['size'][1]//2 - 5) and (sy - cy) < (small_rect['size'][1] + 200)

        if within_x_small and small_above_big and within_x_sphere and sphere_above_small:
            # Snap positions to neat stack
            small_rect['center'][0] = bx
            small_rect['center'][1] = by - (big_rect['size'][1]//2 + small_rect['size'][1]//2 + 5)
            sphere['center'][0] = small_rect['center'][0]
            sphere['center'][1] = small_rect['center'][1] - (small_rect['size'][1]//2 + sphere['radius'] + 5)
            stacked_correctly = True
            stacked_message_time = time.time()
            engine.say("Stack complete")
            engine.runAndWait()
        else:
            stacked_correctly = False

    check_and_snap()

    # Show stack success message briefly
    if stacked_correctly and (time.time() - stacked_message_time) < 3.0:
        cv2.putText(frame, "STACKED CORRECTLY!", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    # Instructions & status
    cv2.putText(frame, "Press 'r' to reset objects & photos. 'q' to quit.", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.putText(frame, f"Photos taken: {photo_count}/{MAX_PHOTOS}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 100), 2)

    # Show which hand holds what
    cv2.putText(frame, f"Left holds: {hand_holds['Left']}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Right holds: {hand_holds['Right']}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("AI Vision Project - Stack & Gesture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        # reset positions and photos
        big_rect['center'] = orig_positions['big_rect'].copy()
        small_rect['center'] = orig_positions['small_rect'].copy()
        sphere['center'] = orig_positions['sphere'].copy()
        sphere['radius'] = orig_positions['sphere_radius']
        photo_count = 0
        last_photo_time = 0.0
        # clear saved photos folder (optional) - commented out, uncomment to remove files
        # for f in os.listdir(photos_dir):
        #     os.remove(os.path.join(photos_dir, f))
        engine.say("Reset done")
        engine.runAndWait()

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands_detector.close()
