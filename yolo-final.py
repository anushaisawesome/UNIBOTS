"""
yolo-final.py
=============
CHANGES FROM PREVIOUS VERSION:
  - Removed inference + roboflow entirely
  - Model now loaded locally from best.onnx (no API key, no internet required)
  - Uses Ultralytics YOLO to run the ONNX model directly
  - Updated class names + diameter mapping to match actual dataset:
      "0", "1"                                    → ball bearing (10mm)
      "PingPong", "orange_pp", "ping-pong",
      "ping-pong-ball"                            → ping pong ball (40mm)
  - Everything else (state machine, steering, beam sensor) unchanged

HOW TO RUN:
-----------
1. Place best.onnx in the same folder as this script

2. pip install ultralytics opencv-python python-dotenv rpi-lgpio
   (no inference, no roboflow needed)

3. python yolo-final.py

4. Press Q in the camera window or Ctrl+C to stop.

DEBUG FLAGS:
   DEBUG_VISION   = True  — every detection and distance estimate
   DEBUG_STEERING = True  — motor speeds sent each loop iteration
   DEBUG_BEAM     = True  — beam sensor reads
   DEBUG_STATE    = True  — state transitions
"""

from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import time
import numpy as np

import movement_v2
from sensor import beam_broken


# =============================================================
# DEBUG FLAGS
# =============================================================

DEBUG_VISION   = True
DEBUG_STEERING = True
DEBUG_BEAM     = True
DEBUG_STATE    = True


def dbg(section, msg):
    print(f"[{section}] {msg}")


# =============================================================
# 1.  MODEL + CAMERA
# =============================================================

MODEL_PATH = "best.onnx"   # must be in same folder as this script

print("=" * 60)
print("STARTUP — loading local ONNX model")
print(f"  Model: {MODEL_PATH}")

model = YOLO(MODEL_PATH)
print("Model loaded ✓")

print("\nOpening camera...")
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
)
picam2.start()
time.sleep(1)   # let first frames stabilise
print("Camera started with Picamera2 ✓")
print("Resolution: 640x480")

# Classes that are ping pong balls
PINGPONG_CLASSES = {"PingPong", "orange_pp", "ping-pong", "ping-pong-ball"}

# Classes that are ball bearings (poorly named in the merged dataset)
BEARING_CLASSES  = {"0", "1"}

# Real-world diameters in metres — used for distance estimation
def get_real_diameter(class_name):
    if class_name in PINGPONG_CLASSES:
        return 0.040   # 40mm ping pong ball
    if class_name in BEARING_CLASSES:
        return 0.010   # 10mm ball bearing
    return 0.020       # fallback for unknown classes

FOCAL_LENGTH = 800.0   # pixels — matches camera matrix (800, 0, 320)

print(f"Focal length: {FOCAL_LENGTH}px")
print("=" * 60)


# =============================================================
# 2.  ROBOT PARAMETERS
# =============================================================

FRAME_CENTER       = 320   # horizontal centre of 640px frame
BASE_SPEED         = 80    # forward drive speed (0-100)
SEARCH_SPEED       = 40    # rotation speed while scanning
MAX_CAPACITY       = 4     # collect this many balls then go home
RETURN_HOME_TIME_S = 3     # seconds of reverse to reach home

# Steering gain — pixel error of 100px at Kp=0.3 → adjustment of 30 speed units
# Increase Kp if the robot is slow to correct. Decrease if it wobbles.
Kp = 0.3

# Cap the maximum steering adjustment so neither motor ever goes negative
MAX_STEER_ADJUST = 60

print("\n[PARAMS] Robot parameters:")
print(f"  FRAME_CENTER       = {FRAME_CENTER}px")
print(f"  BASE_SPEED         = {BASE_SPEED}")
print(f"  SEARCH_SPEED       = {SEARCH_SPEED}")
print(f"  Kp                 = {Kp}")
print(f"  MAX_STEER_ADJUST   = {MAX_STEER_ADJUST}")
print(f"  MAX_CAPACITY       = {MAX_CAPACITY} balls")
print(f"  RETURN_HOME_TIME_S = {RETURN_HOME_TIME_S}s")


# =============================================================
# 3.  DETECTION
# =============================================================

_collected_ids = set()


def get_detections(frame):
    """
    Runs YOLO inference on a frame and returns structured detections.
    Filters out already-collected track IDs.
    """
    if frame is None:
        dbg("VISION", "❌ frame is None")
        return []

    if DEBUG_VISION:
        dbg("VISION", "Frame captured — running inference...")

    results = model(frame, conf=0.5, verbose=False)[0]
    detections = []

    if DEBUG_VISION:
        dbg("VISION", f"Raw predictions: {len(results.boxes)}")

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf     = float(box.conf[0])
        cls_id   = int(box.cls[0])
        class_name = model.names[cls_id]

        # Ultralytics track() assigns IDs; plain model() won't — default to -1
        track_id = int(box.id[0]) if box.id is not None else -1

        if DEBUG_VISION:
            dbg("VISION", f"  [{i}] class='{class_name}' conf={conf:.2f} "
                          f"box=({x1},{y1},{x2},{y2}) track_id={track_id}")

        if track_id in _collected_ids:
            if DEBUG_VISION:
                dbg("VISION", f"    ↳ Skipping ID {track_id} (already collected)")
            continue

        width  = x2 - x1
        height = y2 - y1
        pixel_diameter = max(width, height)

        if pixel_diameter == 0:
            dbg("VISION", "    ↳ Skipping — zero pixel diameter")
            continue

        real_diameter = get_real_diameter(class_name)
        distance_cm   = (real_diameter * FOCAL_LENGTH) / pixel_diameter * 100

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if DEBUG_VISION:
            dbg("VISION", f"    ↳ pixel_diam={pixel_diameter}px "
                          f"real_diam={real_diameter*1000:.0f}mm "
                          f"dist={distance_cm:.1f}cm")

        detections.append({
            "class":       class_name,
            "confidence":  conf,
            "bbox":        (x1, y1, x2, y2),
            "center_px":   (cx, cy),
            "distance_cm": distance_cm,
            "track_id":    track_id,
        })

    if DEBUG_VISION:
        dbg("VISION", f"Returning {len(detections)} valid detections")

    return detections


def best_target(detections):
    """Returns the closest uncollected ball, or None."""
    if not detections:
        return None
    target = min(detections, key=lambda d: d["distance_cm"])
    if DEBUG_VISION:
        dbg("VISION", f"Target: '{target['class']}' at {target['distance_cm']:.1f}cm "
                      f"track_id={target['track_id']}")
    return target


# =============================================================
# 4.  STEERING
# =============================================================

def steer_toward(error):
    """
    Drives both motors forward with differential speed to curve
    toward the target. Never stops forward motion.

    error : int — pixels, positive = ball is right of centre
    """
    adjust = Kp * error
    adjust = max(-MAX_STEER_ADJUST, min(MAX_STEER_ADJUST, adjust))

    left_speed  = BASE_SPEED - adjust
    right_speed = BASE_SPEED + adjust

    left_speed  = max(0, min(100, left_speed))
    right_speed = max(0, min(100, right_speed))

    if DEBUG_STEERING:
        dbg("STEER", f"error={error:+d}px | adjust={adjust:+.1f} | "
                     f"left={left_speed:.0f} right={right_speed:.0f}")

    movement_v2.turn(left_speed, right_speed)


# =============================================================
# 5.  STATE MACHINE
# =============================================================

def run():
    global _collected_ids

    state           = "SEARCHING"
    active_track_id = -1
    ball_count      = 0

    print("\n" + "=" * 60)
    print("STARTING — spinner ON from boot")
    print(f"Collecting {MAX_CAPACITY} balls then returning home")
    print("Press Q in camera window or Ctrl+C to stop")
    print("=" * 60 + "\n")

    dbg("STARTUP", "start_spinner() — will stay on until all balls collected")
    movement_v2.start_spinner()

    loop_count = 0

    try:
        while True:
            loop_count += 1

            if loop_count % 50 == 0:
                print(f"[LOOP] iter={loop_count} | state={state} | "
                      f"balls={ball_count}/{MAX_CAPACITY} | "
                      f"collected_ids={_collected_ids}")

            # --- Grab frame ---
            frame = picam2.capture_array()

            if frame is None:
                dbg("LOOP", "No frame — skipping")
                time.sleep(0.05)
                continue

            # Picamera2 gives RGB; convert to BGR for OpenCV + YOLO
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # --- Run detections ---
            detections    = get_detections(frame)
            target        = best_target(detections)
            ball_detected = target is not None

            if ball_detected:
                ball_cx  = target["center_px"][0]
                error    = ball_cx - FRAME_CENTER
                track_id = target["track_id"]

            # --- Draw annotations ---
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame,
                            f'{d["class"]} {d["distance_cm"]:.0f}cm id={d["track_id"]}',
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.putText(frame,
                        f"State: {state}  Balls: {ball_count}/{MAX_CAPACITY}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            #cv2.imshow("Ball Detection", frame)
            #if cv2.waitKey(1) & 0xFF == ord("q"):
            #    print("[LOOP] Q pressed — exiting")
            #    break

            # ==========================================
            # BEAM SENSOR — checked every loop iteration
            # ==========================================
            beam_status = beam_broken()

            if DEBUG_BEAM:
                dbg("BEAM", f"beam_broken() = {beam_status} | ball_count={ball_count}")

            if beam_status:
                dbg("BEAM", f"✅ Beam BROKEN — ball confirmed! track_id={active_track_id}")

                if active_track_id != -1:
                    _collected_ids.add(active_track_id)
                    dbg("BEAM", f"collected_ids={_collected_ids}")

                ball_count += 1
                dbg("BEAM", f"ball_count → {ball_count}/{MAX_CAPACITY}")

                if ball_count >= MAX_CAPACITY:
                    dbg("STATE", f"🎯 {MAX_CAPACITY} balls collected! "
                                 f"stop_spinner() → RETURNING_HOME")
                    movement_v2.stop_spinner()
                    movement_v2.stop_drive()
                    state = "RETURNING_HOME"
                    time.sleep(0.3)
                else:
                    dbg("STATE", f"Ball in! Need {MAX_CAPACITY - ball_count} more → SEARCHING")
                    active_track_id = -1
                    state = "SEARCHING"

                time.sleep(0.5)
                continue

            # ==========================================
            # SEARCHING
            # ==========================================
            if state == "SEARCHING":
                if DEBUG_STATE:
                    dbg("STATE", f"SEARCHING — turn_left({SEARCH_SPEED})")
                movement_v2.turn_left(SEARCH_SPEED)

                if ball_detected:
                    if DEBUG_STATE:
                        dbg("STATE", f"Ball spotted! '{target['class']}' "
                                     f"track_id={track_id} → APPROACHING")
                    active_track_id = track_id
                    state = "APPROACHING"

            # ==========================================
            # APPROACHING
            # ==========================================
            elif state == "APPROACHING":
                if not ball_detected:
                    dbg("STATE", "⚠️  Ball lost — stop_drive() → SEARCHING")
                    movement_v2.stop_drive()
                    state = "SEARCHING"
                else:
                    if DEBUG_STEERING:
                        dbg("STEER", f"APPROACHING '{target['class']}' | "
                                     f"ball_cx={ball_cx}px | "
                                     f"centre={FRAME_CENTER}px | "
                                     f"error={error:+d}px | "
                                     f"dist={target['distance_cm']:.1f}cm")
                    steer_toward(error)

            # ==========================================
            # RETURNING_HOME
            # ==========================================
            elif state == "RETURNING_HOME":
                dbg("STATE", f"🏠 RETURNING_HOME — "
                             f"move_backward({BASE_SPEED}) for {RETURN_HOME_TIME_S}s")
                movement_v2.move_backward(BASE_SPEED)
                time.sleep(RETURN_HOME_TIME_S)
                movement_v2.stop_drive()
                dbg("STATE", "Arrived home → DROPPING")
                state = "DROPPING"

            # ==========================================
            # DROPPING
            # ==========================================
            elif state == "DROPPING":
                dbg("STATE", "📦 DROPPING — lift_up(speed=80)")
                movement_v2.lift_up(speed=80)
                time.sleep(2.0)
                movement_v2.stop_lift()

                dbg("STATE", "Lowering lift — lift_down(speed=80)")
                movement_v2.lift_down(speed=80)
                time.sleep(1.5)
                movement_v2.stop_lift()

                _collected_ids.clear()
                ball_count      = 0
                active_track_id = -1

                dbg("STATE", "start_spinner() — beginning new collection run → SEARCHING")
                movement_v2.start_spinner()
                state = "SEARCHING"

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[LOOP] Ctrl+C — shutting down")

    finally:
        print("[SHUTDOWN] Stopping all motors...")
        movement_v2.shutdown()
        print("[SHUTDOWN] Stopping camera...")
        picam2.stop()
        #cv2.destroyAllWindows()
        print("[SHUTDOWN] Complete ✓")


# =============================================================
# 6.  ENTRY POINT
# =============================================================

if __name__ == "__main__":
    run()
