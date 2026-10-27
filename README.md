# UNIBOTS

The final version of our KCL Robotics UniBots robot script. It does the same job as
[unibots-yolo](https://github.com/anushaisawesome/unibots-yolo), but the ball detector runs from a local ONNX file with
Ultralytics, so there's no Roboflow account, no API key and no internet needed at all. This is the version that ran on the
robot.

<!-- TODO visual: the robot in the arena, or a camera frame with detections -->

## What's different from unibots-yolo

- The model loads from `best.onnx` in the same folder, with Ultralytics, instead of through Roboflow Inference.
- The camera is read with `picamera2`.
- Class names match my merged dataset: `0` and `1` are ball bearings (10 mm) and the ping pong classes are 40 mm, which the
  distance estimate needs.
- There are debug flags that print detections, distances, motor speeds, beam reads and state changes.

The state machine, steering and break-beam logic are the same, and the unibots-yolo README explains how they work.

## Running it

On the Pi, with `best.onnx` next to the script (the model file isn't in this repo):

```bash
pip install ultralytics opencv-python python-dotenv rpi-lgpio
python yolo-final.py
```

It also imports `sensor.py` from unibots-yolo, so copy that across too. Get rid of the camera window with Q, or Ctrl+C.

## Limits

The camera calibration is still a placeholder, so distances are rough.

## Credits

Built with the KCL Robotics UniBots team. The detector was trained on public Roboflow Universe datasets that I merged into
12,548 images.
