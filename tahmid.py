"""
CubeSat Flight Software

Includes:
- Mission scheduler loop
- Periodic camera capture
- Metadata logging
- Hazard grid (10x10 safety map)
- Hazard grid overlay on captured image (displayed locally + saved)
- Optional GitHub upload
"""

# ========================
# IMPORTS
# ========================
import json
import math
import time
from datetime import datetime
from pathlib import Path

import board
import cv2
import numpy as np
from adafruit_lis3mdl import LIS3MDL
from adafruit_lsm6ds.lsm6dsox import LSM6DSOX
from picamera2 import Picamera2
from git import Repo

# ========================
# CONFIGURATION
# ========================

CAPTURE_INTERVAL = 10.0      # seconds between captures
GRID_SIZE = 10               # hazard grid size
GRAVITY = 9.80665            # m/s^2

# ---- LOCAL STORAGE ----
BASE_PATH        = Path("/home/palace-stuy/CubeSat")
IMAGE_FOLDER     = BASE_PATH / "Images"
DATA_FOLDER      = BASE_PATH / "Data"
ANNOTATED_FOLDER = BASE_PATH / "Annotated"

NAME = "TahmidI"

# ---- GITHUB SETTINGS ----
ENABLE_GITHUB_UPLOAD = True
REPO_PATH            = BASE_PATH
GITHUB_BRANCH        = "main"
GITHUB_REMOTE_NAME   = "origin"
COMMIT_EVERY_CAPTURE = True

# ========================
# STARTUP
# ========================

def print_banner() -> None:
    print("=" * 60)
    print("CubeSat Flight Software")
    print("=" * 60)
    print(f"Base path         : {BASE_PATH}")
    print(f"Image folder      : {IMAGE_FOLDER}")
    print(f"Data folder       : {DATA_FOLDER}")
    print(f"Annotated folder  : {ANNOTATED_FOLDER}")
    print(f"Capture interval  : {CAPTURE_INTERVAL} sec")
    print(f"Grid size         : {GRID_SIZE}x{GRID_SIZE}")
    print(f"GitHub upload     : {'ENABLED' if ENABLE_GITHUB_UPLOAD else 'DISABLED'}")
    print(f"Repo path         : {REPO_PATH}")
    print(f"Remote name       : {GITHUB_REMOTE_NAME}")
    print(f"Branch            : {GITHUB_BRANCH}")
    print("=" * 60)


def initialize_folders() -> None:
    IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    ANNOTATED_FOLDER.mkdir(parents=True, exist_ok=True)
    print("[INIT] Data directories ready")


def initialize_sensors():
    i2c = board.I2C()
    accel_gyro   = LSM6DSOX(i2c)
    magnetometer = LIS3MDL(i2c)
    print("[INIT] IMU and magnetometer initialized")
    return accel_gyro, magnetometer


def initialize_camera():
    picam2 = Picamera2()
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    print("[INIT] Camera initialized and warmed up")
    return picam2


# ========================
# UTILITY FUNCTIONS
# ========================

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_filenames() -> tuple[str, str, str]:
    t = timestamp()
    image_path     = IMAGE_FOLDER     / f"{NAME}_{t}.jpg"
    data_path      = DATA_FOLDER      / f"{NAME}_{t}.json"
    annotated_path = ANNOTATED_FOLDER / f"{NAME}_{t}_grid.jpg"
    return str(image_path), str(data_path), str(annotated_path)


# ========================
# SENSOR FUNCTIONS
# ========================

def get_acceleration(accel_gyro) -> tuple[float, float, float]:
    ax, ay, az = accel_gyro.acceleration
    return ax, ay, az


def get_orientation(accel_gyro, magnetometer) -> dict[str, float]:
    ax, ay, az = accel_gyro.acceleration
    mx, my, _mz = magnetometer.magnetic

    roll  = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2))
    yaw   = math.atan2(my, mx)

    return {
        "roll_deg":  round(math.degrees(roll),  3),
        "pitch_deg": round(math.degrees(pitch), 3),
        "yaw_deg":   round(math.degrees(yaw),   3),
    }


# ========================
# CAMERA
# ========================

def capture_image(picam2, filepath: str) -> bool:
    try:
        print(f"[CAMERA] Capturing image -> {filepath}")
        picam2.capture_file(filepath)
        print("[CAMERA] Capture successful")
        return True
    except Exception as e:
        print(f"[ERROR] Camera capture failed: {e}")
        return False


# ========================
# HAZARD GRID
# ========================

def generate_hazard_grid(image_path: str, grid_size: int = GRID_SIZE) -> list[list[int]] | None:
    print("[GRID] Generating hazard grid...")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("[ERROR] Could not read image for hazard grid generation")
        return None

    height, width = img.shape

    if height < grid_size or width < grid_size:
        print("[ERROR] Image too small for grid generation")
        return None

    grid = []

    for i in range(grid_size):
        row = []
        y0 = (i * height) // grid_size
        y1 = ((i + 1) * height) // grid_size

        for j in range(grid_size):
            x0 = (j * width) // grid_size
            x1 = ((j + 1) * width) // grid_size

            cell = img[y0:y1, x0:x1]

            if cell.size == 0:
                score = 0
            else:
                avg_brightness = float(np.mean(cell))
                score = max(0, min(10, int(round((avg_brightness / 255.0) * 10))))

            row.append(score)

        grid.append(row)

    print("[GRID] Hazard grid complete")
    return grid


def print_hazard_grid(grid: list[list[int]] | None) -> None:
    if grid is None:
        print("[GRID] No grid to display")
        return

    print("[GRID] Hazard Grid:")
    for row in grid:
        print(" ".join(f"{cell:2d}" for cell in row))


# ========================
# IMAGE OVERLAY
# ========================

def overlay_hazard_grid_on_image(
    image_path: str,
    grid: list[list[int]],
    output_path: str,
    grid_size: int = GRID_SIZE,
) -> bool:
    """
    Draw a colour-coded hazard grid on top of the captured image,
    save it to the Annotated/ folder, and display it locally for 5 seconds.

    Colour key (BGR):
      Green  = safe       score 7-10
      Yellow = caution    score 4-6
      Red    = hazardous  score 0-3
    """
    print("[OVERLAY] Creating annotated image...")

    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Could not read original image for overlay")
        return False

    height, width = img.shape[:2]
    overlay = img.copy()    # receives the solid colour fills
    base    = img.copy()    # receives the grid borders and text

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness  = 1

    for i in range(grid_size):
        y0 = (i * height) // grid_size
        y1 = ((i + 1) * height) // grid_size

        for j in range(grid_size):
            x0 = (j * width) // grid_size
            x1 = ((j + 1) * width) // grid_size

            score = grid[i][j]

            # Colour by safety band (BGR)
            if score >= 7:
                color = (34, 197, 94)       # green
            elif score >= 4:
                color = (50, 200, 234)      # yellow
            else:
                color = (68, 68, 239)       # red

            # Solid fill on the overlay layer
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)

            # White cell border on the base layer
            cv2.rectangle(base, (x0, y0), (x1, y1), (255, 255, 255), 1)

            # Score label centred in cell — drop shadow then white text
            text = str(score)
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            tx = x0 + ((x1 - x0) - tw) // 2
            ty = y0 + ((y1 - y0) + th) // 2

            cv2.putText(base, text, (tx + 1, ty + 1),
                        font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(base, text, (tx, ty),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Blend: 30% colour overlay, 70% original + borders/text
    alpha     = 0.30
    annotated = cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)

    # ---- LEGEND (bottom-left corner) ----
    legend_items = [
        ("SAFE   (7-10)", (34, 197, 94)),
        ("CAUTION(4-6) ", (50, 200, 234)),
        ("HAZARD (0-3) ", (68, 68, 239)),
    ]
    swatch      = 28
    row_height  = 44
    leg_x       = 10
    leg_y       = height - (len(legend_items) * row_height + 20)
    lg_scale    = max(0.55, width / 1600)

    for idx, (label, color) in enumerate(legend_items):
        ly = leg_y + idx * row_height
        cv2.rectangle(annotated, (leg_x, ly),
                      (leg_x + swatch, ly + swatch), (255, 255, 255), 2)
        cv2.rectangle(annotated, (leg_x, ly),
                      (leg_x + swatch, ly + swatch), color, -1)
        tx2 = leg_x + swatch + 8
        ty2 = ly + swatch - 4
        cv2.putText(annotated, label, (tx2 + 1, ty2 + 1),
                    font, lg_scale, (0, 0, 0), 2)
        cv2.putText(annotated, label, (tx2, ty2),
                    font, lg_scale, (255, 255, 255), 1)

    # ---- TITLE BAR (top-left corner) ----
    title = "LUNAR HAZARD MAP"
    (ttw, tth), _ = cv2.getTextSize(title, font, lg_scale * 1.2, 2)
    cv2.rectangle(annotated, (0, 0), (ttw + 20, tth + 16), (0, 0, 0), -1)
    cv2.putText(annotated, title, (10, tth + 8),
                font, lg_scale * 1.2, (255, 255, 255), 2)

    try:
        cv2.imwrite(output_path, annotated)
        print(f"[OVERLAY] Annotated image saved -> {output_path}")

        # Display locally only if a screen is connected
        import os
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            cv2.imshow("Lunar Hazard Map", annotated)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()
        else:
            print("[OVERLAY] No display detected — skipping local preview")

        return True
    except Exception as e:
        print(f"[ERROR] Failed to save annotated image: {e}")
        return False


# ========================
# DATA LOGGING
# ========================

def save_metadata(
    filepath: str,
    accel,
    orientation,
    grid,
    image_path: str,
    annotated_path: str,
) -> None:
    magnitude = math.sqrt(accel[0]**2 + accel[1]**2 + accel[2]**2)

    data = {
        "timestamp": timestamp(),
        "image_path": image_path,
        "annotated_image_path": annotated_path,
        "acceleration_m_s2": {
            "x": round(accel[0], 6),
            "y": round(accel[1], 6),
            "z": round(accel[2], 6),
            "magnitude": round(magnitude, 6),
            "gravity_offset": round(abs(magnitude - GRAVITY), 6),
        },
        "orientation": orientation,
        "hazard_grid": grid,
    }

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"[DATA] Metadata saved -> {filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to save metadata: {e}")


def print_event_summary(accel, orientation, image_path, annotated_path, data_path) -> None:
    magnitude = math.sqrt(accel[0]**2 + accel[1]**2 + accel[2]**2)

    print("[SUMMARY] Capture complete")
    print(f"  Image file     : {image_path}")
    print(f"  Annotated file : {annotated_path}")
    print(f"  Data file      : {data_path}")
    print(f"  Accel          : x={accel[0]:.3f}, y={accel[1]:.3f}, z={accel[2]:.3f} m/s^2")
    print(f"  Magnitude      : {magnitude:.3f} m/s^2")
    print(
        f"  Orientation    : roll={orientation['roll_deg']:.3f}, "
        f"pitch={orientation['pitch_deg']:.3f}, "
        f"yaw={orientation['yaw_deg']:.3f}"
    )


# ========================
# GITHUB UPLOAD
# ========================

def upload_to_github(files_to_upload: list[str], commit_message: str) -> None:
    if not ENABLE_GITHUB_UPLOAD:
        print("[GIT] Upload disabled")
        return

    try:
        repo = Repo(str(REPO_PATH))

        repo.index.add(files_to_upload)

        if repo.is_dirty(untracked_files=True):
            repo.index.commit(commit_message)
            remote = repo.remote(name=GITHUB_REMOTE_NAME)
            remote.push(refspec=f"{GITHUB_BRANCH}:{GITHUB_BRANCH}")
            print("[GIT] Upload successful")
        else:
            print("[GIT] No changes detected, skipping push")

    except Exception as e:
        print(f"[GIT ERROR] {e}")


# ========================
# MAIN CAPTURE WORKFLOW
# ========================

def handle_capture_cycle(picam2, accel_gyro, magnetometer) -> None:
    print("[CYCLE] Starting capture cycle...")

    image_path, data_path, annotated_path = generate_filenames()

    accel       = get_acceleration(accel_gyro)
    orientation = get_orientation(accel_gyro, magnetometer)
    print("[CYCLE] Sensor readings collected")

    capture_ok = capture_image(picam2, image_path)
    if not capture_ok:
        print("[CYCLE] Capture cycle failed — image not saved\n")
        return

    grid = generate_hazard_grid(image_path, GRID_SIZE)
    if grid is None:
        print("[CYCLE] Grid generation failed\n")
        return

    print_hazard_grid(grid)

    overlay_ok = overlay_hazard_grid_on_image(
        image_path=image_path,
        grid=grid,
        output_path=annotated_path,
        grid_size=GRID_SIZE,
    )

    save_metadata(data_path, accel, orientation, grid, image_path, annotated_path)
    print_event_summary(accel, orientation, image_path, annotated_path, data_path)

    if overlay_ok and COMMIT_EVERY_CAPTURE:
        upload_to_github(
            files_to_upload=[image_path, annotated_path, data_path],
            commit_message=f"CubeSat capture {timestamp()}",
        )

    print("[CYCLE] Capture cycle finished\n")


# ========================
# MISSION LOOP
# ========================

def mission_loop(picam2, accel_gyro, magnetometer) -> None:
    print("[BOOT] Mission loop started")
    print(f"[BOOT] Capturing one image every {CAPTURE_INTERVAL} seconds\n")

    while True:
        cycle_start = time.time()

        try:
            handle_capture_cycle(picam2, accel_gyro, magnetometer)
        except Exception as e:
            print(f"[LOOP ERROR] {e}")

        elapsed    = time.time() - cycle_start
        sleep_time = max(0.0, CAPTURE_INTERVAL - elapsed)

        print(f"[WAIT] Sleeping for {sleep_time:.2f} seconds...\n")
        time.sleep(sleep_time)


# ========================
# MAIN
# ========================

if __name__ == "__main__":
    picam2 = None

    try:
        print_banner()
        initialize_folders()
        accel_gyro, magnetometer = initialize_sensors()
        picam2 = initialize_camera()
        mission_loop(picam2, accel_gyro, magnetometer)

    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Program stopped by user")

    except Exception as e:
        print(f"[FATAL ERROR] {e}")

    finally:
        if picam2 is not None:
            try:
                picam2.stop()
                print("[SHUTDOWN] Camera stopped")
            except Exception:
                pass