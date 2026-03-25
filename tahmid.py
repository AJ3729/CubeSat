"""CubeSat Flight Software

Includes:
- Mission scheduler loop
- Periodic camera capture
- Metadata logging
- Hazard grid (10x10 safety map)
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

# ========================
# CONFIGURATION
# ========================

CAPTURE_INTERVAL = 10.0      # seconds between captures
GRID_SIZE = 10               # hazard grid size
GRAVITY = 9.80665            # m/s^2

BASE_PATH = Path("/home/palace-stuy/CubeSat")
REPO_PATH = BASE_PATH
NAME = "Tahmidl"
GITHUB_BRANCH="main"
GITHUB_REMOTE_NAME = "origin"
ENABLE_GITHUB_UPLOAD = True
IMAGE_FOLDER = BASE_PATH / "Images"
DATA_FOLDER = BASE_PATH / "Data"

NAME = "TahmidI"

# ========================
# STARTUP
# ========================

def print_banner() -> None:
    print("=" * 50)
    print("FlatSat / CubeSat Flight Software")
    print("=" * 50)
    print(f"Image folder     : {IMAGE_FOLDER}")
    print(f"Data folder      : {DATA_FOLDER}")
    print(f"Capture interval : {CAPTURE_INTERVAL} sec")
    print(f"Grid size        : {GRID_SIZE}x{GRID_SIZE}")
    print("GitHub upload    : DISABLED")
    print("=" * 50)


def initialize_folders() -> None:
    IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    print("[INIT] Data directories ready")


def initialize_sensors():
    i2c = board.I2C()
    accel_gyro = LSM6DSOX(i2c)
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


def generate_filenames() -> tuple[str, str]:
    t = timestamp()
    image_path = IMAGE_FOLDER / f"{NAME}_{t}.jpg"
    data_path = DATA_FOLDER / f"{NAME}_{t}.json"
    return str(image_path), str(data_path)


# ========================
# SENSOR FUNCTIONS
# ========================

def get_acceleration(accel_gyro) -> tuple[float, float, float]:
    ax, ay, az = accel_gyro.acceleration
    return ax, ay, az


def get_orientation(accel_gyro, magnetometer) -> dict[str, float]:
    ax, ay, az = accel_gyro.acceleration
    mx, my, _mz = magnetometer.magnetic

    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2))
    yaw = math.atan2(my, mx)

    return {
        "roll_deg": round(math.degrees(roll), 3),
        "pitch_deg": round(math.degrees(pitch), 3),
        "yaw_deg": round(math.degrees(yaw), 3),
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
# DATA LOGGING
# ========================

def save_metadata(filepath: str, accel, orientation, grid) -> None:
    magnitude = math.sqrt(accel[0]**2 + accel[1]**2 + accel[2]**2)

    data = {
        "timestamp": timestamp(),
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


def print_event_summary(accel, orientation, image_path, data_path) -> None:
    magnitude = math.sqrt(accel[0]**2 + accel[1]**2 + accel[2]**2)

    print("[SUMMARY] Capture complete")
    print(f"  Image file : {image_path}")
    print(f"  Data file  : {data_path}")
    print(
        f"  Accel      : x={accel[0]:.3f}, y={accel[1]:.3f}, z={accel[2]:.3f} m/s^2"
    )
    print(f"  Magnitude  : {magnitude:.3f} m/s^2")
    print(
        f"  Orientation: roll={orientation['roll_deg']:.3f}, "
        f"pitch={orientation['pitch_deg']:.3f}, "
        f"yaw={orientation['yaw_deg']:.3f}"
    )


# ========================
# MAIN CAPTURE WORKFLOW
# ========================

def handle_capture_cycle(picam2, accel_gyro, magnetometer) -> None:
    print("[CYCLE] Starting capture cycle...")

    image_path, data_path = generate_filenames()

    accel = get_acceleration(accel_gyro)
    orientation = get_orientation(accel_gyro, magnetometer)
    print("[CYCLE] Sensor readings collected")

    capture_ok = capture_image(picam2, image_path)
    if not capture_ok:
        print("[CYCLE] Capture cycle failed\n")
        return

    grid = generate_hazard_grid(image_path, GRID_SIZE)
    print_hazard_grid(grid)

    save_metadata(data_path, accel, orientation, grid)
    print_event_summary(accel, orientation, image_path, data_path)
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

        elapsed = time.time() - cycle_start
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
