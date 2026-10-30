# Replaying a LUNARIS route on an Arduino Alvik (4 ft × 4 ft grid)

The console shrinks the planned lunar route into a 4 ft × 4 ft (1219 mm) floor grid. It converts the route into
Alvik `rotate()` / `move()` steps and uploads a MicroPython `main.py` to the robot. You place the Alvik at the
marked start, press **OK (✓)**, and it traces the route.

## One-time setup

1. The Alvik needs its standard MicroPython library (`arduino_alvik`), which ships with the robot. If you reflashed it,
   follow https://github.com/arduino/arduino-alvik-mpy (`install.sh -p <port>`).
2. On the computer running the LUNARIS server: `pip install mpremote` (already in `.venv`). On Linux, your user must be
   in the `dialout` group to open `/dev/ttyACM*`.
3. Close Arduino Lab for MicroPython or any serial monitor; only one program can hold the USB port.

## Steps

1. In the console, load a dataset (or a crater from **◎ Craters**), plan a route (step 4), and optionally run the rover
   simulation (step 6).
2. Click **⬢ Send to Alvik**. The uploader shows:
   * the 4 ft grid (1 ft squares, 6 in minor lines) with the dashed safety margin (default 120 mm, so the 10 cm robot
     never leaves the floor)
   * the full-resolution scaled route (thin line), the simplified Alvik path (thick, coloured by simulated condition),
     and the start pose with its heading arrow
   * scale, number of steps, maximum deviation from the scaled route, and estimated run time
   * **Preview run**: animates the robot at the Alvik's real speeds (100 mm/s, 100°/s, from the official library)
3. Connect the Alvik by USB-C, switch it ON, pick its port (the Nano ESP32 shows as "Arduino"), and click
   **Upload & run**. The robot's existing `main.py` is backed up to `data/processed/alvik/` first.
   *No USB access from the server machine?* Use **Download main.py** and copy it with Arduino Lab for MicroPython.
4. Place the robot as instructed (centre position in cm from the left and near edges, and the heading). Press **✓**.
   * **✗** stops it immediately, with red flashes. Press ✓ to run again from the start position.
   * When it finishes, it flashes white and prints its odometry pose over USB serial.

## LED meaning during the run

| Colour | Segment in the simulation |
|---|---|
| green | normal driving |
| blue | inside a permanently shadowed region (LOLA PSR map) |
| yellow | wheel slip above the rover's operational limit |
| red | the segment where the simulated rover failed (the program stops there when "stop where the rover failed" is on) |

## How the conversion works (`core/hardware/alvik.py`)

1. **Fit.** The route is scaled uniformly (aspect preserved) into the arena minus the margin and centred. Map "up"
   (the 0°E meridian direction of the polar grid) points away from you, so the floor is a north-up miniature of the 2D map.
   Typical scale: 1 : 24 000 (39 km of Moon becomes 1.5 m of floor).
2. **Simplify.** Ramer–Douglas–Peucker reduces the route to ≤ 80 straight steps of ≥ 15 mm (tunable). The UI reports
   the maximum deviation from the full-resolution scaled route (≈ 4–6 mm by default).
3. **Steps.** Each step is `rotate(Δheading)` then `move(distance)`. Positive angles turn **left**, matching the official
   examples (`'left' → rotate(90)`). Turns are computed against the *commanded* heading, so 0.1° rounding never accumulates.
4. **Failure replay.** If the simulation failed, the program can stop at that point. It keeps the full route's scale
   and start position, so the robot stops where the simulated rover did.

## What it is and isn't

* It **is** an accurate geometric miniature of the planned route. In tests, the generated `main.py` run against a mock
  Alvik reproduces every waypoint within 0.5 mm (`tests/test_alvik.py`).
* It **is not** a physical replay. Floor friction is not lunar regolith; slopes, slip and sinkage are only *signalled*
  by the LEDs.
* The real Alvik navigates by wheel odometry. Expect a few cm of accumulated drift over a 1.5 m path, more on slippery
  floors or with many turns. Increase the path tolerance or minimum step length for fewer, longer moves (less drift),
  or decrease them for more faithful shape.
