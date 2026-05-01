import spidev
import cv2
import cv2Test
import time
from enum import IntEnum

SPI_BUS          = 0
SPI_DEVICE       = 0
SPI_SPEED_HZ     = 500_000
SPI_MODE         = 0
SPI_FRAME_LEN    = 32
PKT_HEADER       = 0xAA

POLL_INTERVAL_S  = 0.1
CMD_TIMEOUT_S    = 15.0
ACK_TIMEOUT_S    = 5.0
DONE_TIMEOUT_S   = 30.0
CMD_RETRIES      = 3

class ESP32Status(IntEnum):
    READY = 0x00
    BUSY  = 0x01
    ERROR = 0x02
    DONE  = 0x03

class Command(IntEnum):
    NON       = 0x00
    WHITE     = 0x01
    NON_WHITE = 0x02

spi = spidev.SpiDev()
spi.open(SPI_BUS, SPI_DEVICE)
spi.max_speed_hz = SPI_SPEED_HZ
spi.mode         = SPI_MODE

def build_packet(command: Command) -> list[int]:
    hdr  = PKT_HEADER
    cmd  = int(command)
    csum = hdr ^ cmd
    return [hdr, cmd, csum] + [0x00] * (SPI_FRAME_LEN - 3)

def parse_response(raw: list[int]) -> ESP32Status | None:
    try:
        return ESP32Status(raw[0])
    except ValueError:
        print(f"[SPI] Unknown status byte: 0x{raw[0]:02X}")
        return None

def transact(command: Command) -> ESP32Status | None:
    response = spi.xfer2(build_packet(command))
    status   = parse_response(response)
    if command != Command.NON:
        print(f"[SPI] sent={command.name}  status={status.name if status else '?'}")
    return status

def poll_status() -> ESP32Status | None:
    status = transact(Command.NON)
    if status:
        print(f"[POLL] {status.name}")
    return status

def wait_for_ready(timeout: float = CMD_TIMEOUT_S) -> None:
    print("[PI] Waiting for READY...")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        s = poll_status()
        if s == ESP32Status.READY:
            print("[PI] ESP32 is READY")
            return
        if s == ESP32Status.ERROR:
            raise RuntimeError("[PI] ESP32 is in ERROR state -- check hardware")
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"[PI] ESP32 never became READY within {timeout}s")

def send_command(command: Command) -> None:
    wait_for_ready()
    for attempt in range(1, CMD_RETRIES + 1):
        print(f"[PI] Sending {command.name} (attempt {attempt}/{CMD_RETRIES})")
        status = transact(command)
        if status == ESP32Status.BUSY:
            print(f"[PI] ESP32 acknowledged {command.name} with BUSY")
            return
        if status == ESP32Status.ERROR:
            raise RuntimeError("[PI] ESP32 entered ERROR state during command send")
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"[PI] ESP32 never ACK'd command {command.name} after {CMD_RETRIES} attempts")

def wait_for_done(timeout: float = DONE_TIMEOUT_S) -> None:
    print("[PI] Waiting for DONE...")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        s = poll_status()
        if s == ESP32Status.DONE:
            print("[PI] Cycle complete -- ESP32 DONE")
            return
        if s == ESP32Status.ERROR:
            raise RuntimeError("[PI] ESP32 entered ERROR state during cycle")
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"[PI] DONE not received within {timeout}s")

def value_detection(frame_count: int, cap: cv2.VideoCapture) -> Command:
    white_votes    = 0
    nonwhite_votes = 0
    for _ in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            continue
        is_white, frac = cv2Test.classify(frame)
        print(f"[CAM] {'WHITE' if is_white else 'NON-WHITE':10s}  frac={frac:.3f}")
        if is_white:
            white_votes += 1
        else:
            nonwhite_votes += 1
    result = Command.WHITE if white_votes >= nonwhite_votes else Command.NON_WHITE
    print(f"[CAM] Vote result: {result.name}  ({white_votes}W / {nonwhite_votes}N)")
    return result

def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("[PI] Could not open camera")

    print("[PI] Camera open. Giving ESP32 2s to boot...")
    time.sleep(2.0)

    try:
        while True:
            detected = value_detection(120, cap)
            send_command(detected)
            wait_for_done()
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[PI] KeyboardInterrupt -- shutting down.")
    except (RuntimeError, TimeoutError) as exc:
        print(f"\n[PI] Fatal error: {exc}")
    finally:
        cap.release()
        spi.close()
        print("[PI] SPI and camera released.")

if __name__ == "__main__":
    main()