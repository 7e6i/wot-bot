import numpy as np
from multiprocessing import shared_memory
import redis
import time
import cv2
from pipewire_capture import PortalCapture, CaptureStream, is_available

if not is_available():
    print("PipeWire capture not available.")
    exit(1)

TARGET_FPS = 60
RING_BUFFER_SIZE = 60
WIDTH, HEIGHT, CHANNELS = 1920, 1080, 3
SHM_NAME = "/wot_ring_buffer"
FRAME_BYTES = WIDTH * HEIGHT * CHANNELS

r = redis.Redis(host='localhost', port=6379)
r.set("producer:alive", 0)

# Clean up any crashed memory blocks
try:
    existing_shm = shared_memory.SharedMemory(name=SHM_NAME)
    existing_shm.unlink()
except FileNotFoundError:
    pass

shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=FRAME_BYTES * RING_BUFFER_SIZE)
ring_buffer = np.ndarray((RING_BUFFER_SIZE, HEIGHT, WIDTH, CHANNELS), dtype=np.uint8, buffer=shm.buf)

portal = PortalCapture()
print("Opening Wayland window picker.")

# The new API returns a session object directly
session = portal.select_window()

if not session:
    print("Window selection canceled. Exiting.")
    shm.close()
    shm.unlink()
    exit(1)

# Extract stream information from the session object, not the portal
stream = CaptureStream(
    session.fd, 
    session.node_id, 
    session.width, 
    session.height, 
    capture_interval=1.0/TARGET_FPS
)
stream.start()


print("Recording to /dev/shm and publishing to Redis 'stream:frames'")
r.set("producer:alive", 1)

frame_count = 0
try:
    # Use session.window_invalid if stream.window_invalid throws an error on this new version
    while not stream.window_invalid: 
        t0 = time.time()

        frame = stream.get_frame() 
        if frame is not None:
            resized_frame = cv2.resize(frame, (WIDTH, HEIGHT))
            
            if resized_frame.shape[2] == 4:
                resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGRA2BGR)
            
            index = frame_count % RING_BUFFER_SIZE
            ring_buffer[index] = resized_frame
            
            r.xadd(
                "producer:frames", 
                {"frame_id": frame_count, "shm_index": index},
                maxlen=1000,
                approximate=True
            )
            frame_count += 1

        elapsed = time.time() - t0
        time.sleep(1.0 / TARGET_FPS)
except KeyboardInterrupt:
    print("\nShutting down stream...")
finally:
    stream.stop()
    try:
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass


    print("Cleaned up shared memory successfully.")
    r.set("producer:alive", 0)