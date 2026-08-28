import numpy as np
from multiprocessing import shared_memory
from multiprocessing.resource_tracker import unregister
import redis
import subprocess
import time
from datetime import datetime
import sqlite3

# Configuration Variables
WIDTH, HEIGHT, CHANNELS = 1920, 1080, 3
RING_BUFFER_SIZE = 60
SHM_NAME = "/wot_ring_buffer"
DB_NAME = "data/data.sqlite"


def save_to_db(filename, clip_type, frames, fps):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO clips (filename, clip_type, frames, fps)
        VALUES (?, ?, ?, ?)
    ''', (filename, clip_type, frames, fps))
    conn.commit()
    conn.close()


# Connect to Redis
r = redis.Redis(host='localhost', port=6379)

while int(r.get('producer:alive') or 0) == 0:
    time.sleep(1)

# Attach to existing shared memory ring buffer created by the producer
try:
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    try:
        unregister(shm._name, 'shared_memory')
        print('unregistered')
    except Exception:
        pass

    ring_buffer = np.ndarray((RING_BUFFER_SIZE, HEIGHT, WIDTH, CHANNELS), dtype=np.uint8, buffer=shm.buf)
    print(f"Successfully attached to shared memory ring buffer: {SHM_NAME}")
except FileNotFoundError:
    print(f"Error: Shared memory block '{SHM_NAME}' not found. Start your producer script first.")
    exit(1)

print("Recorder worker listening on 'record:request' and 'stream:frames'...")

try:
    last_frame_id = '$'
    last_req_id = '$'
    
    recording = False
    waiting_for_anchor = False
    p = None
    output_file = ""
    target_fps = 60
    step = 1
    anchor_frame_id = 0
    stop_frame_target = None
    frame_counter = 0
    clip_type = "default"

    while True:
        # 1. Check for command instructions from 'record:request' stream (non-blocking)
        req_messages = r.xread({'record:request': last_req_id}, count=1, block=10)
        if req_messages:
            _, msg_list = req_messages[0]
            last_req_id, msg_data = msg_list[0]

            
            command = msg_data.get(b'command').decode('utf-8')
            
            if command == 'start' and not recording:
                print('starting')
                target_fps = int(msg_data.get(b'fps'))
                clip_type = msg_data.get(b'clip_type').decode('utf-8')
                
                
                # Calculate modulo step based on 60fps base rate (e.g., 60fps/1fps = 60 step)
                if target_fps <= 0 or 60 % target_fps != 0:
                    print(f"Warning: Invalid or non-clean divisor FPS ({target_fps}). Defaulting step to 1.")
                    step = 1
                else:
                    step = 60 // target_fps

                output_file = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                print(output_file)
                frame_counter = 0

                gop_size = max(target_fps * 4, 10)
                
                # Dynamic FFmpeg command based on requested fps
                ffmpeg_command = [
                    'ffmpeg',
                    '-y',
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-s', f'{WIDTH}x{HEIGHT}',
                    '-pix_fmt', 'bgr24',
                    '-r', str(target_fps),
                    '-i', '-',
                    '-c:v', 'hevc_nvenc',
                    '-tune', 'hq',
                    '-b:v', '0',
                    '-preset', 'p7',
                    '-pix_fmt', 'yuv444p',
                    '-rc', 'vbr',
                    '-cq', '22',
                    '-bf', '4',
                    '-g', str(gop_size),
                    '-rc-lookahead', '32',
                    '-spatial-aq', '1',
                    '-temporal-aq', '1',
                    f'data/{clip_type}/{output_file}.mp4'
                ]
                
                p = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
                recording = True
                waiting_for_anchor = True  # Will lock onto the very next frame's ID as our phase reference
                stop_frame_target = None
                print(f"Started recording: {output_file}.mp4 at {target_fps} FPS (Step interval: {step})")
                
            elif command == 'stop' and recording:
                stop_frame_target = int(msg_data.get(b'stop_frame', 0))
                print(f"Stop requested. Will halt recording once frame tracker passes: {stop_frame_target}")

        # 2. Process video frames from 'stream:frames'
        frame_messages = r.xread({'producer:frames': last_frame_id}, count=1, block=10)
        if frame_messages:
            _, frame_list = frame_messages[0]
            last_frame_id, frame_data = frame_list[0]
            
            current_frame_id = int(frame_data.get(b'frame_id'))
            shm_index = int(frame_data.get(b'shm_index'))
            
            if recording:
                if waiting_for_anchor:
                    anchor_frame_id = current_frame_id
                    waiting_for_anchor = False

                # Modulo arithmetic alignment with the producer's stream
                if (current_frame_id - anchor_frame_id) % step == 0:
                    frame_bytes = ring_buffer[shm_index].tobytes()
                    if p and p.stdin:
                        p.stdin.write(frame_bytes)
                    
                    frame_counter += 1
                    
                    # Check if we have met the stop condition
                    if stop_frame_target is not None and current_frame_id >= stop_frame_target:
                        print(f"Target frame {stop_frame_target} reached (current: {current_frame_id}). Stopping recording.")
                        if p and p.stdin:
                            p.stdin.close()
                        p.wait()
                        
                        save_to_db(output_file, clip_type, frame_counter, target_fps)
                        print(f"Saved recording details to SQLite: {output_file}.mp4, {frame_counter} frames, {target_fps} FPS")
                        
                        recording = False
                        p = None

except KeyboardInterrupt:
    print("\nStopping recorder worker...")
finally:
    if 'p' in locals() and p and p.stdin:
        try:
            p.stdin.close()
            p.wait()
        except Exception:
            pass

    if 'shm' in locals():
        shm.close()
    
    print("Recorder worker shut down cleanly.")