"""
Rumble — Real-time system audio to Xbox controller haptic feedback.
Captures Windows system audio via WASAPI Loopback, applies a low-pass filter,
and drives Xbox controller vibration based on bass intensity.
"""

import ctypes
import struct
import threading
import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt

# ── Config ──────────────────────────────────────────────────────────
CUTOFF_HZ = 60          # Low-pass filter cutoff
FILTER_ORDER = 4         # Butterworth filter order (steepness)
GAIN = 3.0               # Vibration intensity multiplier
BLOCK_SIZE = 256         # Audio buffer size (lower = less latency, more CPU)
SMOOTHING = 0.15         # Envelope smoothing factor (0 = instant, 1 = frozen)

# ── XInput via ctypes ───────────────────────────────────────────────
xinput = ctypes.windll.xinput1_4

class XINPUT_VIBRATION(ctypes.Structure):
    _fields_ = [("wLeftMotorSpeed", ctypes.c_ushort),
                ("wRightMotorSpeed", ctypes.c_ushort)]

def set_vibration(controller_id, left, right):
    vibration = XINPUT_VIBRATION(int(left), int(right))
    xinput.XInputSetState(controller_id, ctypes.byref(vibration))

# ── Low-pass filter ────────────────────────────────────────────────
def make_lowpass(cutoff, fs, order=4):
    nyq = fs / 2
    return butter(order, cutoff / nyq, btype='low', output='sos')

# ── Main ───────────────────────────────────────────────────────────
def find_loopback_device():
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0 and 'loopback' in d['name'].lower():
            return i
    for i, d in enumerate(devices):
        hostapi = sd.query_hostapis(d['hostapi'])
        if hostapi['name'] == 'Windows WASAPI' and d['max_input_channels'] > 0:
            if d['name'].startswith('Speaker') or d['name'].startswith('Lautsprecher'):
                return i
    return None

def run():
    device_idx = find_loopback_device()
    if device_idx is None:
        print("No WASAPI loopback device found.")
        print("Available devices:")
        print(sd.query_devices())
        return

    device_info = sd.query_devices(device_idx)
    samplerate = int(device_info['default_samplerate'])
    channels = device_info['max_input_channels']

    print(f"Using: {device_info['name']}")
    print(f"Sample rate: {samplerate} Hz, Channels: {channels}")
    print(f"Filter: {CUTOFF_HZ} Hz low-pass, Gain: {GAIN}x")
    print(f"Block size: {BLOCK_SIZE} (~{BLOCK_SIZE/samplerate*1000:.1f} ms)")
    print("Press Ctrl+C to stop.\n")

    sos = make_lowpass(CUTOFF_HZ, samplerate, FILTER_ORDER)
    envelope = [0.0]

    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Audio status: {status}")

        mono = indata.mean(axis=1) if channels > 1 else indata[:, 0]
        filtered = sosfilt(sos, mono)
        rms = np.sqrt(np.mean(filtered ** 2))
        envelope[0] = SMOOTHING * envelope[0] + (1 - SMOOTHING) * rms
        intensity = min(envelope[0] * GAIN * 65535, 65535)
        set_vibration(0, int(intensity), int(intensity * 0.6))

    try:
        with sd.InputStream(device=device_idx, channels=channels,
                            samplerate=samplerate, blocksize=BLOCK_SIZE,
                            callback=audio_callback):
            print("Rumbling... ♫")
            threading.Event().wait()
    except KeyboardInterrupt:
        set_vibration(0, 0, 0)
        print("\nStopped. Motors off.")

if __name__ == '__main__':
    run()
