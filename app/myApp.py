# Import modules
from time import time_ns, strftime
from ctypes import POINTER, c_int16, c_uint32
from collections import Counter
from itertools import takewhile
from enum import IntEnum
import json
import socket
import threading

# Import packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, find_peaks, peak_widths, peak_prominences
from scipy import ndimage

# Import picosdk
from picosdk.ps2000 import ps2000
from picosdk.functions import assert_pico2000_ok
from picosdk.ctypes_wrapper import C_CALLBACK_FUNCTION_FACTORY



class TriggerDirection(IntEnum) :
    PS2000_RISING = 0
    PS2000_FALLING = 1

CALLBACK = C_CALLBACK_FUNCTION_FACTORY(None, POINTER(POINTER(c_int16)), c_int16, c_uint32, c_int16, c_int16, c_uint32)

# reimplement this because the other one only takes ctypes
def adc_to_mv(values, range_, bitness=16):
    v_ranges = [10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000]

    return [(x * v_ranges[range_]) / (2**(bitness - 1) - 1) for x in values]

def determine_time_unit(interval_ns):
    unit = 0
    units = ['ns', 'us', 'ms', 's']

    while interval_ns > 5_000:
        interval_ns /= 1000
        unit += 1

    return interval_ns, units[unit]

def seconds_to_samples(time, sample_period=500, sample_units=1e-9):
    return int( time / ( sample_period * sample_units ) )

def samples_to_seconds(samples, sample_period=500, sample_units=1e-9):
    return samples * sample_period * sample_units

def norm_to_mv(normed_val, mean, std):
    return normed_val * std + mean

def mv_to_norm(mv_val, mean, std):
    return (mv_val - mean) / std

def handle_request(c): 
        buffer = c.socket.recv(1024)
        try:
            buffer.data 
            response = "ACQUISITION STARTED"
            c.send(response.encode())

            # Extract settings
            settings = buffer.data
            picoDevice.set_trigger(leading_wave= settings.leading_wave)
            picoDevice.set_samples(expected_pulses= settings.pulses)
            picoDevice.run_streaming()
            
            print("Gathering...")
            valuesA, valuesB, trigger_start = picoDevice.gather()
            picoDevice.stop()

            # Close device
            print('Values gathered: {}'.format(len(valuesA)))
            print('Triggered at: {} samples'.format(trigger_start))

            # Save values
            data = {
                "setup": { 
                    "first_edge": settings.leading_wave,
                    "expected_pulses": settings.pulses,
                    "expected_period": 1 / settings.expected_pulses,
                    "expected_pulse_width": 1 / settings.expected_pulses * 0.4,
                    "sample_interval": 500,
                    "time_interval": 1 + 0.2 * 1 / settings.expected_pulses,
                    "samples": picoDevice.samples
                },
                "signal_A": valuesA,
                "signal_B": valuesB
            }
            timestamp = strftime("%Y%m%d-%H%M%S")
            with open("./app/__pycache__/signals/signal_data_" + timestamp + ".json", 'w') as f:
                json.dump(data, f)
            results = data['signal_A'], data['signal_B']
            c.send("ACQUISITION COMPLETED".encode())
            c.send(results.encode())

            if True:

                # Data processing

                # Make np.array
                A = np.array(valuesA)
                B = np.array(valuesB)

                nsamples = A.size

                # Normalize data
                A_mean = A.mean()
                A_std = A.std()
                B_mean = B.mean()
                B_std = B.std()

                A -= A_mean; A /= A_std
                B -= B_mean; B /= B_std

                # Filter data
                A_filtrd = ndimage.uniform_filter1d(A, 100)  # 100 samples filter width
                B_filtrd = ndimage.uniform_filter1d(B, 100)

                # Frequency and period
                fft_A = np.fft.rfft(A_filtrd, norm="ortho")
                freq_A = abs(fft_A).argmax() * samples_to_seconds(nsamples)
                period_A = 1 / freq_A

                fft_B = np.fft.rfft(B_filtrd, norm="ortho")
                freq_B = abs(fft_A).argmax() * samples_to_seconds(nsamples)
                period_B = 1 / freq_B

                print(f'Recovered frequency: A {freq_A} Hz, B {freq_B} Hz')
                print(f'Recovered period: A {period_A * 1e3} ms, B {period_B * 1e3} ms')

                # Phase
                xcorr = correlate(A_filtrd, B_filtrd)
                dt = np.arange(1-nsamples ,nsamples)
                recovered_timeshift = samples_to_seconds(dt[xcorr.argmax()]) * (360 / period_A)
                print('Recovered offset: {} degrees'.format(recovered_timeshift))

                # Statistics
                def modes(data):
                    freq = Counter(data)
                    mostfreq = freq.most_common()
                    return list(takewhile(lambda x_f: x_f[1] == mostfreq[0][1], mostfreq))

                def get_thresholds(data):
                    _sorted = ndimage.uniform_filter1d(sorted(data), 50)
                    _gradient = np.gradient(_sorted)
                    _gradient -= _gradient.mean()
                    _gradient /= _gradient.std()
                    _gradient[:int(_gradient.size/5)] = _gradient[-int(_gradient.size/5):] = 0
                    _lower = _sorted[np.where(_gradient >= 9*_gradient.std())[0][0]]
                    _upper = _sorted[np.where(_gradient >= 5*_gradient.std())[0][-1]]
                    return _lower, _upper, _gradient

                A_max = np.max(A_filtrd)
                A_min = np.min(A_filtrd)
                B_max = np.max(B_filtrd) 
                B_min = np.min(B_filtrd)
                A_span = A_max - A_min
                B_span = B_max - B_min

                A_lower_threshold, A_upper_threshold, A_grd = get_thresholds(A_filtrd)
                B_lower_threshold, B_upper_threshold, B_grd = get_thresholds(B_filtrd)

                A_mid = np.mean([A_lower_threshold, A_upper_threshold])
                B_mid = np.mean([B_lower_threshold, B_upper_threshold])

                A_clipped = np.clip(A_filtrd, A_lower_threshold, A_upper_threshold)
                A_flipped = 2 * A_mid - A_clipped
                B_clipped = np.clip(B_filtrd, B_lower_threshold, B_upper_threshold)
                B_flipped = 2 * B_mid - B_clipped

                # Peaks
                A_peaks, _ = find_peaks(A_clipped, distance=seconds_to_samples(period_A * 0.6), height=(A_mid, A_mid + A_span), prominence= np.max(A_filtrd) - A_mid, plateau_size=seconds_to_samples(period_A * 0.25))
                A_valleys, _ = find_peaks(A_flipped, distance=seconds_to_samples(period_A * 0.6), height=(A_mid, A_mid + A_span), prominence= np.max(A_filtrd) - A_mid, plateau_size=seconds_to_samples(period_A * 0.25))
                A_peak_widths = peak_widths(A_clipped, A_peaks, rel_height=0.01)
                A_pk_prominences = peak_prominences(A_clipped, A_peaks)[0]
                A_valley_widths = peak_widths(A_flipped, A_valleys, rel_height=0.01)
                A_nr_peaks = len(A_peaks)

                B_peaks, _ = find_peaks(B_clipped, distance=seconds_to_samples(period_A * 0.6), height=(B_mid, B_mid + B_span), prominence= np.max(B_filtrd) - B_mid, plateau_size=seconds_to_samples(period_A * 0.25))
                B_valleys, _ = find_peaks(B_flipped, distance=seconds_to_samples(period_A * 0.6), height=(B_mid, B_mid + B_span), prominence= np.max(B_filtrd) - B_mid, plateau_size=seconds_to_samples(period_A * 0.25))
                B_peak_widths = peak_widths(B_clipped, B_peaks, rel_height=0.01)
                B_pk_prominences = peak_prominences(B_clipped, B_peaks)[0]
                B_valley_widths = peak_widths(B_flipped, B_valleys, rel_height=0.01)
                B_nr_peaks = len(B_peaks)

                print('Nr. of peaks A: {}'.format(len(A_peaks)))
                print('Nr. of peaks B: {}'.format(len(B_peaks)))
                print('Widths of A [ms]: \n{}'.format(1000 * samples_to_seconds(A_peak_widths[0])))
                print('Widths of B [ms]: \n{}'.format(1000 * samples_to_seconds(B_peak_widths[0])))
                print(A_peak_widths[2:])

                A_bounces = np.empty((3, A_nr_peaks * 2))
                if A_nr_peaks > 0:
                    A_bounces[1][0] = np.where(A_filtrd[:int(A_peak_widths[2][0])] > A_lower_threshold)[0][0]
                    A_bounces[2][0] = A_peak_widths[2][0]
                    for x in range(0, A_nr_peaks - 1):
                        A_bounces[1][2 * x + 1] = A_peak_widths[3][x]
                        A_bounces[2][2 * x + 1] = A_valley_widths[2][x]
                    for x in range(1, A_nr_peaks):
                        A_bounces[1][2 * x] = A_valley_widths[3][x - 1]
                        A_bounces[2][2 * x] = A_peak_widths[2][x]
                    A_bounces[1][A_nr_peaks * 2 - 1] = A_peak_widths[3][A_nr_peaks - 1]
                    A_bounces[2][A_nr_peaks * 2 - 1] = int(A_peak_widths[3][A_nr_peaks - 1]) + np.where(A_filtrd[int(A_peak_widths[3][A_nr_peaks - 1]):] > A_lower_threshold)[0][-1]
                    for x in range(0, A_nr_peaks * 2):
                        A_bounces[0][x] = A_bounces[2][x] - A_bounces[1][x]


                B_bounces = np.empty((3, B_nr_peaks * 2))
                if B_nr_peaks > 0:
                    B_bounces[1][0] = np.where(B_filtrd[:int(B_peak_widths[2][0])] > B_lower_threshold)[0][0]
                    B_bounces[2][0] = B_peak_widths[2][0]
                    for x in range(0, B_nr_peaks - 1):
                        B_bounces[1][2 * x + 1] = B_peak_widths[3][x]
                        B_bounces[2][2 * x + 1] = B_valley_widths[2][x]
                    for x in range(1, B_nr_peaks):
                        B_bounces[1][2 * x] = B_valley_widths[3][x - 1]
                        B_bounces[2][2 * x] = B_peak_widths[2][x]
                    B_bounces[1][B_nr_peaks * 2 - 1] = B_peak_widths[3][B_nr_peaks - 1]
                    B_bounces[2][B_nr_peaks * 2 - 1] = int(B_peak_widths[3][B_nr_peaks - 1]) + np.where(B_filtrd[int(B_peak_widths[3][B_nr_peaks - 1]):] > B_lower_threshold)[0][-1]
                    for x in range(0, B_nr_peaks * 2):
                        B_bounces[0][x] = B_bounces[2][x] - B_bounces[1][x]

                print("A Bounces in samples:")
                print(A_bounces)
                print("A Bounces in milliseconds:")
                print(samples_to_seconds(A_bounces)*1000)




                fig, axs = plt.subplots(4)
                _, units = determine_time_unit(nsamples * sample_interval)
                interval = samples_to_seconds(nsamples) * 1000
                n = 0

                axs[n].set_xlabel('time/{}'.format(units))
                axs[n].hlines(A_mid, 0, interval, linestyle='dotted')
                for i, (x1, x2, s) in enumerate(zip(A_bounces[1], A_bounces[2], A_bounces[0])):
                    s = samples_to_seconds(s) * 1000
                    x1 = samples_to_seconds(x1) * 1000
                    x2 = samples_to_seconds(x2) * 1000
                    axs[n].text(x1, A_mid - 0.2 + 0.3 * (i % 2), '{0:.2f} ms'.format(s), size='small')
                    axs[n].hlines(A_mid, x1, x2, color = 'red')
                axs[n].hlines(A_upper_threshold, 0, interval, color = 'green')
                axs[n].hlines(A_lower_threshold, 0, interval, color = 'green')
                axs[n].plot(np.linspace(0, interval, nsamples), A_filtrd)
                n += 1


                axs[n].set_xlabel('time/{}'.format(units))
                axs[n].plot(np.linspace(0, interval, nsamples), A_clipped)
                axs[n].plot(samples_to_seconds(A_peaks)*1000, A_clipped[A_peaks], 'x')
                axs[n].plot(samples_to_seconds(A_valleys)*1000, A_clipped[A_valleys], 'x')
                for i, (x1, x2, s) in enumerate(zip(A_bounces[1], A_bounces[2], A_bounces[0])):
                    s = samples_to_seconds(s) * 1000
                    x1 = samples_to_seconds(x1) * 1000
                    x2 = samples_to_seconds(x2) * 1000
                    axs[n].text(x1, A_mid - 0.1 + 0.2 * (i % 2), '{0:.2f} ms'.format(s), size='small')
                    axs[n].hlines(A_mid, x1, x2, color = 'red')
                for x1, x2 in zip(A_peak_widths[2], A_peak_widths[3]):
                    x1 = samples_to_seconds(x1) * 1000
                    x2 = samples_to_seconds(x2) * 1000
                    axs[n].hlines(A_mid + 0.1, x1, x2, color='orange')
                for x1, x2 in zip(A_valley_widths[2], A_valley_widths[3]):
                    x1 = samples_to_seconds(x1) * 1000
                    x2 = samples_to_seconds(x2) * 1000
                    axs[n].hlines(A_mid - 0.1, x1, x2, color='green') 
                n += 1


                axs[n].set_xlabel('time/{}'.format(units))
                axs[n].hlines(B_mid, 0, interval, linestyle='dotted')
                for i, (x1, x2, s) in enumerate(zip(B_bounces[1], B_bounces[2], B_bounces[0])):
                    s = samples_to_seconds(s) * 1000
                    x1 = samples_to_seconds(x1) * 1000
                    x2 = samples_to_seconds(x2) * 1000
                    axs[n].text(x1, B_mid - 0.2 + 0.3 * (i % 2), '{0:.2f} ms'.format(s), size='small')
                    axs[n].hlines(B_mid, x1, x2, color = 'red')
                axs[n].hlines(B_upper_threshold, 0, interval, color = 'green')
                axs[n].hlines(B_lower_threshold, 0, interval, color = 'green')
                axs[n].plot(np.linspace(0, interval, nsamples), B_filtrd)
                n += 1


                axs[n].set_xlabel('time/{}'.format(units))
                axs[n].plot(np.linspace(0, interval, nsamples), B_clipped)
                axs[n].plot(samples_to_seconds(B_peaks)*1000, B_clipped[B_peaks], 'x')
                axs[n].plot(samples_to_seconds(B_valleys)*1000, B_clipped[B_valleys], 'x')
                for i, (x1, x2, s) in enumerate(zip(B_bounces[1], B_bounces[2], B_bounces[0])):
                    s = samples_to_seconds(s) * 1000
                    x1 = samples_to_seconds(x1) * 1000
                    x2 = samples_to_seconds(x2) * 1000
                    axs[n].text(x1, B_mid - 0.1 + 0.2 * (i % 2), '{0:.2f} ms'.format(s), size='small')
                    axs[n].hlines(B_mid, x1, x2, color = 'red')
                for x1, x2 in zip(B_peak_widths[2], B_peak_widths[3]):
                    x1 = samples_to_seconds(x1) * 1000
                    x2 = samples_to_seconds(x2) * 1000
                    axs[n].hlines(B_mid + 0.1, x1, x2, color='orange')
                for x1, x2 in zip(B_valley_widths[2], B_valley_widths[3]):
                    x1 = samples_to_seconds(x1) * 1000
                    x2 = samples_to_seconds(x2) * 1000
                    axs[n].hlines(B_mid - 0.1, x1, x2, color='green') 
                n += 1


                """ axs[n].set_xlabel('time/{}'.format(units))
                axs[n].hlines(B_mid, 0, interval, color = 'red')
                axs[n].hlines(B_upper_threshold, 0, interval, color = 'green')
                axs[n].hlines(B_lower_threshold, 0, interval, color = 'green')
                axs[n].plot(np.linspace(0, interval, nsamples), B_filtrd)
                n += 1


                contour_heights = B_clipped[B_peaks] - B_pk_prominences
                B_flip_valley_widths = []
                B_flip_valley_widths[0:0] = B_valley_widths
                B_flip_valley_widths[1] = 2 * B_mid - B_valley_widths[1]
                axs[n].set_xlabel('time/{}'.format(units))
                axs[n].plot(B_clipped)
                axs[n].plot(B_peaks, B_clipped[B_peaks], 'x')
                axs[n].plot(B_valleys, B_clipped[B_valleys], 'x')
                axs[n].hlines([B_mid + 0.1 for x in B_peak_widths[1]], *B_peak_widths[2:], color='orange')
                axs[n].hlines([B_mid - 0.1 for x in B_valley_widths[1]],*B_valley_widths[2:], color='magenta')  # 2 * B_mid - B_clipped
                # axs[n].vlines(x=B_peaks, ymin=contour_heights, ymax=B_clipped[B_peaks])
                n += 1
                """
                """ axs[n].set_xlabel('time/{}'.format(units))
                axs[n].plot(A_flipped)
                axs[n].plot(A_valleys, A_flipped[A_valleys], 'x')
                axs[n].hlines(*A_valley_widths[1:])
                n += 1 """

                """ axs[n].set_xlabel('time/{}'.format(units))
                axs[n].hlines(B_mid, 0, 1000, color = 'red')
                axs[n].plot(np.linspace(0, interval, nsamples), B)
                n += 1 """

                """ axs[n].set_xlabel('time/{}'.format(units))
                axs[n].hlines(B_mid, 0, 1000, color = 'red')
                axs[n].hlines(B_upper_threshold, 0, 1000, color = 'green')
                axs[n].hlines(B_lower_threshold, 0, 1000, color = 'green')
                axs[n].plot(np.linspace(0, interval, nsamples), B_filtrd)
                n += 1

                axs[n].hlines(B_upper_threshold, 0, 2e6, color = 'green')
                axs[n].hlines(B_lower_threshold, 0, 2e6, color = 'green')
                axs[n].plot(ndimage.uniform_filter1d(sorted(B_filtrd), 50), color='blue')
                n += 1

                #axs[n].hist(A_filtrd, density=True, bins=1000)
                axs[n].plot(B_grd, color='green')
                n += 1 """

                """ 
                axs[n].hlines(A_upper_threshold, 0, 2e6, color = 'green')
                axs[n].hlines(A_lower_threshold, 0, 2e6, color = 'green')
                axs[n].plot(ndimage.uniform_filter1d(sorted(A_filtrd), 50), color='blue')
                n += 1
                """
                """ #axs[n].hist(A_filtrd, density=True, bins=1000)
                axs[n].plot(A_grd, color='green')
                n += 1 """


                plt.show()
        except:
            response = f"[+] ERROR STARTING ACQUISITION DUE TO MISSING DATA"
            c.send(response.encode())
        



class StreamingDevice:
    def __init__(self, gather_values, sample_interval, potential_range=ps2000.PS2000_VOLTAGE_RANGE['PS2000_1V'], pretrigger = 4000):
        self.device = ps2000.open_unit()
        # signal generator for testing
        #res = ps2000.ps2000_set_sig_gen_built_in(self.device.handle, 1_000_000, 2_000_000, 1, 32, 32, 0, 0, 0, 0)
        #assert_pico2000_ok(res)

        self.potential_range = potential_range
        self.gather_values = gather_values
        self.sample_interval = sample_interval
        self.pretrigger = pretrigger # seconds_to_samples(2e-3) # 2 millisecond pretrigger


        res = ps2000.ps2000_set_channel(self.device.handle, ps2000.PICO_CHANNEL["A"], True, True, potential_range)
        assert_pico2000_ok(res)
        res = ps2000.ps2000_set_channel(self.device.handle, ps2000.PICO_CHANNEL["B"], True, True, potential_range)
        assert_pico2000_ok(res)
        self.set_trigger(leading_wave= 'A')

    def set_trigger(self, leading_wave):
        threshold = int(32_767 / 2) # about half the potential range in ADC values (-32_767 -> +32_767)
        direction = TriggerDirection.PS2000_RISING
        delay = 0 # percent -100% -> +100%
        auto_trigger = 10_000 # milliseconds
        res = ps2000.ps2000_set_trigger(
            self.device.handle, 
            ps2000.PICO_CHANNEL[leading_wave], 
            threshold, 
            direction, 
            delay, 
            auto_trigger)
        assert_pico2000_ok(res)

    def set_samples(self, expected_pulses):
        expected_period = 1 / expected_pulses
        time_interval = 1 + 0.2 * expected_period
        self.samples = seconds_to_samples(time_interval, self.sample_interval)

    def set_pretrigger(self, expected_pulses):
        expected_period = 1 / expected_pulses # seconds
        self.pretrigger = seconds_to_samples(0.5 * expected_period)

    def run_streaming(self):
        # start 'fast-streaming' mode
        res = ps2000.ps2000_run_streaming_ns(
            self.device.handle,
            self.sample_interval,
            ps2000.PS2000_TIME_UNITS['PS2000_NS'], #Units: Nanoseconds
            22_000_000, #100_000, # max_samples
            False,  # auto_stop
            1,  # noOfSamplesPerAggregate
            50_000  # overview_buffer_size
        )
        assert_pico2000_ok(res)

        self.start_time = time_ns()
        self.end_time = time_ns()

    def close(self):
        ps2000.ps2000_stop(self.device.handle)
        self.device.close()

    def gather(self):
        adc_valuesA = []
        adc_valuesB = []
        pretriggerA = []
        pretriggerB = []
        triggered = False
        triggered_at = 0

        def get_overview_buffers(buffers, _overflow, _triggered_at, _triggered, _auto_stop, n_values):
            nonlocal triggered
            nonlocal triggered_at

            if not triggered:
                pretriggerA.extend(buffers[0][0:n_values])
                pretriggerB.extend(buffers[2][0:n_values])

            if _triggered:
                triggered = True
                triggered_at = len(pretriggerA) + _triggered_at
                self.start_time = time_ns() + (_triggered_at - self.pretrigger - n_values ) * self.sample_interval

            if triggered:
                adc_valuesA.extend(buffers[0][0:n_values])
                adc_valuesB.extend(buffers[2][0:n_values])

            
        callback = CALLBACK(get_overview_buffers)

        while ((len(adc_valuesA) < self.gather_values) or not triggered):
            ps2000.ps2000_get_streaming_last_values(
                self.device.handle,
                callback
            )

            if len(pretriggerA) > self.pretrigger:
                pretriggerA[0:-self.pretrigger] = []
                pretriggerB[0:-self.pretrigger] = []
            

        adc_valuesA[0:0] = pretriggerA[-self.pretrigger:]
        adc_valuesB[0:0] = pretriggerB[-self.pretrigger:]

        self.end_time = time_ns()

        return adc_to_mv(adc_valuesA, self.potential_range), adc_to_mv(adc_valuesB, self.potential_range), triggered_at

    def stop(self):
        ps2000.ps2000_stop(self.device.handle)


# Setup
#first_edge = 'A' # A or B, depending on the direction of rotation
expected_pulses = 8 # how many pulses should the encoder have in one turn / second
expected_period = 1 / expected_pulses # seconds
expected_pulse_width = expected_period * 0.4 # seconds
sample_interval = 500 # sample interval in nanoseconds
time_interval = 1 + 0.2 * expected_period # time interval for testing, in seconds
samples = seconds_to_samples(time_interval, sample_interval) # how many samples in the time interval
pretrigger = seconds_to_samples(0.5 * expected_period)



# Setup server
bind_ip = "0.0.0.0" 
bind_port = 8000
server = socket.create_server((bind_ip, bind_port))

# we tell the server to start listening with a maximum backlog of connections set to 5
server.listen(5) 
print(f"[+] Listening on port {bind_ip} : {bind_port}")  

# Start device
picoDevice = StreamingDevice(samples, sample_interval, potential_range=ps2000.PS2000_VOLTAGE_RANGE['PS2000_20V'], pretrigger=pretrigger)

# Collect data
# main loop
while True:
    # wait trigger
    c_sock, addr = server.accept() 
    print(f"[+] Connection established from: {addr[0]}:{addr[1]} | Socket: {c_sock}")
    print(f"[+] Accepted connection from: {addr[0]}:{addr[1]}")
    
    if c_sock.recv(1024):
        request = c_sock.recv(1024).decode()
        print(f"[+] Recieved: {request}")
        match request.keyword:
            case "START ACQUISITION":
                client_handler = threading.Thread(handle_request, args=(c_sock))
                client_handler.start()
                client_handler.join()
            case _:
                print("Unknown request received")
    

    
    #Close device
    #picoDevice.close()



