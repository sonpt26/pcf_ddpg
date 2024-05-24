import time
import random
import threading
from atomiclong import AtomicLong
from time import perf_counter_ns

class Generator:
    def __init__(self, total, tps, use_sleep):
        self.total = total
        self.tps = tps
        self.time_to_sleep = 1/self.tps
        self.use_sleep = use_sleep

    def generate_packet(self):
        # Assuming packet size between 64 bytes (minimum Ethernet frame size) and 1500 bytes (maximum Ethernet frame size)
        return bytearray(random.randint(128, 256))

    # Function to generate packets at a specified rate (in Mbps)
    def generate_packets(self):        
        while True:       
            if self.use_sleep:
                time.sleep(self.time_to_sleep)
            else:
                spinwait_us(self.time_to_sleep*10**6)            
            self.total += 1   
            
def spinwait_us(delay):
    target = perf_counter_ns() + delay * 1000
    while perf_counter_ns() < target:
        pass


total = AtomicLong(0)
for i in range(50):
    gen = Generator(total, 200, True)
    generator_thread = threading.Thread(target=gen.generate_packets)    
    generator_thread.start()

while True:
    time.sleep(1)
    actual_rate_mbps = total.value
    print("Actual tps: {:.2f}".format(actual_rate_mbps))
    total.value = 0