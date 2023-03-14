import hdc_template
import time
import numpy as np
from tqdm import tqdm

# Used only by El-Sewisy 2020 to test the model's real time capabilities when solely relying on PI (no ECD stimulus)

dt = 0.01
t = 10
iterations = t / dt
its_at_a_time = [1, 100, 1000]
stepTimes = []
for its in its_at_a_time:
    times = []
    hdc = hdc_template.generateHDC()
    hdc.setStimulus('hdc_shift_left', lambda _ : 0.1)
    its_remaining = iterations
    time_start = time.time()
    while its_remaining > 0:
        before = time.time()
        if its == 1:
            hdc.step(dt)
        else:
            hdc.step(dt, numsteps=its)
        after = time.time()
        times.append((after - before) / its)
        its_remaining -= its
    stepTimes.append(np.mean(times))
    time_total = time.time() - time_start
    print("Total time: {}".format(time_total))
    print("For {} iterations without stimulus change, the simulation is {:.2f} times as fast as real time".format(its, t / time_total))

for i in range(len(stepTimes)):
    print("Average step time for {} \titeration{} without stimulus change:\t{:.5f} ms ({:.2f} it/s)".format(its_at_a_time[i], " " if its_at_a_time[i] == 1 else "s", 1000 * stepTimes[i], 1/stepTimes[i]))