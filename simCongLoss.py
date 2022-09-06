import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import random
import numpy as np
import csv

from utils.myplot import savefig


if __name__ == '__main__':
    max_timestamp = -1
    bw_timestamp = []
    bw_value = []
    with open('./dataset/bw.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            bw_timestamp.append(int(row[0]))
            if int(row[0]) > max_timestamp:
                max_timestamp = int(row[0])
            bw_value.append(float(row[1]))
            line_count += 1
    sr_timestamp = []
    sr_value = []
    with open('./dataset/sr.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            sr_timestamp.append(int(row[0]))
            sr_value.append(float(row[1]))
            line_count += 1
    
    ### turn the segmented timestamp to real timestamp
    bw_real = [0]*max_timestamp
    sr_real = [0]*max_timestamp
    prev_v = bw_real[0]
    prev_t = bw_timestamp[0]
    for i in range(1, len(bw_timestamp)):
        if bw_timestamp[i] != prev_t:
            for j in range(prev_t, bw_timestamp[i]):
                bw_real[j] = prev_v
        prev_v = bw_value[i]
        prev_t = bw_timestamp[i]

    prev_v = sr_real[0]
    prev_t = sr_timestamp[0]
    for i in range(1, len(sr_timestamp)):
        if sr_timestamp[i] != prev_t:
            for j in range(prev_t, sr_timestamp[i]):
                sr_real[j] = prev_v
        prev_v = sr_value[i]
        prev_t = sr_timestamp[i]

    loss = []
    queue_size = 0
    # The general rule of thumb is that you need 50ms of line-rate output queue buffer, so for for a 10G switch, there should be around 60MB of buffer.
    # https://fasterdata.es.net/network-tuning/router-switch-buffer-size-issues/
    max_queue = 15000 # unit: bytes
    for i in range(max_timestamp):
        queue_size += (sr_real[i] - bw_real[i])
        queue_size = max(0, queue_size)
        if queue_size > max_queue:
            loss.append((queue_size-max_queue) / sr_real[i])
            queue_size = max_queue
        else:
            loss.append(0)
    

    plt.rcParams['font.size'] = 15
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 3))
    ax1.plot(bw_timestamp, [i /1000000 for i in bw_value], label='Bandwidth')
    ax1.plot(sr_timestamp, [i /1000000 for i in sr_value], label='Sending Rate')
    ax2.plot(range(max_timestamp), [int(i*100) for i in loss])
    plt.xlabel("Time (hours)")
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x / 3600)}'))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x / 3600)}'))
    ax1.set(ylabel="Rate (Mbps)")
    ax2.set(ylabel="Loss (%)")
    ax1.legend(loc=4)
    savefig(plt, "./result/bw_loss", ftype="pdf")
    plt.close()