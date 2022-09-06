import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import numpy as np

from utils.myplot import savefig

def sim_TCP(NUM_TEST = 1000, rtt = 0.5, n_packets = 100, size_packet = 1.5, bandwidth = 1000):
    ''' Simulation with parameters
    NUM_TEST = 1000 # number of tests
    rtt = 0.5   # unit: second
    n_packets = 100 # number of packets
    size_packet = 1.5   # size of a packet, unit: KB, limited by MTU
    bandwidth = 1000    # unit: Kbps
    '''

    y_udp = []
    y_tcp = []
    y_re = []
    x = []
    for loss_rate in np.arange(0, 0.201, 0.003):
        x.append(loss_rate)
        delay_re = []
        for i in range(NUM_TEST):
            delay_re_tmp = 0.
            for packet in range(n_packets):
                p = random.random()
                if p <= loss_rate:  # lost
                    while(True):
                        pp = random.random()
                        if pp <= loss_rate:
                            delay_re_tmp += rtt
                        else:
                            break
            delay_re.append(delay_re_tmp)
        delay_re = sum(delay_re)/len(delay_re)
        delay_udp = n_packets * size_packet * 8 / bandwidth
        delay_tcp = n_packets * size_packet * 8 / bandwidth + rtt + delay_re
        y_udp.append(1000 * (delay_udp / n_packets))
        y_tcp.append(1000 * (delay_tcp / n_packets))
        y_re.append(1000 * (delay_re / n_packets))
        # print(loss_rate, delay_re, delay_tcp, delay_udp)
    
    return x, y_udp, y_tcp, y_re


if __name__ == '__main__':
    # x, y_udp, y_tcp, y_re = sim_TCP(size_packet=1.5)
    # y1 = [100*y_re[i]/y_tcp[i] for i in range(len(y_tcp))]

    # x, y_udp, y_tcp, y_re = sim_TCP(size_packet=1)
    # y2 = [100*y_re[i]/y_tcp[i] for i in range(len(y_tcp))]
    # xi = list(range(len(x)))
    # x = [int(i*100) for i in x]

    # plt.plot(xi, y1, label='High bitrate')
    # plt.plot(xi, y2, label='Low bitrate')
    # plt.xlabel("Loss rate (%)")
    # plt.xticks(xi, x, rotation=45)
    # plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
    # plt.ylabel("% of retransmission delay in total TCP delay")
    # plt.legend()
    # savefig(plt, "./result/delay_pcnt", fontsize=10)

    # import pandas as pd
    # df1 = pd.read_csv('./result/bitrate.csv', delim_whitespace=True, dtype=float)
    # df1.columns = ['time (s)', 'sending_rate (Kbps)', 'bandwidth (Kbps)']
    # df2 = pd.read_csv('./result/delay.csv', delim_whitespace=True, dtype=float)
    # df2.columns = ['time (s)', 'e2e_delay (ms)']
    # x1 = df1.loc[:,"time (s)"].tolist()
    # y1 = df1.loc[:,"bandwidth (Kbps)"].tolist()
    # x2 = df2.loc[:,"time (s)"].tolist()
    # y2 = df2.loc[:,"e2e_delay (ms)"].tolist()
    # plt.clf()
    # plt.subplot(2, 1, 1)
    # plt.plot(x1, y1)
    # plt.ylabel("Bandwidth (Kbps)")
    # plt.subplot(2, 1, 2)
    # plt.plot(x2, y2)
    # plt.xlabel("Time (s)")
    # plt.ylabel("End-to-end delay (ms)")
    # savefig(plt, "./result/delay_bd", fontsize=10)
    # plt.close()