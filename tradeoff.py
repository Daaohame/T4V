import os
import time
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib import colors as mcolors

from dnn import i3d
from utils.myplot import savefig
from utils import video_lib, netsim
import decord   # must be imported after gluoncv


# conf_bench = [1.0 for i in samples]
# label_bench = [-1 for i in samples]
loss = [0.3]
expected_loss = list(np.arange(max(0.01, loss[0] - 0.1), loss[0] + 0.11, 0.05).round(2))
sal_threshold = 0.01


baselines = {
    "TCP": 0,  # TCP must be the first as groundtruth
    "UDP": 1,
    "T4V": 2,
    # 'naive T4V': 3,
    "FEC": 3,
    "Selective Re-tx": 4,
}


def compute_diff(video_path):
    vr = decord.VideoReader(video_path)
    frame_id_list = list(range(0, 64, 2))
    video_data = vr.get_batch(frame_id_list).asnumpy()
    maxlen = len(video_data)
    video_data = video_data.astype("int16")  # original type: uint8
    max_diff = np.size(video_data) * np.iinfo("uint8").max
    # compute the difference
    tt0 = time.time()
    diff = []
    for i in range(maxlen):
        diff_i = []
        for j in range(maxlen):
            diff_i.append(np.sum(np.abs(video_data[i] - video_data[j])))
        diff.append(diff_i)
    # normalization
    diff = np.array(diff)
    normalized_diff = diff / max_diff
    tt1 = time.time()
    print(f"!!!!!!! {tt1 - tt0} for {video_path}")
    return normalized_diff


def result_of_transmission(mode, frame_id_list, loss_rate, indspsb, utility, diff, ddl):
    # tx_mode is for transmission
    if mode == "T4V":
        tx_mode = "T4V_n"
    elif mode == "Selective Re-tx" or mode == "naive T4V":
        tx_mode = "T4V"
    elif mode == "FEC":
        tx_mode = "UDP"
    else:
        tx_mode = mode

    delay, frame_id_list_rx = netsim.transmit(
        frame_id_list, loss_rate, utility, diff, indspsb, tx_mode, ddl
    )
    frame_id_list_lost = list(set(frame_id_list) - set(frame_id_list_rx))
    frame_id_list_lost.sort()

    return delay, frame_id_list_rx, frame_id_list_lost


def simulate(net, path, sal, label, loss_rate, mode, args):
    print(f"~~~~~{path}~~~~~")
    interval = args.interval
    n_iter = args.num_iter if threshold > 0.0 else 1
    ddl = args.deadline
    tt0 = time.time()
    diff = compute_diff(path)
    tt1 = time.time()
    print(f"!!!!!!! {tt1 - tt0} for {path}")
    assert mode in baselines, f"mode must be one of {baselines}"
    vr = decord.VideoReader(path)
    frame_id_list = list(range(0, 64, interval))
    original_len = len(frame_id_list)

    if mode == "T4V" or mode == "naive T4V":
        indspsb = [no * 2 for (no, val) in enumerate(sal) if val >= sal_threshold]
        # for i in [0, frame_id_list[-1]]:
        #     if i not in indspsb:
        #         indspsb.append(i)
        if mode == "random T4V" and len(indspsb) > 0:
            indspsb = random.sample(range(0, 64, 2), len(indspsb))
        print(f"    mode: {mode}, indispensable: {indspsb}")
    elif mode == "Selective Re-tx":
        indspsb = [0, frame_id_list[-1]]
        print(f"    mode: {mode}, indispensable: {indspsb}")
    else:
        indspsb = []
        print(f"    mode: {mode}, indispensable: {indspsb}")

    if mode == "FEC":
        from math import ceil

        frame_id_list_FEC = []
        original_list = frame_id_list.copy()
        for el in expected_loss:
            # fec_rate = fec_lookup(el, 1000)
            fec_rate = el
            print(f"        FEC: expected loss {el} -> FEC rate {fec_rate}")
            redundancy_len = ceil(original_len * fec_rate)
            redundancy_list = list(range(64, 64 + redundancy_len * interval, interval))
            frame_id_list_FEC.append((redundancy_len, original_list + redundancy_list))

    if mode == "FEC":
        total = [0] * len(expected_loss)
        correct = [0] * len(expected_loss)
        conf_tmp = [[] for _ in range(len(expected_loss))]
        delay_tmp = [[] for _ in range(len(expected_loss))]
    else:
        total = 0
        correct = 0
        conf_tmp = []
        delay_tmp = []

    for j in range(n_iter):

        if mode == "FEC":
            for i in range(len(expected_loss)):
                delay, frame_id_list_rx, frame_id_list_lost = result_of_transmission(
                    mode, frame_id_list_FEC[i][1], loss_rate, indspsb, sal, diff, ddl
                )
                ### frame_id_list_FEC[i][1] is the list, while [0] is the redundancy length

                if (
                    len(frame_id_list_lost) > frame_id_list_FEC[i][0]
                ):  # cannot restore the lost
                    frame_id_list_rx = [
                        i for i in frame_id_list_rx if i <= original_list[-1]
                    ]
                    frame_id_list_rx = netsim.padding(
                        frame_id_list_rx, original_len, interval
                    )
                    assert (
                        len(frame_id_list_rx) == original_len
                    ), "Padding error for FEC"
                else:  # can restore the lost
                    frame_id_list_rx = original_list

                video_data = vr.get_batch(frame_id_list_rx).asnumpy()
                print(f"        lost: {frame_id_list_lost}, delay: {delay}", end="")
                pred = net.predict(
                    video_lib.video_transform_gluoncv(video_data, device=args.device)
                )
                result = np.argmax(pred)
                total[i] += 1
                if result == label:
                    correct[i] += 1
                print(f" correctness:{result == label}")
                delay_tmp[i].append(delay)

        else:
            delay, frame_id_list_rx, frame_id_list_lost = result_of_transmission(
                mode, frame_id_list, loss_rate, indspsb, sal, diff, ddl
            )
            frame_id_list_rx = netsim.padding(frame_id_list_rx, original_len, interval)
            video_data = vr.get_batch(frame_id_list_rx).asnumpy()
            print(f"        lost: {frame_id_list_lost}, delay: {delay}", end="")
            pred = net.predict(
                video_lib.video_transform_gluoncv(video_data, device=args.device)
            )
            result = np.argmax(pred)
            # if loss_rate == 0.0:
            #     label_bench[no] = result
            #     conf_bench[no] = pred[label_bench[no]]
            # print(f"        Label {label}, result {result}, label_bench {label_bench[no]}")
            # # conf = pred[label]/pred[result]
            # conf = pred[label_bench[no]]
            # conf_tmp.append(conf)
            total += 1
            if result == label:
                correct += 1
            print(f"correctness:{result == label}")
            delay_tmp.append(delay)

    return (correct, total, result, conf_tmp, delay_tmp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", type=int, nargs="+", help="the destined lost frames")
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--deadline", type=float, default=None, help="dealy deadline")
    parser.add_argument("--num_iter", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset_path", type=str, default="./dataset/test")
    parser.add_argument("--sample_path", type=str, default="./dataset/tiny.pkl")
    parser.add_argument("--result_path", type=str, default="./result")
    args = parser.parse_args()

    samples = pd.read_pickle(args.sample_path)

    net = i3d.myModel(args.device, "I3D", "eval")

    print("~~~~~~~~~~~simulation is set up now~~~~~~~~~~~~~")

    x = []  # loss_rate
    y_conf = []
    y_acc = []
    y_delay = []

    for threshold in loss:
        print(f"|----------- loss rate : {threshold} -----------|")
        correct = [0] * len(baselines)
        correct[baselines["FEC"]] = [0] * len(expected_loss)
        total = [0] * len(baselines)
        total[baselines["FEC"]] = [0] * len(expected_loss)
        # conf_set = [[] for i in range(len(baselines))]
        delay_set = [[] for i in range(len(baselines))]
        delay_set[baselines["FEC"]] = [[] for i in range(len(expected_loss))]

        for _, row in tqdm(samples.iterrows(), total=samples.shape[0]):
            path = os.path.join(args.dataset_path, row["video_path"])
            saliency = row["saliency"]
            label = row["category"]
            groundtruth_label = -1  # result of TCP
            assert (
                baselines["TCP"] == 0
            ), "1st baseline, as acccuracy benchmark, must be TCP"

            for mode, index in baselines.items():
                correct_tmp, total_tmp, result, conf_tmp, delay_tmp = simulate(
                    net, path, saliency, groundtruth_label, threshold, mode, args
                )
                if index == 0:  # Groundtruth
                    groundtruth_label = result
                    correct_tmp = total_tmp
                if index == baselines["FEC"]:
                    correct[index] = [
                        correct_tmp[i] + correct[index][i]
                        for i in range(len(correct_tmp))
                    ]
                    total[index] = [
                        total_tmp[i] + total[index][i] for i in range(len(total_tmp))
                    ]
                    for i in range(len(expected_loss)):
                        delay_set[index][i] += delay_tmp[i]
                else:
                    correct[index] += correct_tmp
                    total[index] += total_tmp
                    # conf_set[index] += conf_tmp
                    delay_set[index] += delay_tmp

        print(mode)
        print(correct)
        print(total)
        print(delay_set)

        # print(f"Summary of all samples with loss rate {threshold}")
        # print(f"Acc: {correct} / {total} = {correct/total}")
        # print(f"Conf: {sum(conf_set)/len(conf_set)}")

        if threshold > 0.0:
            x.append(threshold)
            for mode, index in baselines.items():
                # print(mode, correct[index], total[index], delay_set[index])
                if index == baselines["FEC"]:
                    acc_FEC = []
                    delay_FEC = []
                    for i in range(len(expected_loss)):
                        acc_FEC.append(100 * correct[index][i] / total[index][i])
                        delay_FEC.append(
                            sum(delay_set[index][i]) / len(delay_set[index][i])
                        )
                        # delay_FEC.append(np.percentile(delay_set[index][i], 95))
                    y_acc.append(acc_FEC)
                    y_delay.append(delay_FEC)
                else:
                    # y_conf.append(sum(conf_set[index])/len(conf_set[index]))
                    y_acc.append(100 * correct[index] / total[index])
                    y_delay.append(sum(delay_set[index]) / len(delay_set[index]))
                    # y_delay.append(np.percentile(delay_set[index], 95))
    print("-----------stat of results--------------")
    print(x)
    print(y_acc)
    print(y_delay)

    # plt.style.use('ggplot')
    # lines = [
    #     [
    #         (y_delay[baselines['UDP']], y_acc[baselines['UDP']]),
    #         (y_delay[baselines['TCP']], y_acc[baselines['TCP']])
    #     ]
    # ]
    # c = np.array([(1, 0.42, 0.50, 1)])
    # lc = mc.LineCollection(lines, colors=c, linewidths=2, linestyle='dotted')
    # fig, ax = plt.subplots()
    # ax.add_collection(lc)
    # ax.autoscale()
    plt.rcParams["font.size"] = 25
    fig, ax = plt.subplots(figsize=(10, 5))
    min_acc = 101.0
    for mode, index in baselines.items():
        if mode == "FEC":
            continue
        else:
            delta = 0, 0.2
            if mode == "TCP":
                delta = -0.2, -1.2
            if "Se" in mode:
                delta = 0, -1
            ax.scatter(y_delay[index], y_acc[index], s=50)
            ax.annotate(
                mode, (y_delay[index] + delta[0], y_acc[index] + delta[1]), fontsize=25
            )
            min_acc = y_acc[index] if y_acc[index] < min_acc else min_acc
    ## For FEC
    delays = y_delay[baselines["FEC"]]
    accs = y_acc[baselines["FEC"]]
    ax.plot(
        [delays[i] for i in range(0, len(expected_loss), 2)],
        [accs[i] for i in range(0, len(expected_loss), 2)],
        label="FEC",
    )
    for i in range(len(expected_loss)):
        if i % 2 != 0:
            continue
        ax.scatter(delays[i], accs[i], s=50, c="black")
        ax.annotate(
            f"{int(expected_loss[i]*100)}%", (delays[i], accs[i] + 0.2), fontsize=25
        )
        if i == 2:
            ax.annotate("FEC", (delays[i] + 0.1, accs[i] - 1), fontsize=25)
    # ## plotting
    ax.set_ylabel("Avg. Accuracy (%)")
    ax.set_ylim([87, 100])
    ax.set_xlabel("Avg. Delay (s)")
    # plt.xlim([y_delay[baselines['UDP']]*0.5, y_delay[baselines['TCP']]*1.2])
    ax.set_xlim([0.0, 3.2])
    # ax.legend()
    ax.grid(True)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([100, 98, 96, 94, 92, 90, 88])
    savefig(
        plt,
        os.path.join(args.result_path, f"tradeoff_{loss[0]*100}%"),
        ftype="pdf",
    )

    # min_acc = 101.0
    # for mode, index in baselines.items():
    #     if mode=='FEC':
    #         continue
    #     else:
    #         plt.scatter(y_delay[index], y_acc[index], label=mode)
    #         min_acc = y_acc[index] if y_acc[index] < min_acc else min_acc
    # ## For FEC
    # delays = y_delay[baselines['FEC']]
    # accs = y_acc[baselines['FEC']]
    # plt.plot(delays, accs, label="FEC")
    # for i in range(len(expected_loss)):
    #     plt.scatter(delays[i], accs[i], s=6)
    #     plt.annotate(f"{int(expected_loss[i]*100)}%", (delays[i], accs[i] + 0.2), fontsize=15)
    # ## plotting
    # plt.ylabel("Avg. Accuracy (%)")
    # plt.ylim([int(min_acc)-4, 100])
    # plt.xlabel("Avg. Delay (s)")
    # # plt.xlim([y_delay[baselines['UDP']]*0.5, y_delay[baselines['TCP']]*1.2])
    # plt.xlim([0.0, y_delay[baselines['TCP']]*1.2])
    # plt.legend()
    # plt.grid(True)
    # if not os.path.exists(args.result_path):
    #     os.makedirs(args.result_path)
    # savefig(plt,
    #     os.path.join(args.result_path, f'tradeoff_{loss[0]*100}%'),
    #     ftype='png',
    #     fontsize=50,
    #     dpi=250
    # )
    plt.close()
    print(f"~~~~~~~~~~Results saved to {args.result_path}~~~~~~~~~~~~")
