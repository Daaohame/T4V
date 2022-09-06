import os
import time
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

from dnn import i3d
from utils.misc import get_label
from utils.myplot import savefig
from utils import video_lib
import decord

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='probability', help='loss mode')
    # mode options
    #     probability: each frame lost with prob. of {loss rate}
    #     destined: lose certain frames
parser.add_argument('--loss', type=int, nargs='+', help='the destined lost frames')
parser.add_argument('--interval', type=int, default=2)
parser.add_argument('--num_iter', type=int, default=10)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--sample_path', type=str, nargs='+', default='')
parser.add_argument('--output_path', type=str, default='result/')
args = parser.parse_args()

if os.path.exists("result/act_recg.log"):
    os.remove("result/act_recg.log")
logger = logging.getLogger("act_recg")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler("result/act_recg.log"))


x = []
y = []
z = []

mode = args.mode
interval = args.interval
device = args.device
sample_dir = "dataset/test"
samples = args.sample_path if args.sample_path else [
    'watering plants/-nWf-00Vhc4_000011_000021.mp4',
    'playing ice hockey/0E8h0BPYOmE_000238_000248.mp4',
    'abseiling/3E7Jib8Yq5M_000078_000088.mp4',
    'getting a haircut/3bdQR_2juZQ_000000_000010.mp4',
    'sled dog racing/1habnDjBc0g_000033_000043.mp4',
    'playing harmonica/2klAbqiHaJc_000011_000021.mp4',
    'washing hair/2yf6_k4mYRQ_000007_000017.mp4',
    'belly dancing/0QHFsMT93_k_000178_000188.mp4',
    'cooking chicken/0jVKCRryoHk_000348_000358.mp4',
    'clapping/1b1ExQtYn8A_000075_000085.mp4',
    'petting animal (not cat)/0CE_vXhy5NU_000081_000091.mp4'
]
samples = [(get_label(os.path.dirname(i)), sample_dir+"/"+i) for i in samples]
loss = []
if mode == 'probability':
    for i in range(0, 101, 10):
        threshold = i/1000
        loss.append(threshold)
    loss.append(0.15)
    loss.append(0.2)
else:
    loss.append(0.0)
    loss.append(0.1)
    loss.append(0.5)

conf_bench = [1.0 for i in samples]
label_bench = -1
# indispensable = random.sample(list(range(0, 64, interval)), 4)
indispensable = []

# model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
# model = model.eval()
# model = model.to(device)
net = i3d.myModel(device, 'I3D', 'eval')

for threshold in loss:
    logger.info(f"|------------Loss rate {threshold}------------|")
    conf_max = -10.0
    conf_min = 10.0
    total = 0
    correct = 0
    acc_set = []
    conf_set = []

    for no, (label, path) in enumerate(samples):
        logger.info(f"~~~~~sample {no}: {path}~~~~~")
        # url = 'https://github.com/bryanyzhu/tiny-ucf101/raw/master/abseiling_k400.mp4'
        # video_fname = download(url, path="testset")
        vr = decord.VideoReader(path)
        # frame_id_list = list(range(0, min(len(vr),64), interval))
        frame_id_list = list(range(0, 10, 1))

        n_iteration = args.num_iter if threshold > 0 else 1
        conf_tmp = []
        for j in tqdm(range(n_iteration)):
            frame_id_list_rx = []  # frames analyst receives after loss
            frame_id_list_lost = []
            if mode == 'destined' and threshold > 0.1:
                frame_id_list_lost = args.loss
                for i in range(len(frame_id_list)):
                    if frame_id_list[i] in frame_id_list_lost:
                        frame_id_list_rx.append(frame_id_list_rx[i-1] if i>0 else frame_id_list[0])
                    else:
                        frame_id_list_rx.append(frame_id_list[i])
            else:
                for i in range(len(frame_id_list)):
                    p = random.random()
                    if p <= threshold and frame_id_list[i] not in indispensable:
                        frame_id_list_rx.append(frame_id_list_rx[i-1] if i>0 else frame_id_list[0])
                        # print(f"loss {frame_id_list[i]}, filled with {frame_id_list_rx[(i-1) if i>0 else 0]}")
                        # print(frame_id_list_rx)
                    else:
                        frame_id_list_rx.append(frame_id_list[i])
                        # print(f"get {frame_id_list[i]}")
                frame_id_list_lost = list(set(frame_id_list) - set(frame_id_list_rx))
                frame_id_list_lost.sort()
            video_data = vr.get_batch(frame_id_list_rx).asnumpy()

            t1 = time.time()
            pred = net.predict(
                    video_lib.video_transform_gluoncv(video_data, device=args.device)
                )
            t2 = time.time()
            result = np.argmax(pred)
            if threshold == 0.0:
                label_bench = result
                conf_bench[no] = pred[label_bench]
            logger.info(f"    Label {label}, result {result}, label_bench {label_bench}, using time {t2-t1} -> {(t2-t1)/video_data.shape[0]} per frame")
            # conf = pred[label]/pred[result]
            conf = pred[label_bench]
            logger.info(f"    Confidence: {conf- conf_bench[no]} = {conf} - {conf_bench[no]}")
            conf = conf - conf_bench[no]
            conf_tmp.append(conf)
            if conf < conf_min:
                conf_min = conf
                conf_min_info = f"{label}, {path}, {frame_id_list_rx}"
            if conf > conf_max:
                conf_max = conf
                conf_max_info = f"{label}, {path}, {frame_id_list_rx}"
            tops = np.argpartition(pred, -5)[-5:]
            logger.info(f"    Top 5:{tops} with {pred[tops]}")
            total += 1
            if result == label:
                correct += 1
        conf_set += conf_tmp
    logger.info(f"Summary of all samples with loss rate {threshold}")
    logger.info(f"Acc: {correct} / {total} = {correct/total}")
    acc_set.append(correct/total)
    logger.info(f"Conf: {sum(conf_set)/len(conf_set)}")
    
    x.append(threshold)
    y.append(conf_set)
    z.append(sum(acc_set)/len(acc_set))

    logger.info(f"Conf_max: {conf_max} with {conf_max_info}")
    logger.info(f"Conf_min: {conf_min} with {conf_min_info}")

    if mode == 'destined' and threshold > 0.1:
        print(frame_id_list_lost[0], ',', conf_max)

x = [int(i*100) for i in x]
z = [i * 100 for i in z]
plt.rcParams["figure.figsize"] = (6.4, 5.4)
figure,axes = plt.subplots() #得到画板、轴
axes.set_title(f"{samples[0][1]}")
bplot = axes.boxplot(y, labels=x, vert=True, whis=(5,95), sym='', showmeans=True)
# colors = ['pink', 'lightblue', 'lightgreen']
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)
locs=axes.get_xticks()
# plt.style.use('ggplot')
plt.xlabel("Loss rate(%)")
plt.xticks(rotation=45)
plt.ylabel("Confidence change")
plt.ylim([-1.0, 0.5])
plt.grid(True)
plt.savefig(args.output_path + f'{os.path.basename(samples[0][1])}_conf.png', format='png', dpi=200)

plt.clf()
plt.title(f"{samples[0][1]}")
xi = list(range(len(x)))
plt.plot(xi, z, 'b-')
plt.xlabel("Loss rate(%)")
plt.xticks(rotation=45)
plt.ylabel("Accuracy(%)")
plt.xticks(xi, x)
plt.ylim([60, 102])
plt.grid(True)
plt.savefig(args.output_path + f'{os.path.basename(samples[0][1])}_acc.png', format='png', dpi=200)

# # error bar of sem
# plt.clf()
# mean = [np.mean(it) for it in y]
# error = [sem(it) for it in y]
# plt.style.use('ggplot')
# xi = list(range(len(x)))
# plt.plot(xi, mean, 'b-')
# plt.errorbar(xi, mean, yerr=error, linestyle='None', marker='^')
# plt.xlabel("loss rate")
# ## tune the x axis
# plt.xticks(xi, x, rotation=45)
# ##
# plt.ylabel("Mean of confidence")
# plt.grid(True)
# plt.savefig('result/sem.png', format='png', dpi=200)