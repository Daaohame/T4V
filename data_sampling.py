from dnn import i3d
from utils.misc import get_label
from utils import video_lib

import numpy as np
import os
import random
import decord
import argparse
import urllib
from tqdm import tqdm
import pandas as pd


NUM_CAT = 400   # 400 is for kinetics-400
MIN_VIDEO_LEN = 65


def count_dir(dir):
    totalFiles = 0
    totalSubDir = 0
    for base, dirs, files in os.walk(dir):
        for directories in dirs:
            totalSubDir += 1
        for Files in files:
            totalFiles += 1
    return totalFiles, totalSubDir


def generate_sample_entry(net, dataset_path, video_path, cat=None):
    video_clip = video_lib.read_video_clip(os.path.join(dataset_path, video_path))
    video_clip = video_lib.video_transform_gluoncv(video_clip, net.device)
    sal, pred = net.get_saliency(video_clip)
    if cat is None:
        cat = get_label(os.path.basename(os.path.dirname(video_path)))
    return [video_path, sal.tolist(), cat, pred.argmax().item()]


def sampling(net, args):
    dataset_path = args.dataset_path
    num_category = args.num_category
    num_sample_per_cat = args.num_sample_per_cat

    if any(h in dataset_path[:7] for h in ["http", "ftp"]):
        file = urllib.request.urlopen(dataset_path)
        full_list = {}
        for line in file:
            decoded_line = line.decode("utf-8").strip('\n')
            parsed_path = os.path.split(decoded_line)
            cat = os.path.basename(parsed_path[0])
            if cat not in full_list:
                full_list[cat] = [parsed_path[1]]
            else:
                full_list[cat].append(parsed_path[1])
        dirc = list(full_list.keys())
    else:
        dirc = os.listdir(dataset_path)
    dirc.sort()
    dirc = list(enumerate(dirc))  # add indexes for category directories
    if num_category < NUM_CAT:
        dirc = random.sample(dirc, num_category)
    sample_list = []

    for no, it in tqdm(dirc):   # a (not-too-short) video from each category

        cat_path = os.path.join(dataset_path, f'{it}')
        if any(h in dataset_path[:7] for h in ["http", "ftp"]):
            video_list = full_list[it]
        else:
            video_list = os.listdir(cat_path)
        selected_videos = random.sample(
            video_list, 
            min(num_sample_per_cat, len(video_list))
        )
        for v in selected_videos:
            video_list.remove(v)
        for v in selected_videos:
            vr = decord.VideoReader(os.path.join(cat_path, v))
            short_flag = False
            while len(vr) < MIN_VIDEO_LEN:
                short_flag = True
                if v in video_list:
                    video_list.remove(v)
                v = random.sample(video_list, 1)[0]
                vr = decord.VideoReader(os.path.join(cat_path, v))
            if short_flag:
                selected_videos.append(v)
        
        print("category:", it, selected_videos)
        for v in selected_videos:
            print("        video:", v)
            sample_list.append(generate_sample_entry(net, dataset_path, f"{it}/{v}", no))
    
    return sample_list


def store_data(sample_list, out_path='./dataset/testset.pkl'):
    df = pd.DataFrame(
        sample_list,
        columns=['video_path','saliency','category','prediction']
    )
    # df.to_csv('./dataset/testset.csv', encoding='utf-8')
    df.to_pickle(out_path, protocol=4)


tiny_samples = [
    'playing piano/1lztLegtDLo_000010_000020.mp4',
    'abseiling/3E7Jib8Yq5M_000078_000088.mp4',
    'watering plants/-nWf-00Vhc4_000011_000021.mp4',
    'playing ice hockey/0E8h0BPYOmE_000238_000248.mp4',
    'getting a haircut/3bdQR_2juZQ_000000_000010.mp4',
    'sled dog racing/1habnDjBc0g_000033_000043.mp4',
    'playing harmonica/2klAbqiHaJc_000011_000021.mp4',
    'washing hair/2yf6_k4mYRQ_000007_000017.mp4',
    'belly dancing/0QHFsMT93_k_000178_000188.mp4',
    'cooking chicken/0jVKCRryoHk_000348_000358.mp4',
    'clapping/1b1ExQtYn8A_000075_000085.mp4',
    'petting animal (not cat)/0CE_vXhy5NU_000081_000091.mp4'
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', type=str, default='./dataset/test')
    parser.add_argument('-op', '--output_path', type=str, default='./result')
    parser.add_argument('-sn', '--sample_name', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('-nc', '--num_category', type=int, default=400)
    parser.add_argument('-ns', '--num_sample_per_cat', type=int, default=1)
    args = parser.parse_args()

    net = i3d.myModel(args.device, 'I3D', 'eval')
    
    # sample_list = [generate_sample_entry(net, args.dataset_path, i) for i in tqdm(tiny_samples)]
    sample_list = sampling(net, args)
    # print(sample_list)

    store_data(sample_list, os.path.join(args.output_path, f'{args.sample_name}.pkl'))