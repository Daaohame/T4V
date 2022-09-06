import os
import argparse
import decord

import utils
from core import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='simple', choices=["simple", "verbose"])
        # mode options
        #     simple: only to output saliency
        #     verbose: to output saliency, saliency figure, and heatmap
    parser.add_argument('--interval', type=int, default=2)
    parser.add_argument('--num_iter', type=int, default=10)
    parser.add_argument('--sample_path', type=str, nargs='+', default='')
    args = parser.parse_args()
    interval = args.interval

    sample_dir = "dataset/"
    samples = args.sample_path if args.sample_path else [
        # 'watering plants/-nWf-00Vhc4_000011_000021.mp4',
        # 'playing ice hockey/0E8h0BPYOmE_000238_000248.mp4',
        # 'abseiling/3E7Jib8Yq5M_000078_000088.mp4',
        # 'getting a haircut/3bdQR_2juZQ_000000_000010.mp4',
        # 'sled dog racing/1habnDjBc0g_000033_000043.mp4',
        # 'playing harmonica/2klAbqiHaJc_000011_000021.mp4',
        # 'washing hair/2yf6_k4mYRQ_000007_000017.mp4',
        # 'belly dancing/0QHFsMT93_k_000178_000188.mp4',
        # 'cooking chicken/0jVKCRryoHk_000348_000358.mp4',
        # 'clapping/1b1ExQtYn8A_000075_000085.mp4',
        'petting animal (not cat)/0CE_vXhy5NU_000081_000091.mp4'
    ]
    samples = [(utils.get_label(os.path.dirname(i)), sample_dir+"/"+i) for i in samples]

    for no, (label, path) in enumerate(samples):
        vr = decord.VideoReader(path)
        frame_id_list = list(range(0, min(len(vr),64), interval))
        video_data = vr.get_batch(frame_id_list).asnumpy()
        sal = get_saliency(video_data, args.mode)
        indispensable = [no*2 for (no,val) in enumerate(sal) if val >= 0.1]
        print(indispensable)