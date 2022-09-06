import torch
import time


def compute_diff(video_data):
    maxlen = len(video_data)
    # compute the difference
    tt0 = time.time()
    diff = []
    for i in range(maxlen):
        # diff_i = []
        for j in range(maxlen):
            # diff_i.append(torch.sum(torch.abs(video_data[i] - video_data[j])))
            torch.sum(torch.abs(video_data[i] - video_data[j]))
        # diff.append(diff_i)
    tt1 = time.time()
    print(f"!!!!!!! {tt1 - tt0}")


if __name__ == "__main__":
    video = torch.randint(
        low=0,
        high=3,
        size=[32, 3, 256, 256],
    )
    compute_diff(video)
