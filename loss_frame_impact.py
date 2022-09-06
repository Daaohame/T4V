samples = [
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
    # 'petting animal (not cat)/0CE_vXhy5NU_000081_000091.mp4',
    # 'riding camel/-bGVnGCy2yY_000004_000014.mp4',
    'playing trumpet/-KCuZn3n2ZU_000056_000066.mp4'
]

import os
for i in samples:
    if os.path.exists("verif.log"):
        os.remove("verif.log")
    for l in range(0, 64, 2):
        cmd = f"python3 action.py --m destined --l {l} --n 3 --s '{i}' >> verif.log"
        print(cmd)
        os.system(cmd)