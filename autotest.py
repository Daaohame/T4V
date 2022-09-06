# samples = ['petting animal (not cat)/0CE_vXhy5NU_000081_000091.mp4', 'drinking/1iz3UTeecZk_000000_000010.mp4', 'jumping into pool/3uSViSsQTog_000005_000015.mp4', 'playing trumpet/-KCuZn3n2ZU_000056_000066.mp4', 'golf putting/2damYoBWFyo_000036_000046.mp4', 'slacklining/-y-gbZRSlfk_000241_000251.mp4', 'front raises/1Y4V9NeqzOU_000005_000015.mp4', 'zumba/0_tMEDs_rzM_000116_000126.mp4', 'singing/0_tiV36Yris_000186_000196.mp4', 'headbanging/1dfXFbGAemo_000000_000010.mp4', 'drop kicking/1hlj7QlFeH8_000000_000010.mp4', 'skiing crosscountry/156xthDE5zY_000041_000051.mp4', 'eating chips/2B-ZEkCfTmg_000022_000032.mp4', 'swimming breast stroke/1AD6ltFmiaI_000004_000014.mp4', 'exercising arm/0mp3gm2Jpqs_000009_000019.mp4', 'playing organ/0ygbcm02tLs_000002_000012.mp4', 'shoveling snow/0pPveLvxNOM_000038_000048.mp4', 'dying hair/38KdiUtaHQc_000404_000414.mp4', 'feeding goats/3GYbX6JHWqE_000080_000090.mp4', 'sanding floor/33zEQAH-DtM_000160_000170.mp4', 'drinking beer/--6bJUbfpnQ_000017_000027.mp4', 'trimming trees/0GTjQ5ps8uk_000004_000014.mp4', 'javelin throw/-9iGs_3h12Q_000004_000014.mp4', 'snowmobiling/487RhEQtlHk_000012_000022.mp4', 'playing chess/3i2H9U4NZQc_000142_000152.mp4', 'tapping guitar/3N8XvW-g03Y_000153_000163.mp4', 'punching bag/1QVEw8wIPbw_000003_000013.mp4', 'walking the dog/0vhwiO66Tig_000017_000027.mp4', 'smoking/IQaoRUQif14.mp4', 'playing monopoly/2h89Ynbyq1w_000031_000041.mp4', 'clapping/-LeQsXtoNg0_000004_000014.mp4', "massaging person's head/-GVV9MG9ZSM_000068_000078.mp4", 'washing hair/-1nCh3Vk4Qk_000441_000451.mp4', 'changing wheel/3ZxNQ4gbdkc_000001_000011.mp4', 'salsa dancing/3ZGGnMUdQ9o_000233_000243.mp4', 'hammer throw/3xxCPie-73s_000001_000011.mp4', 'auctioning/0WzeFbAAJrk_000107_000117.mp4', 'playing kickball/-IdatujDsqA_000049_000059.mp4', 'riding unicycle/0pe8SgWUp74_000049_000059.mp4', 'cutting watermelon/1MRCSNIlKHE_000009_000019.mp4', 'water sliding/2svszvcZNwc_000007_000017.mp4', 'shining shoes/-OwsGvItrzg_000366_000376.mp4', 'playing piano/-7aeB7vFtB4_000037_000047.mp4', 'writing/0ABg9R9boM4_000086_000096.mp4', 'strumming guitar/3L6nHgxMo-w_000269_000279.mp4', 'braiding hair/1zfek_qVjiA_000003_000013.mp4', 'tapping pen/-TQ_pI1z4Bo_000024_000034.mp4', 'bouncing on trampoline/1o-7jhVIn-U_000001_000011.mp4', 'riding camel/-bGVnGCy2yY_000004_000014.mp4', 'catching fish/3VvC9Sep1kM_000104_000114.mp4']

samples = [
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
]

import os
import subprocess
for i in samples:
    cmd = f"python3 action.py --n 50 --s \"{i}\" --o result/sample2/"
    print(cmd)
    subprocess.run(cmd, shell=True)
    # for l in range(0, 64, 2):
    #     cmd = f"python3 action.py --m destined --l {l} --n 3 --s '{i}' >> verif.log"
    #     print(cmd)
    #     subprocess.run(cmd)