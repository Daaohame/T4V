## 常用设置
设定 GPU：
```
nvidia-smi (gpustat)
export CUDA_VISIBLE_DEVICES=1
```
用Tensorboard查看结果：
```
python3 train.py -bs 1 -ne 50 -dp ./dataset/tiny.pkl
python3 train.py -bs 1 -ne 40 -dp ./dataset/min3.pkl --model_path ./checkpoint/ -lr 1e-6 --device_num 1
tensorboard --logdir_spec=Train:/datamirror/haodongw/T4V/result/train,Test:/datamirror/haodongw/T4V/result/test
```

## Major Stages
### 1. Choosing the models
- HAR 的 pre-trained model 我只找到了这个 https://github.com/CMU-Perceptual-Computing-Lab/openpose
还有别的吗？
- 换成 `pytorch action recognition` 来搜索
    - Gluon CV Toolkit（用处似乎更多一点）
- Object Detection 的，model zoo 上有太多了，怎样选呢
    - mobilenetSSD?
    - fasterRCNN?
// 目前还只是用 CPU 测试吧

### 2. How to drop frames
- current solution: sampling at equal intervals
    - pro: the continuity of the video is broken, just like losing frames during transmission
    - con (from a larger perspective): something special took place during a few frames, and they are all lost?
    - con: unfair for videos of different sizes
- sampling at equal intervals for a fixed fraction of total frames
    - i.e., longer the video, larger the interal
- each frame lost at a fixed probability
    - is it good for pre-test?
- each frame lost at a random probability
    - is it good for pre-test?
interval = 5 -> Acc: 0.5165668662674651 1294 / 2505
interval = 4 -> Acc: 0.536127744510978 1343 / 2505
interval = 3 -> Acc: 0.5473053892215569 1371 / 2505
interval = 2 -> Acc: 0.5536926147704591 1387 / 2505
interval = 1 -> Acc: 0.5365269461077844 1344 / 2505

根据丢帧情况，大概感受下丢包率的影响，特别是在网络的正常状况下的范围内
在此基础上，需要进一步增大


#### 2.1
- 用 softmax 来 normalization 分数
- 观察初始最高分所对应的类的分数（此即confidence），以排除模型本身正确与否带来的影响
- ensure that you can *fully* interpter the figure
- pick loss rates unevenly for more observations
- eyeball the figure to find the "loss pattern" of those who can still remain a high score
Also, I can add another term to punish the case that the score of the correct label drops from the highest to lower ranking. Anyway, the accuracy of the analyst depends on the highest score.  

#### 2.2
- run on all the 10 videos
- focus on those whose performance is significantly affected by frame loss
- observe the log to find if some frames are critical to the performace (instead of just any random frame)
    - Concl: *some frames are critical*

### 3. How to figure out the importance of each frame
- 总思路：heuristics may not work -> DNN-aware way
- 效果目标：loss tolerant -> less retrasmission -> less delay
    - Therefore, 不能server-driven，要用本地小model
- 初步验证：saliency-based
    - saliency 计算
        - 分不可以直接加总，不然正数负数都抵消了
        - 第二范数和直接的结果：不是很match
     - 首先：验证saliency基本的合理性（一点观察）
    - 多找几个样本来验证
        - criteria：avg conf does not drop but increase & conf has large var with higher loss rate, provided that classification with no loss is correct
    - heat map of saliency

Keep it mind: debugging and tuning require tricks and wisdom, do it yourself and ask to check the reasonalibility of the approach

#### 3.2 丢帧方法的改进和对saliency的质疑
- saliency 的基本合理性：对那些不怎么受影响的video，saliency很小；对受影响的video，saliency较大，其中较大的帧丢失后**不一定**有明显影响；丢失后有明显影响的帧，其saliency**往往**很大
- 综上，目前方案：将saliency较大的帧标记为“不可丢”

#### 3.2 Summary Dec/03/2021
- we have found a reasonable way, now it's time to train the model
- [AW] the mysterious period (4 frames) of saliency
- [AW] a new way to fill the lost frame (e.g. previous & following)
- [Figure] scatter: to show the correlation between loss and saliency
- [Ongoing] current simulation of the tradeoff between delay and accuracy
    - the correctness of simulation of TCP delay
    - a better way to show the plot

### 4. The cheap model to determine packet utility
- Architecture of NN
    - combine (`cat`) intermediate results of different dimension sizes (after upsamping to the same size)
    - properly set our newly-defined layers so that they can be trained, i.e., their parameters can be updated (use `nn.Parameters` to check)
- How to tune NN
    - number of parameters: underfit -> more, overfit -> less (like number of channels)
    - To apply `Dropout` and `ReLU`
        - By now you can only use `batchsize=1`, otherwise dropout does not work as video sizes are different (som smaller than 224)
- Serious bug
    - After __one__ passing of the input (`requires_grad=True`) through Conv3d, `requires_grad` turned to `False`.
    - We found before entering the NN, `grad_fn` of `clip_input` already went `None`.
    - Final: stupid animator wheel caused memory leakage (`plt` not closed, too much memory used)
    - Surprise! This is the true reason: each time we read data using `DataLoader` the saliency is calculated, which requires gradient. But the model has been set as `eval` during the process of training.  


### 5. Checkpoint: trial of workshop & Practicum midterm
- expriment: results with more loss rates to show the rationality in more cases
- Pedro: remote surgery -> 50 ms 
    - really? ideally < 200 ms, at most no more than 400-500ms 
- experiment: more baselines
- finish training the model [as early!]

### 6. Acc-delay tradeoff results
- Figure: error ellipse, utility curves? [AW]
- Benchmark (of accuracy): TCP
- Baselines: FEC, selective re-tx
    - FEC
        - how to emphasize the drawbacks of FEC? 
        - we need adaptive video quality [AW]
        - small bw, small delay ddl
- Add the delay deadline to our simulation?　Not now, as deadline may not be a good formulation.

### 7. Go back to the cheap model
- the test accuracy is wrong and unfinished ...
- prediction = [0_score, 1_score]
- groudtruth = 0/1
- Cross entropy loss: class imbalance
	- The output is very likely to be composed of all the dominant class in the label
    - solution: Focal loss
- Update: co-debugging with Kuntai -> problems reside in the net architecture
    - kernel size: odd, and <=3 (even -> centroid shift)
    - better use padding, while stride=1 (to reduce info loss), unless memory limits

### 8. First submission!
- skip the cheap model (predictor)
- logic augmentation: retransmission desicion based on both packet impact on inference and received packets -- *incremental influence*
- To-do
    - citation for the "putting on a coat" example
    - still room for improvement of writing, but marginal revenue was shrinking
    - cascading effect on the ones following the lost frame
    - how to encode: influence info in packet or metadata? how to split a frame into packet?
    - sim results: deadline; error ellipse


### Misc
- Backup plans for action recognition
    - [MMActions2](https://github.com/open-mmlab/mmaction2)



## Autoencoder
- AE连finalDNN后求导可能有问题
- 读Qizheng（重要region由server决定但必须传一个low quality的frame作为判断依据），我们能不能做出来 online & generalizable 的？（根据之前的frame判断下一个frame）

## KT: tune inference based on knobs' influence on accuracy
- NN 优化的目标是 max acc & min resource，但缺乏直接的handle，导致优化困难（例如 VA 中对视频的调节需要经过复杂的过程才能反映在 accuracy 上）
- 原因：accuracy不可导
- solution: 
    - $$ \frac{\partial{output}}{\partial{knob}} = \frac{\partial{output}}{\partial{input}} \cdot \frac{\partial{input}}{\partial{knob}} $$ 
    - 其中 $\frac{\partial{input}}{\partial{knob}}$ 可用数值导数计算
- drawback
    - 尚未将根据 $\frac{\partial{output}}{\partial{knob}}$ 调 knob 整合为统一的 API
    - accuracy proxy：用有限资源下的 inference results 来估计 accuracy （而不需要算无限资源下的结果作为 benchmark，不然不适用于 VA）

## YH: video encoding not dependent on loss prediction
- 传统方法：如 FEC，预估 redundancy rate 以保证丢包后的视频质量
- solution: 用 autoencoder 训练，且训练中加入一定的 packet loss，以此提升 inference 对任意 data loss 的 resilience, while not sacrificing the encoding efficiency
- 目前结果：用 30% loss 训练出来的 autoencoder，在任意 loss rate 下，decode 所得视频质量与传统方法打平 （但 10% 和 60% 不行）