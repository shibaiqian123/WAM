# WB-WAM 训练

[English](./README.md)

先按[仓库总览](../README_zh.md)安装训练与 HumanoidArena 评测共用的 `wbwam` conda 环境。以下其余命令均从 `training/` 目录运行。

## 环境

```bash
cp -n .env.example .env
```

安装后用一段实际训练视频检查解码（替换视频路径）：

```bash
VIDEO=/path/to/episode.mp4 python -c 'import os; from torchcodec.decoders import VideoDecoder; print(VideoDecoder(os.environ["VIDEO"])[0].shape)'
```

按机器修改 `.env`，然后执行 `source scripts/load_env.sh`。训练还需要 Wan2.2 的模型资产和 ActionDiT 初始化权重；按 `.env.example` 配置路径。若尚无 ActionDiT 初始化文件，可运行：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/wbwam_optional_idm.yaml \
  --output "$WBWAM_ACTION_DIT_PATH" --device cuda --dtype bfloat16
```

## 模型下载

从 [WB-WAM 预训练与中间训练模型仓库](https://huggingface.co/WB-WAM/WB-WAM-Pretrain-Midtrain)下载全量预训练和最新中间训练权重，以及配套配置、统计量：

下载脚本使用当前的 `HF_ENDPOINT`；如需镜像，可先设置 `export HF_ENDPOINT=https://hf-mirror.com`。

```bash
python ../scripts/download_models.py pretrain --output ./checkpoints/wbwam
python ../scripts/download_models.py midtrain --output ./checkpoints/wbwam
```

如需单独下载 [HumanoidArena 权重](https://huggingface.co/WB-WAM/WB-WAM-HumanoidArena)，可运行 `python ../scripts/download_models.py arena --task hammer --output ./checkpoints/wbwam`（将 `hammer` 换成目标任务）。脚本保留训练时的目录结构，并检查权重、配置和统计量是否齐全。

在 `.env` 中设置：

```dotenv
WBWAM_PRETRAIN_CHECKPOINT=./checkpoints/wbwam/pretrain/step_037617.pt
WBWAM_MIDTRAIN_CHECKPOINT=./checkpoints/wbwam/midtrain/step_020560.pt
```

## 数据

### Pico 与自收数据

从 [Pico](https://huggingface.co/datasets/WB-WAM/Pico) 和[自收数据](https://huggingface.co/datasets/WB-WAM/Self-Collected)直接下载解压后的 LeRobot recording：

```bash
hf download WB-WAM/Pico --repo-type dataset --local-dir ./data/pico_archive
hf download WB-WAM/Self-Collected --repo-type dataset --local-dir ./data/real_archive
```

保持 `.env` 中的 `WB_WAM_DATA_ROOT=./data`；Pico 用于中间训练，`real_archive` 用于八个自收任务的后训练。无需解压。

### HumanoidArena

只下载[官方 SONIC v3.1 数据](https://huggingface.co/datasets/WilliamWang16/HumanoidArena_dataset_v3_1)中的 `sonic_8_refpose_v3_1`：

```bash
huggingface-cli download WilliamWang16/HumanoidArena_dataset_v3_1 \
  --repo-type dataset --revision a079beddd6b1521f762c991be8f36993f17ebeca \
  --include 'HumanoidArena_merged_datasets_v3_1/sonic_8_refpose_v3_1/**' \
  --local-dir ./data/humanoid_arena
```

将 `.env` 的 `WBWAM_ARENA_NATIVE50_ROOT` 设为 `./data/humanoid_arena/HumanoidArena_merged_datasets_v3_1/sonic_8_refpose_v3_1`。数据保持官方 50 Hz 和原始 reference action，不做离线重采样或 action 后移。

## 训练

先执行 `source scripts/load_env.sh`。midtrain 复用已下载的 pretrain norm stats；每个后训练任务独立计算 stats。各任务还需预计算语言 embedding。

### Pico 中间训练与自收后训练

Pico midtrain 从预训练权重启动：

```bash
python scripts/precompute_text_embeds.py wb_task=midtrain
bash scripts/train_zero2.sh 8 wb_task=midtrain \
  "data.pretrained_norm_stats=./checkpoints/wbwam/pretrain/dataset_stats.json" \
  "resume=$WBWAM_PRETRAIN_CHECKPOINT"
```

自收后训练以 `pillow` 为例，从 pretrain 权重启动：

```bash
TASK=pillow
python scripts/compute_wb_norm_stats.py --output-dir "$WBWAM_STATS_ROOT/$TASK" "wb_task=posttrain/$TASK"
python scripts/precompute_text_embeds.py "wb_task=posttrain/$TASK"
bash scripts/train_zero2.sh 8 "wb_task=posttrain/$TASK" \
  "data.pretrained_norm_stats=$WBWAM_STATS_ROOT/$TASK/dataset_stats.json" \
  "resume=$WBWAM_PRETRAIN_CHECKPOINT"
```

其余任务：`make_bed`、`close_curtain`、`wipe_table`、`checkout`、`push_cart`、`cloth_basket`、`fruit_basket`。`fruit_basket` 联合训练三条水果指令。

### HumanoidArena 后训练

七个任务均从全量预训练权重启动；以 `hammer` 为例：

```bash
TASK=hammer
python scripts/compute_wb_norm_stats.py \
  --output-dir "$WBWAM_STATS_ROOT/humanoid_arena_native50/$TASK" \
  "wb_task=posttrain/humanoid_arena_native50/$TASK"
python scripts/precompute_text_embeds.py "wb_task=posttrain/humanoid_arena_native50/$TASK"
bash scripts/train_zero2.sh 8 "wb_task=posttrain/humanoid_arena_native50/$TASK" \
  "data.pretrained_norm_stats=$WBWAM_STATS_ROOT/humanoid_arena_native50/$TASK/dataset_stats.json" \
  "resume=$WBWAM_PRETRAIN_CHECKPOINT"
```

其余任务：`kick_football`、`box_shelf`、`punch_markers`、`open_door`、`sit_sofa`、`obstacle_navigation`。

## SONIC latent 接口（仅模型验证）

使用 `model=sonic_latent66`，状态原生 31 维，动作原生 66 维。状态依次为 `observation.sonic_state` 的 29 个身体关节、左 Dex1、右 Dex1；动作依次为 64 维 SONIC motion token、左右 Dex1 连续目标。输入输出使用训练归一化空间，不增加 Root、不二值化夹爪、不经过 SONIC Encoder。保留视频／动作联合建模，官方物理动作配置保持原样。

`wb_task=sonic_latent66` 是仅模型配置：选择新模型、关闭 FK、从 `WBWAM_MIDTRAIN_CHECKPOINT` 初始化，尚未提供数据集。数据转换、统计量和语言缓存需另行准备，不能直接接入未适配的物理动作数据集。

`init_from_checkpoint` 与 `resume` 互斥。从物理动作 midtrain 初始化时严格保留完整兼容主干，只以种子 42 重建动作输入／输出和状态投影（包含偏置），不恢复旧优化器和步数。新的 SONIC checkpoint 保存接口契约并严格恢复；普通 resume 不接受物理动作权重。完整初始化前禁止训练／推理，此路径不需要单独生成 ActionDiT 初始化文件。

SONIC 损失排除时间与维度 padding，有效维度等权；记录 `sonic_latent_flow_mse` 和 `dex1_flow_mse`，它们是未加调度权重的归一化 flow 目标误差，不是解码后的物理误差。全 padding 样本贡献零。推理返回 `[H,66]`；反归一化和 SONIC／Dex1 控制接入属于后续工作。

仓库根目录运行 `PYTHONPATH=training/src python -m unittest discover -s training/tests -v`。已明确分配 A100 后，在训练环境的 `training/` 下加载 `source scripts/load_env.sh`，设置该卡的 `CUDA_VISIBLE_DEVICES`，运行 `python scripts/smoke_sonic_latent.py --checkpoint "$WBWAM_MIDTRAIN_CHECKPOINT" --report /path/to/smoke.json`。该 BF16 合成检查执行完整权重迁移、一次前向／反向（无优化器更新）和两步首帧条件采样，不代表任务成功或真机就绪。

## TensorBoard

训练命令增加 `tensorboard.enabled=true`。仅主进程向 `<output_dir>/tensorboard` 写入训练和验证标量，不依赖 W&B。在训练服务器运行以下命令，通过 SSH 转发 6006 端口后在本地查看。

```bash
tensorboard --logdir "$WBWAM_RUNS_ROOT" --host 127.0.0.1 --port 6006
```

SONIC 任务预设设置 `optimizer_foreach=false`。在已测试的 PyTorch 2.7.1 + CUDA 12.8 环境中，默认 foreach AdamW 在含 3,008,932,864 个 FP32 元素的 ZeRO 分片上产生负二阶动量与 NaN 参数；单张量 AdamW 通过同规模独立测试。训练损失出现非有限值时，所有进程会在反向传播前停止。NaN 运行应从干净权重重新开始。

SONIC 同时把 ZeRO 分片前的每个优化器参数组限制为 5 亿元素，以控制单张量 AdamW 临时显存。参数顺序、各参数的学习率与权重衰减保持不变。优化器状态分组发生变化，应启动新运行，不要恢复旧分组的完整训练状态。
