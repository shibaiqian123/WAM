# 原生 31 -> 66 SONIC/Dex1 后训练数据

本预设使用 20 Hz 数据：31 维实测状态是按关节名选取的 29 个身体关节和
左右 Dex1 位置；66 维动作是 64 维 SONIC v1.1 latent 和左右 Dex1 连续
目标位置（电机输出轴弧度）。不添加 Root 状态。

转换程序只写入新目录。遵守 discarded_episode_indices，并整条排除任意
Dex1 反馈、命令、触发器有效性标记不通过的轨迹。非有限标签、错误时间网格
和视频帧数会使转换失败。以时间戳最近邻选择整行记录及对应图像；
半帧距离在 2 微秒容差内选较早帧，50 -> 20 Hz 采样误差上限为
10.01 毫秒。不平移动作，也不对 latent 插值。视频以无损 H.264 保存，
保留原始分辨率，由训练加载器缩放至 224x224。

仓库根目录执行：
```bash
python training/scripts/convert_sonic_dex1_dataset.py --source "$SOURCE" --audit-only
python training/scripts/convert_sonic_dex1_dataset.py --source "$SOURCE" --output "$OUTPUT" --workers 4
PYTHONPATH=training/src python training/scripts/validate_sonic_dex1_dataset.py --source "$SOURCE" --root "$OUTPUT"
```

在 training/ 中使用训练环境执行：
```bash
export WBWAM_SONIC_DATA_ROOT="$OUTPUT"
source scripts/load_env.sh
python scripts/compute_wb_norm_stats.py --output-dir "$OUTPUT/meta" wb_task=posttrain/sonic_dex1_20hz
python scripts/precompute_text_embeds.py wb_task=posttrain/sonic_dex1_20hz
python scripts/validate_sonic_dex1_dataset.py --root "$OUTPUT" --batch-only
```

WBWAM_MIDTRAIN_CHECKPOINT 应指向已下载的物理动作 midtrain 权重，用于初始化，
不作为 resume。数据必须搭配原生 SONIC 模型，不能直接交给原版 96 维动作头。
只有 .INCOMPLETE 已移除、完整验证与归一化 batch 检查均通过后，才能启动
另行安排的训练。

固定按完整 episode 顺序划分 90/10；93 条有效轨迹对应 83 条训练、10 条验证。
只用训练集计算 min/max 和官方逐动作步统计量（32 步）。统计量、真实语言缓存
和任务配置应与数据一同使用。每个训练样本为 32 个 20 Hz 动作和每隔 4 步取样
的 9 帧视频。

输出包含来源 episode/帧/时间戳、排除原因和数值、解码像素、视频 PTS 验证报告。
无损保存针对解码后的源图像，不是原压缩码流。数据验证通过不代表模型已收敛
或真机任务已成功；部署时不能将 20 Hz 动作序列直接按 50 Hz 播放。
