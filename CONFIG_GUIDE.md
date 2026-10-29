# 使用统一实验配置

`config.yaml` 保存参数，`experiment.py` 负责读取参数、设置随机种子、解析路径和配置 DataLoader。01—15 的实验脚本已接入；`models.py`、`losses.py` 保留可复用函数/类接口。

在 VS Code 中选择 `medai` 解释器。该环境已安装 PyYAML；新环境需要先执行 `python -m pip install pyyaml`。

## 先修改当前的肺炎分类实验

在根目录 `config.yaml` 中找到：

```yaml
seed: 42

pneumonia:
  validation_fraction: 0.2
  batch_size: 32
  learning_rate: 0.001
  epochs: 3
```

`seed` 控制随机初始化、随机数据、数据打乱和重新生成的数据划分。`batch_size` 是每批图片数量，`learning_rate` 是优化器步长，`epochs` 是训练轮数，`validation_fraction` 是从训练目录划给验证集的比例。

修改后保存，从 `04_train_pneumonia.py` 的第一个单元格重新运行，或在项目终端执行：

```powershell
conda activate medai
python 04_train_pneumonia.py
```

直接运行会训练并保存模型。04 保留原来的根目录权重文件位置和格式，并在 `results/pneumonia_config.yaml` 保存配置快照。

## Python 如何读取

已有脚本使用以下模式：

```python
from experiment import load_config, seed_everything, dataloader_options

cfg = load_config()
seed_everything(cfg["seed"], deterministic=cfg["runtime"]["deterministic"])

batch_size = cfg["pneumonia"]["batch_size"]
learning_rate = cfg["pneumonia"]["learning_rate"]

# DataLoader(..., shuffle=True, **dataloader_options(cfg))
```

YAML 用空格缩进；学习率写成 `0.0001`，避免把科学计数法误读为字符串。所有 `paths` 中的相对路径都以项目根目录为基准。

## 其他配置对应的实验

| 配置项 | 主要使用位置 |
|---|---|
| `runtime` | 设备、确定性算法、DataLoader worker 数 |
| `models`、`text` | 文本模型、投影维度、文本长度与编码批次 |
| `basics` | 01 合成二分类 |
| `pairs` | 08 图文划分及特征导出 |
| `contrastive` | 09 检查、10 投影训练 |
| `retrieval` | 11 随机检索重复次数 |
| `finetune` | 12 局部微调 |
| `robustness` | 14 噪声、删词、截断 |
| `survival` | 15 合成生存分析 |

11、12 加载旧检查点时，投影维度优先沿用检查点；12 的温度也沿用检查点，避免权重与结构不匹配。生效值与来源写入配置快照。修改投影维度后，应先运行 10 训练对应的新模型。

## 随机种子和已有缓存

- `seed_everything` 同时设置 Python、NumPy、PyTorch CPU/CUDA 随机数；独立的 NumPy `default_rng` 仍显式传入同一个种子。
- DataLoader 使用独立的有种子生成器，避免模型初始化或其他 loader 消耗随机数后改变打乱顺序。
- `deterministic: true` 启用确定性算法；遇到不支持的操作会报错。不同软件版本、设备之间仍不保证结果完全一致。
- Windows 的当前教学脚本包含顶层执行代码，`num_workers` 请保持 `0`；公共函数会拒绝非零设置，直到脚本加入 `__main__` 入口保护。
- 修改种子不会自动改写磁盘中的划分。希望改变图文 train/val/test 划分时，需要重跑 08。
- 修改文本模型或 `text.max_length` 后，需要重跑 07 生成文本特征，再运行 08 和后续实验；当前网络要求文本特征为 768 维。
- 新的独立 DataLoader 生成器改变了随机数的使用方式，因此不能要求新训练逐项复现之前保存的 CSV。已有权重和结果没有在本次配置接入时重生成。

10、11、12、14 实际运行时也会保存对应的配置快照；10、12 的新检查点还会包含配置字典。

验证配置及复现工具可以执行：

```powershell
python -m unittest discover -s tests -v
```

测试使用临时文件和小规模合成数据，不下载模型、不运行胸片训练、不覆盖已有结果。
