# Jev 短信过滤真实实验

本项目在 macOS ARM64 上使用 CPU 实际下载 Kaggle 数据、调用 `typesafe/jev-1.13` 并训练 TF-IDF＋逻辑回归。结果和限制见 [REPORT.md](REPORT.md)，一页说明见 [ONE_PAGE.md](ONE_PAGE.md)。原工作目录的 Notebook 未修改。

公开仓库：<https://github.com/Abchyy/jev-sms-experiment>。克隆后复算已有真实结果，不需要 API 密钥：

```sh
git clone https://github.com/Abchyy/jev-sms-experiment.git
cd jev-sms-experiment
UV_CACHE_DIR="$PWD/.uv-cache" uv sync --frozen --python 3.12
.venv/bin/python experiment.py reproduce
```

仓库包含本次实验的公开数据、固定划分、真实请求体及响应（无认证头）、基线、指标和报告。未包含原目录 Notebook、临时 API 密钥、虚拟环境及运行缓存目录。`results/cache/` 是可复算成绩所需的真实响应文件，已保留。

## 复现已有成绩：不产生 API 请求

```sh
cd /Users/abc/Desktop/jev/jev-sms-experiment
.venv/bin/python experiment.py reproduce
```

此命令校验划分哈希，读取真实 API 缓存与已训练基线，重算指标和图表；保留原始训练及网络耗时。复算指标不需要密钥或联网。

## 环境配置

Python 3.12.13；依赖全锁定在 `uv.lock` 与 `requirements.lock.txt`，实测版本见 `results/environment.json`。优先使用 uv，缓存指定到项目中，不改全局 Python。

```sh
UV_CACHE_DIR="$PWD/.uv-cache" uv sync --frozen --python 3.12
```

无 uv 时：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.lock.txt
```

## 分阶段命令

```sh
.venv/bin/python experiment.py prepare
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python experiment.py baseline
.venv/bin/python experiment.py smoke
.venv/bin/python experiment.py dev
.venv/bin/python experiment.py freeze
.venv/bin/python experiment.py test
.venv/bin/python experiment.py evaluate
```

一次执行整个流程：`.venv/bin/python experiment.py all`。已成功调用会命中缓存，失败样本不会自动重试。复现时优先使用 `reproduce`，避免意外新增调用。

新环境进行真实调用前，用终端隐藏输入把密钥放进当前进程环境，避免写入 shell 历史：

```sh
read -rs 'OPENROUTER_API_KEY?OpenRouter key: '
export OPENROUTER_API_KEY
```

程序优先读取进程环境变量，必要时读取本地 `.env`（600 权限）；`.gitignore` 已提前排除 `.env`。不要把 `.env` 加入提交、归档或分享。临时密钥仅用于本次实验，报告不含密钥。实验完成后按用户授权公开发布到 GitHub；没有发布网站。

## 数据、配置与结果

- `data/kaggle.zip`、`data/spam.csv`：实际公开下载的原始数据；`data/manifest.json` 包含 URL、哈希、字段、清洗计数、类别比例和划分哈希。
- `data/clean.csv`、`data/train.csv`、`data/dev.csv`、`data/test.csv`：完整固定划分，包含真实标签、文本和 SHA-256。原始 `v1`、`v2` 分别作为标签、文本，不拼接三个额外列。
- `data/*_inputs.json`：身份、文本哈希和文本；API 仅发送文本，不发送身份、标签、预测、拆分名称或统计信息。
- `config/pre_registered.json`：API 调用前的配置与划分登记；`config/frozen.json`：测试前冻结的标准、阈值、基线和哈希。
- `results/attempts.jsonl`：逐次开始／结束记录；`results/cache/*.json`：无认证头的请求体、原始响应字符串、解析结果、费用和网络耗时。
- `results/dev_predictions.csv`、`results/test_predictions.csv`：逐条预测，测试 CSV 包含同一样本上的基线结果。
- `results/metrics.csv`：Jev 有效响应、基线同一有效子集、基线全部测试集三种分母。
- `results/workflow_metrics.csv`：全部 100 条为分母的概率策略成绩，失败或未调用一律人工复核。
- `results/api_metrics.csv`、`results/api_phase_costs.csv`、`results/baseline_timing.csv`：成功率、延迟、分阶段费用和本地耗时。
- `results/calibration_bins.csv`、`results/confusion_matrix.png`、`results/calibration.png`：校准分箱计数与图表。
- `results/errors_all.csv`：全部错误；`results/errors_representative.csv`：最多 10 条、按预测类别概率降序选取的错误。
- `results/baseline_model.joblib`、`results/baseline_timing.json`：CPU 模型与原始耗时。仅加载可信的本地模型文件。
- `docs/`：核实过的官方文档快照与来源记录。

## 调用与评测约束

每条短信独立、顺序请求；连接超时 10 秒，读取超时 30 秒，禁用自动重试与重定向。最大 150 次；费用按响应 `usage.cost` 汇总，调用前预留当前官方 32K 上下文的最大输入费 $0.001344；目标 $0.10。官方价格是输入 $0.042／百万 token，输出免费。本地统计不是账单硬上限，超时、服务端失败和中断可能已计费，账户账单为准。

HTTP 400/401/402/403/404/413、连续三次失败、费用缺失或超时会写入 `results/api_stop.json` 并停止。较用户要求更保守：费用不明时即停，不等待三次失败。不要删除停止记录或缓存盲目重跑；先核对 OpenRouter Activity 和余额，不自动充值或修改账户设置。

缓存键含数据源文件哈希、完整文本、模型、接口、判断标准版本及完整 questions。开始事件先持久化；中断留下未完成事件时不会重新请求。`typesafe/jev-1.13` 是版本别名，实际服务快照另行记录；未来在线重跑可能不同，已有响应缓存可精确重算本次指标。

最终已执行 `verify_delivery.py`：独立重算所有主要指标、核对原始响应与冻结时序、核验费用及保存的基线模型，并复跑三阶段缓存确认零新增请求。结果见 `results/verification.json`。临时 `.env` 已在秘密扫描通过后删除，交付归档没有密钥。

如需再次实际训练基线而保留原成绩，可以在新目录解压交付包，删除该副本中的 `results/baseline_predictions.csv`，再运行 `baseline` 和 `evaluate`；只在训练集重新拟合，不调用 Jev。时间测量会随机器负载变化。

阈值固定为 `P(spam) ≤ 0.10` 放行、`≥ 0.90` 拦截、中间人工复核；不使用 confidence 作为正确率。基线使用相同阈值作辅助工作流对比，分类使用固定 0.5，不校准也不调参。人工复核没有被假设为正确。

测试集 spam 比例 50%，这是平衡小样本设计。Precision、Brier、复核比例不能直接移植到真实流量；经典公开数据可能被预训练见过。本实验检验文本分类和简单分流，不能证明通用规划、因果决策或强化学习控制能力。
