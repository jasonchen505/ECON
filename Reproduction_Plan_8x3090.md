# ECON 框架复现计划 (8×RTX 3090)

> 基于实际硬件资源的完整复现方案
> 目标：全流程复现 ECON 框架，包括训练、推理、评估

---

## 一、资源评估与可行性分析

### 1.1 硬件资源

| 资源 | 规格 | 可用情况 |
|------|------|----------|
| GPU | 8× RTX 3090 (24GB VRAM each) | ✓ 可用 |
| 总显存 | 192GB (8×24GB) | ✓ 充足 |
| CPU | 待确认 | 需确认 |
| 内存 | 待确认 | 需确认 |
| 存储 | 待确认 | 需确认 (模型约100GB) |

### 1.2 模型显存需求估算

| 模型 | 参数量 | FP16 显存 | INT8 量化 | INT4 量化 |
|------|--------|-----------|-----------|-----------|
| Llama-3.1-70B | 70B | ~140GB | ~70GB | ~35GB |
| Llama-3.1-8B | 8B | ~16GB | ~8GB | ~4GB |
| bge-large-en-v1.5 | 0.3B | ~0.6GB | ~0.3GB | - |
| ECON 网络 | <0.1B | <0.2GB | - | - |

### 1.3 可行性结论

**✓ 完全可行**，方案如下：

| 组件 | 部署方案 | GPU 分配 |
|------|----------|----------|
| Coordinator (70B) | INT8 量化 + 张量并行 | GPU 0-3 (4卡) |
| Executor (8B) | FP16 单卡 | GPU 4 |
| bge-large-en-v1.5 | FP16 单卡 | GPU 5 |
| ECON 训练网络 | 单卡 | GPU 6 |
| 备用/测试 | - | GPU 7 |

**替代方案**（更节省显存）：
- 70B 用 INT4 量化：只需 2 卡
- 8B 和 bge 共享一卡
- 总共只需 3-4 卡

---

## 二、复现阶段规划

### 阶段 0: 环境准备 (Day 1)

**目标**：搭建基础环境

**任务清单**：
```bash
# 1. 创建 conda 环境
conda create -n econ python=3.10
conda activate econ

# 2. 安装 PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安装 vLLM (用于本地 LLM 推理)
pip install vllm

# 4. 安装其他依赖
cd /home/chenyizhou/ECON
pip install -r requirements.txt

# 5. 验证 GPU 状态
nvidia-smi
python -c "import torch; print(torch.cuda.device_count())"
```

**验证标准**：
- [ ] 8 张 GPU 都能被 PyTorch 识别
- [ ] vLLM 安装成功
- [ ] 所有依赖安装完成

---

### 阶段 1: 模型下载与部署 (Day 1-2)

**目标**：下载并部署所有需要的模型

#### 1.1 下载模型

```bash
# 创建模型目录
mkdir -p ~/models

# 方法 1: 使用 HuggingFace (需要 VPN 或镜像)
# 下载 Llama-3.1-70B-Instruct (需要申请权限)
huggingface-cli download meta-llama/Llama-3.1-70B-Instruct --local-dir ~/models/Llama-3.1-70B-Instruct

# 下载 Llama-3.1-8B-Instruct
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ~/models/Llama-3.1-8B-Instruct

# 下载 bge-large-en-v1.5
huggingface-cli download BAAI/bge-large-en-v1.5 --local-dir ~/models/bge-large-en-v1.5

# 方法 2: 使用 ModelScope (国内镜像)
# pip install modelscope
# modelscope download --model meta-llama/Llama-3.1-70B-Instruct --local_dir ~/models/Llama-3.1-70B-Instruct
```

**注意**：
- Llama-3.1-70B 约 140GB，下载需要时间
- 需要在 HuggingFace 申请 Llama 模型访问权限
- 或者用国内镜像 (ModelScope)

#### 1.2 部署 LLM 服务

**方案 A: 使用 vLLM 部署 (推荐)**

```bash
# 终端 1: 部署 Coordinator (70B, INT8 量化, 4卡张量并行)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
  --model ~/models/Llama-3.1-70B-Instruct \
  --served-model-name llama-70b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 4096 \
  --tensor-parallel-size 4 \
  --quantization bitsandbytes \
  --load-format bitsandbytes

# 终端 2: 部署 Executor (8B, FP16, 单卡)
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
  --model ~/models/Llama-3.1-8B-Instruct \
  --served-model-name llama-8b \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype auto \
  --max-model-len 4096
```

**方案 B: 使用 Together AI API (无需本地部署)**
```bash
# 设置 API Key
export TOGETHER_API_KEY="your_api_key"
```

#### 1.3 验证 LLM 服务

```bash
# 测试 Coordinator
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-70b",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "temperature": 0.3
  }'

# 测试 Executor
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-8b",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "temperature": 0.7
  }'
```

**验证标准**：
- [ ] 两个 LLM 服务都能正常响应
- [ ] 响应时间在可接受范围内 (<5s)
- [ ] GPU 显存使用合理

---

### 阶段 2: 代码适配 (Day 2)

**目标**：修改代码以适配本地部署

#### 2.1 修改 LLM Wrapper 配置

```python
# src/modules/llm/llm_wrapper.py
# 修改默认 base_url 指向本地服务

# Coordinator 使用 8000 端口
# Executor 使用 8001 端口
```

#### 2.2 创建本地部署配置文件

```yaml
# scripts/config_p0_local.yaml
# 复制 config_p0.yaml 并修改以下内容

llm_model_name: "gpt2"  # tokenizer 用 gpt2 即可
together_api_key: "dummy"  # 本地部署不需要真实 key

coordinator_model: "llama-70b"  # vLLM served model name
executor_model: "llama-8b"      # vLLM served model name

# 或者使用不同端口
# coordinator_base_url: "http://localhost:8000/v1"
# executor_base_url: "http://localhost:8001/v1"
```

#### 2.3 修改 LLM Wrapper 支持多端点

```python
# src/modules/llm/llm_wrapper.py
# 添加支持不同模型使用不同 base_url 的功能

class ImprovedLLMWrapper:
    def __init__(self, api_key, model_name, base_url=None, ...):
        if base_url is None:
            # 根据模型名自动选择端口
            if "70b" in model_name.lower() or "coordinator" in model_name.lower():
                base_url = "http://localhost:8000/v1"
            else:
                base_url = "http://localhost:8001/v1"
        ...
```

**验证标准**：
- [ ] 代码能正确调用本地 LLM 服务
- [ ] Coordinator 和 Executor 使用不同的端点

---

### 阶段 3: 小规模验证 (Day 2-3)

**目标**：用少量数据验证整个流程

#### 3.1 运行 Quick Test

```bash
cd /home/chenyizhou/ECON

# 设置环境变量
export TOGETHER_API_KEY="dummy"

# 运行快速测试 (5 个训练 episode, 3 个测试 episode)
python scripts/run_p0_test.py \
  --train-eps 5 \
  --test-eps 3 \
  --log-dir logs_quick_test \
  --model-dir models_quick_test
```

#### 3.2 检查输出

```bash
# 查看训练日志
cat logs_quick_test/training_metrics.csv

# 查看 LLM 调用 trace
cat logs_quick_test/llm_traces_train.json | python -m json.tool

# 查看测试结果
cat logs_quick_test/llm_traces_test_bne_3rounds.json | python -m json.tool
```

#### 3.3 常见问题排查

**问题 1: LLM 调用超时**
```bash
# 检查 vLLM 服务是否正常
curl http://localhost:8000/v1/models

# 增加超时时间
# 在 config 中设置: llm.timeout: 120
```

**问题 2: GPU 显存不足**
```bash
# 检查显存使用
nvidia-smi

# 解决方案：
# 1. 减少 max-model-len
# 2. 使用更激进的量化
# 3. 减少 tensor-parallel-size (如果用更多卡)
```

**问题 3: 模型加载失败**
```bash
# 检查模型路径
ls ~/models/

# 检查模型完整性
python -c "from transformers import AutoModelForCausalLM; model = AutoModelForCausalLM.from_pretrained('~/models/Llama-3.1-8B-Instruct')"
```

**验证标准**：
- [ ] 训练流程能跑通
- [ ] 测试流程能跑通
- [ ] 能得到初步的准确率数据

---

### 阶段 4: 完整训练 (Day 3-5)

**目标**：完成完整的训练流程

#### 4.1 训练配置

```yaml
# scripts/config_p0_full.yaml
# 复制 config_p0.yaml 并修改

t_max: 300  # 训练 300 个 episode
test_interval: 50  # 每 50 个 episode 测试一次
test_nepisode: 50  # 测试 50 个 episode

logging:
  save_model_interval: 50
  checkpoint_path: "./models_full"
  log_path: "./logs_full"
  experiment_name: "econ_full_training"
```

#### 4.2 运行完整训练

```bash
# 使用 tmux 或 screen 保持训练进程
tmux new -s econ_train

# 运行训练
python scripts/run_p0_test.py \
  --train-eps 300 \
  --test-eps 50 \
  --config scripts/config_p0_full.yaml \
  --log-dir logs_full \
  --model-dir models_full

# 分离 tmux: Ctrl+B, D
# 重新连接: tmux attach -t econ_train
```

#### 4.3 监控训练

```bash
# 实时查看训练日志
tail -f logs_full/training_metrics.csv

# 查看 GPU 使用
watch -n 1 nvidia-smi

# 查看训练曲线 (如果有 tensorboard)
tensorboard --logdir logs_full
```

#### 4.4 训练时间估算

| 阶段 | Episode 数 | 预计时间 |
|------|-----------|----------|
| 单 Episode | 1 | ~30-60s (取决于 LLM 响应) |
| 训练 300 Episode | 300 | ~2.5-5 小时 |
| 测试 50 Episode | 50 | ~25-50 分钟 |
| **总计** | - | ~3-6 小时 |

**注意**：
- 时间主要取决于 LLM 推理速度
- 本地 vLLM 比 API 调用更快
- 可以并行测试以节省时间

**验证标准**：
- [ ] 训练完成，损失下降
- [ ] 模型保存成功
- [ ] 得到最终准确率

---

### 阶段 5: 评估与分析 (Day 5-6)

**目标**：全面评估训练结果

#### 5.1 运行完整评估

```bash
# 测试训练好的模型
python scripts/test_p0.py \
  --model-dir models_full/final \
  --test-eps 100 \
  --log-dir logs_eval
```

#### 5.2 分析结果

```python
# 分析脚本
import json

# 加载测试 traces
with open('logs_eval/llm_traces_test_p0.json', 'r') as f:
    traces = json.load(f)

# 计算准确率
correct = sum(1 for t in traces if t.get('is_correct'))
total = len(traces)
accuracy = correct / total * 100

# 分析错误类型
errors = [t for t in traces if not t.get('is_correct')]
for t in errors[:5]:
    print(f"Pred: {t.get('pred_answer')}, GT: {t.get('ground_truth')}")

# 分析 BNE 收敛情况
converged = sum(1 for t in traces if t.get('bne_converged'))
print(f"BNE Convergence Rate: {converged/total*100:.1f}%")
```

#### 5.3 对比实验

```bash
# 对比: 无 BNE (基线)
python scripts/test_p0.py \
  --model-dir models_full/final \
  --test-eps 100 \
  --bne-rounds 0 \
  --log-dir logs_eval_baseline

# 对比: 不同 BNE 轮数
for rounds in 1 2 3 5; do
  python scripts/test_p0.py \
    --model-dir models_full/final \
    --test-eps 100 \
    --bne-rounds $rounds \
    --log-dir logs_eval_bne_${rounds}
done
```

**验证标准**：
- [ ] 得到完整的准确率数据
- [ ] 对比分析完成
- [ ] 错误案例分析完成

---

### 阶段 6: 文档与总结 (Day 6-7)

**目标**：整理复现文档

#### 6.1 记录关键发现

```markdown
# 复现总结

## 环境配置
- GPU: 8× RTX 3090
- CUDA: 11.x
- PyTorch: 2.x
- vLLM: x.x.x

## 模型配置
- Coordinator: Llama-3.1-70B (INT8, 4卡)
- Executor: Llama-3.1-8B (FP16, 1卡)
- Embedding: bge-large-en-v1.5

## 训练结果
- 训练 Episode: 300
- 最终准确率: XX%
- BNE 收敛率: XX%
- 训练时间: X 小时

## 关键发现
1. ...
2. ...
3. ...
```

#### 6.2 整理代码修改

```bash
# 查看所有修改
git diff

# 创建补丁
git diff > my_changes.patch
```

---

## 三、GPU 资源分配方案

### 方案 A: 标准配置 (推荐)

| GPU | 用途 | 显存使用 |
|-----|------|----------|
| 0 | Coordinator (70B) TP shard 0 | ~18GB |
| 1 | Coordinator (70B) TP shard 1 | ~18GB |
| 2 | Coordinator (70B) TP shard 2 | ~18GB |
| 3 | Coordinator (70B) TP shard 3 | ~18GB |
| 4 | Executor (8B) | ~16GB |
| 5 | bge-large-en-v1.5 | ~1GB |
| 6 | ECON 训练网络 | <1GB |
| 7 | 备用/测试 | - |

**总显存使用**: ~90GB / 192GB (47%)

### 方案 B: 节省配置

| GPU | 用途 | 显存使用 |
|-----|------|----------|
| 0-1 | Coordinator (70B) INT4 TP=2 | ~18GB each |
| 2 | Executor (8B) + bge | ~17GB |
| 3 | ECON 训练网络 | <1GB |
| 4-7 | 备用 | - |

**总显存使用**: ~54GB / 192GB (28%)

### 方案 C: 最小配置

| GPU | 用途 | 显存使用 |
|-----|------|----------|
| 0 | Coordinator (70B) INT4 | ~35GB (需要 >24GB，不可行) |

**结论**: 3090 单卡无法放下 70B，至少需要 2 卡

---

## 四、性能优化建议

### 4.1 LLM 推理优化

```bash
# vLLM 优化参数
--enable-prefix-caching  # 启用前缀缓存
--max-num-batched-tokens 8192  # 增加批处理 token 数
--gpu-memory-utilization 0.9  # 提高 GPU 利用率
```

### 4.2 训练优化

```python
# 使用混合精度训练
torch.backends.cudnn.benchmark = True
torch.cuda.amp.autocast(enabled=True)

# 增加批处理大小
batch_size: 16  # 从 8 增加到 16

# 使用梯度累积
gradient_accumulation_steps: 2
```

### 4.3 数据加载优化

```python
# 使用多进程数据加载
num_workers: 4
pin_memory: True

# 预加载数据
prefetch_factor: 2
```

---

## 五、时间线总览

| 天数 | 阶段 | 任务 | 产出 |
|------|------|------|------|
| Day 1 | 环境准备 | 安装依赖、下载模型 | 环境就绪 |
| Day 1-2 | 模型部署 | 部署 vLLM 服务 | LLM 服务可用 |
| Day 2 | 代码适配 | 修改配置、适配本地部署 | 代码就绪 |
| Day 2-3 | 小规模验证 | 运行 quick test | 流程验证 |
| Day 3-5 | 完整训练 | 运行 300 episode 训练 | 模型训练完成 |
| Day 5-6 | 评估分析 | 运行评估、分析结果 | 评估报告 |
| Day 6-7 | 文档总结 | 整理文档、总结发现 | 复现文档 |

**总时间**: 7 天

---

## 六、风险与应对

### 6.1 潜在风险

| 风险 | 概率 | 影响 | 应对方案 |
|------|------|------|----------|
| 模型下载失败 | 中 | 高 | 使用镜像、离线包 |
| GPU 显存不足 | 低 | 中 | 使用更激进量化 |
| LLM 响应慢 | 中 | 中 | 优化 vLLM 参数 |
| 训练不收敛 | 低 | 高 | 调整超参数 |
| 代码 bug | 中 | 中 | 查看 issue、调试 |

### 6.2 备选方案

**如果 70B 模型无法部署**：
- 使用更小的模型：Llama-3.1-8B 作为 Coordinator
- 使用 API：Together AI 或 OpenAI
- 使用量化：GPTQ/AWQ 4-bit 量化

**如果训练不收敛**：
- 减少 Agent 数量：从 3 减少到 2
- 调整学习率：从 0.0004 调整到 0.0001
- 增加 BNE 轮数：从 1 增加到 3

---

## 七、检查清单

### 环境检查
- [ ] Python 3.10+ 安装
- [ ] PyTorch 2.0+ 安装 (CUDA 11.8)
- [ ] vLLM 安装
- [ ] 8 张 GPU 识别

### 模型检查
- [ ] Llama-3.1-70B 下载完成
- [ ] Llama-3.1-8B 下载完成
- [ ] bge-large-en-v1.5 下载完成
- [ ] 模型完整性验证

### 服务检查
- [ ] Coordinator vLLM 服务启动
- [ ] Executor vLLM 服务启动
- [ ] 服务健康检查通过
- [ ] 响应时间可接受

### 训练检查
- [ ] Quick test 通过
- [ ] 完整训练完成
- [ ] 模型保存成功
- [ ] 评估完成

### 文档检查
- [ ] 环境配置文档
- [ ] 训练日志保存
- [ ] 评估结果保存
- [ ] 复现总结文档

---

*最后更新: 2026年6月*
*基于 8×RTX 3090 硬件配置*
