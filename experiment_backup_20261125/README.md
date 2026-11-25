# 当前项目实验数据备份（2026-11-25）

本目录保存 qwen3.6-27b-DPO 项目的必要实验材料。范围为服务器上的
`/data/liulin/llin-rl-dpo`、该项目的 `llin-rl-dpo-lrsearch` 学习率对照工作区，
以及挂载主工作区的 19 个相关容器中的自建复现材料。不包含后续 P2、paper、GRPO、物流等独立项目；主目录中嵌套的 paper6 目录也已排除。

## 已保存

- `raw/project/`：训练/失败日志与退出码、合成和真实复现数据、启动脚本、配置、补丁、实验记录。
- 训练输出中的 loss/reward 等指标、trainer state、TensorBoard events、训练曲线、base/adapter 推理对照结果，以及 adapter 配置等小型元数据。
- `raw/lrsearch/`：学习率对照工作区的脚本、配置、数据和日志。
- `raw/containers/`：latest MindSpeed-MM 的启动脚本、shape probe、配置、原始复现数据和 rotary 561002 错误日志。容器处于停止状态，通过其文件层只读获取，没有启动训练或修改服务器文件。
- `raw/MANIFEST.json`：逐文件来源、大小、SHA-256；排除文件及原因；框架版本线索；容器镜像 ID 和挂载信息（不采集环境变量）。
- `SHA256SUMS.txt`：全部 994 个源文件的校验值，下载后已逐项通过校验。源文件总计 109,631,429 字节，约 104.6 MiB。

真实业务复现数据按用户于 2026-11-25 的明确确认，纳入公开 GitHub 备份。已扫描备份文本，未发现疑似私钥、常见 API token 或明文密码。

## 明确未保存

模型基础权重、LoRA/adapter 权重、checkpoint 的权重分片、优化器/scheduler/RNG 二进制状态、DCP 权重、虚拟环境、依赖包、第三方框架全量源码、可再生 tokenization cache。输出目录中明确排除的文件共 113 个、约 4.55 GiB；被整目录排除的模型/checkpoints/cache 不计入该体积。

保留 checkpoint 目录中的 JSON 配置和 trainer state，只用于解释训练记录；本备份不能直接恢复模型训练。第三方框架仅保留可获取的版本信息和我们自建的 `llin_*` 文件；未取得版本号时 manifest 对应字段为 null。

本地另保留 `download.tar.gz`（21,637,961 字节）作为原始传输包，未重复提交到 GitHub。GitHub 中已保存完整解压内容。

## 检查和重新获取

在仓库根目录用 Python 检查：

```python
import hashlib, json
from pathlib import Path
root = Path('experiment_backup_20261125/raw')
manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
for item in manifest['files']:
    data = (root / item['path']).read_bytes()
    assert len(data) == item['bytes'], item['path']
    assert hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
print('Verified', len(manifest['files']), 'files')
```

需要重新获取时，使用 `scripts/backup_project_experiments.py`，传入新的目标目录、SSH host、port 和现有私钥路径。脚本仅执行只读远程操作，不备份凭据。

Windows 路径较长，克隆时建议使用 `git -c core.longpaths=true clone <仓库地址>`；本目录通过 `.gitattributes` 保留源文件字节，避免换行转换使校验值失效。
