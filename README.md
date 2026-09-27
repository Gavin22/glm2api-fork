# ChatGLM 2 API

`glm2api` 是一个本地代理服务，用来把 `chatglm.cn` 的网页接口转换成 OpenAI 兼容接口，方便你直接接入 OpenAI SDK、Cherry Studio、Open WebUI、LobeChat 或其他兼容 OpenAI API 的工具。

支持的主要接口：

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/images/generations`
- `GET /v1/models`
- `GET /health`

> 本仓库是 [XxxXTeam/glm2api](https://github.com/XxxXTeam/glm2api) 的改造分支(基线 `f97d6eb`),修掉了工具调用链路的一批缺陷并补充了可靠性测量手段。
> 改造内容、部署要点与已知边界见文末 [改造说明](#12-改造说明fork-分支)。

## 1. 使用前准备

启动前请确认：

- 你已经登录过 `https://chatglm.cn`
> 其实不登陆也行,但是会有部分限制?
- 你能获取到有效的 `refresh_token`，或者接受游客模式的能力限制
- 本地已准备好 Python 虚拟环境

## 2. 获取 GLM Refresh Token / 游客模式

获取方式：

1. 打开 `https://chatglm.cn`
2. 登录你的账号
3. 按 `F12` 打开开发者工具
4. 进入 `Application`
5. 查看 `Local Storage` 或相关存储项
6. 找到 `chatglm_refresh_token`

拿到后，将它填入 `.env` 文件中的：

```env
GLM_REFRESH_TOKEN=你的_refresh_token
```

如果你不想登录账号，也可以直接启用游客模式：

```env
GLM_USE_GUEST_REFRESH_TOKEN=true
```

如果既没有配置 `token.txt`，也没有配置 `GLM_REFRESH_TOKEN`，程序也会自动退回游客模式，并在请求失败时自动重新获取新的游客 `refresh_token` 后重试。

## 3. 配置文件

先复制示例配置：

```bash
cp .env.example .env
```

如果当前目录没有 `.env`，程序启动时也会自动从 `.env.example` 复制一份默认配置再继续加载。

推荐优先准备 `token.txt`，每行一个账号的 `refresh_token`：

```text
token-a
token-b
token-c
```

如果你暂时只有一个账号，也可以继续只改 `.env` 里的这一项：

```env
GLM_REFRESH_TOKEN=你的_refresh_token
```

如果你想显式固定走游客模式，可以这样写：

```env
GLM_USE_GUEST_REFRESH_TOKEN=true
GLM_GUEST_MAX_RETRIES=3
```

启用游客模式后，程序会按 `GLM_MAX_CONCURRENCY` 自动创建同等数量的游客账号槽位，让每个并发请求优先使用独立游客账号，避免多个并发长期挤在同一游客会话上。

常用配置说明：

- `HOST`
  服务监听地址。只给本机使用时填 `127.0.0.1`，局域网访问可填 `0.0.0.0`

- `PORT`
  服务端口，默认 `8000`

- `API_PREFIX`
  OpenAI 兼容路径前缀，默认 `/v1`

- `DEBUG_DUMP_ALL`
  调试狂暴模式。开启后会自动切到 `DEBUG`，并打印入站原始请求、转发给 GLM 的原始 body、上游原始响应和 SSE 分片、工具调用转换结果等几乎所有调试信息
  当 LOG_LEVEL=DEBUG（或 DEBUG_DUMP_ALL=true）时，自动在 log/glm2api_debug.log 写入日志文件（LOG_LEVEL=INFO — 只有终端输出，不写文件）

- `GLM_ASSISTANT_ID`
  普通对话使用的 assistant id

- `GLM_TOKEN_FILE`
  多账号 token 文件路径，默认 `token.txt`，每行一个 `refresh_token`

- `GLM_IMAGE_ASSISTANT_ID`
  图片生成使用的 assistant id

- `GLM_USE_GUEST_REFRESH_TOKEN`
  显式启用游客 ck；开启后会忽略已配置的账号 token

- `GLM_GUEST_MAX_RETRIES`
  游客 ck 请求失败时，最多自动重新拉取游客 token 并重试多少次

- `GLM_DELETE_CONVERSATION`
  是否在请求结束后自动删除 GLM 会话记录

- `GLM_MAX_CONCURRENCY`
  本地代理允许同时占用的上游执行槽位数量，默认 `3`

- `SERVER_API_KEYS`
  如果你希望访问本地代理时也带 Bearer Token，可以在这里填写

说明：

- 如果存在 `token.txt`，程序会优先从这里加载多账号
- 如果显式设置了 `GLM_USE_GUEST_REFRESH_TOKEN=true`，程序会直接走游客模式
- 游客模式下会按 `GLM_MAX_CONCURRENCY` 自动扩展游客账号池，尽量做到每个并发槽位对应一个独立游客账号
- 当某个账号请求失败时，会自动切换到下一账号继续尝试
- 如果本轮所有账号都失败，下一次会从第一个账号重新开始
- 当上游返回新的 `refresh_token` 时，多账号模式会自动写回 `token.txt` 对应行
- 单账号兜底模式下，程序仍会自动写回 `.env`
- 游客模式下不会把临时游客 `refresh_token` 落盘到 `.env` 或 `token.txt`
- 如果完全没有配置账号 token，程序会自动获取游客 `refresh_token` 作为兜底
- 如果你的 `.env` 不存在，程序无法自动落盘新的 token
- `/v1/models` 返回的模型列表已经固定写在代码中，不再通过配置文件自定义

## 4. 启动服务

### 拉取源代码

```bash
git clone https://github.com/XxxXTeam/glm2api.git
```
### 安装依赖

```bash
uv sync
```
### 运行项目

```bash
uv run .\main.py
```

或者：

```bash
.\.venv\Scripts\python.exe main.py
```

启动成功后你会看到类似日志：

```text
启动服务 host=127.0.0.1 port=8000 prefix=/v1 models=...
```

## 5. 健康检查

```bash
curl http://127.0.0.1:8000/health
```

返回示例：

```json
{"status":"ok"}
```

## 6. 查询模型列表

```bash
curl http://127.0.0.1:8000/v1/models
```

返回的是当前配置里暴露的模型列表。

## 7. 聊天接口

### 7.1 Curl 示例

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"glm-4\",\"messages\":[{\"role\":\"user\",\"content\":\"你好，介绍一下你自己\"}]}"
```

### 7.2 Python OpenAI SDK 示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy",
)

resp = client.chat.completions.create(
    model="glm-4",
    messages=[
        {"role": "user", "content": "你好，介绍一下你自己"}
    ],
)

print(resp.choices[0].message.content)
```

### 7.3 流式示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy",
)

stream = client.chat.completions.create(
    model="glm-4",
    messages=[{"role": "user", "content": "写一首七言绝句"}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if getattr(delta, "content", None):
        print(delta.content, end="")
```

### 7.4 OpenAI Responses API 示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy",
)

resp = client.responses.create(
    model="glm-4",
    input=[
        {"role": "user", "content": "你好，介绍一下你自己"}
    ],
)

print(resp.output_text)
```

## 8. 图片生成接口

### 8.1 Curl 示例

```bash
curl http://127.0.0.1:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"glm-image-1\",\"prompt\":\"画个枫叶\",\"size\":\"1024x1024\"}"
```

### 8.2 Python OpenAI SDK 示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy",
)

image = client.images.generate(
    model="glm-image-1",
    prompt="画个枫叶",
    size="1024x1024",
)

print(image.data[0].url)
```

### 8.3 当前支持的图片参数

- `prompt`
- `model`
- `n`
- `size`
- `response_format`
- `style`
- `scene`

说明：

- 默认返回图片 URL
- 如果 `response_format=b64_json`，会返回 base64 图片数据
- `size` 会自动映射到 GLM 所需的宽高比例

## 9. 鉴权方式

如果 `.env` 中 `SERVER_API_KEYS` 为空，则本地接口默认不校验 Bearer Token。

如果你配置了：

```env
SERVER_API_KEYS=sk-local-1,sk-local-2
```

那么请求时需要带：

```http
Authorization: Bearer sk-local-1
```

## 10. 日志说明

程序默认输出彩色日志，常见内容包括：

- 服务启动
- 请求进入队列
- 并发槽位获取/释放
- 上游请求转发
- 会话删除结果
- 错误原因

如果你想查看更多细节，可以把 `.env` 中的：

```env
LOG_LEVEL=DEBUG
```

## 11. 常见问题

### 11.1 启动时报 `GLM_REFRESH_TOKEN` 缺失

新版本默认会自动退回游客模式；如果你仍想固定使用账号，请检查 `.env` 或 `token.txt` 里的 `refresh_token` 是否填写正确。

### 11.2 返回“请等待其他对话生成完毕”

说明同一账号在 GLM 侧存在并发限制。程序已经内置串行队列和自动等待重试。

### 11.3 返回“请登录后继续使用”

说明当前账号状态无效，或者 token 已失效，需要重新登录并更新 `refresh_token`

## 12. 改造说明(fork 分支)

以下内容相对上游 `XxxXTeam/glm2api` @ `f97d6eb`。

数值口径:所有成功率均为 **端到端可用调用率**(含真实执行),由 `tools/agent_loop_probe.py`
打真实上游测得;`--mock` 自检 3/3 用于证明测量本身可信。

### 2026-09-27

#### 17:21 基线 `a93a524`

从上游 tarball 导入,29 个文件。此时的状态:

- 工具调用全部失效(`0/12`)
- 大参数流式解析为 O(n²):100KB 需 0.96s
- `Edit.old_string` / `Write.content` 无法与磁盘内容匹配

#### 17:28 `3cedf9b` 逐字保真 + 截断恢复

`_leaf_text` 原先折叠空白,导致每个文件编辑都匹配失败。改为 `"".join(element.itertext())` 逐字返回。
新增 `_close_truncated_block` / `_recover_trailing_block`,处理 `max_tokens` 截断后缺失闭合标签的情形。

#### 17:32 `8ac4746` 禁止空回合

上游有时返回既无文本也无工具调用的回合,客户端表现为卡死。
新增 `_describe_unusable_turn()`,保证永远不会发出空回合。

#### 17:35 `a500158` JSON 漂移恢复

模型偶尔把工具调用写成 ```` ```json ````、`<tool_call>`、`Bash({...})` 等形态而非 DSML 协议。
新增 `parse_json_tool_calls_from_text`,在 finalize 阶段把这些形态回收成合法调用,而不是当散文发给客户端。

#### 18:26 `cd1eb46` 流式线性化

流式解析大工具参数从 O(n²) 降为线性:

| 输入 | 改前 | 改后 |
|---|---|---|
| 100KB | 0.96s | 0.003s |
| 400KB | 未测 | 0.011s |

手段:append-only 缓冲、`_mask_code_fences` 无围栏时直接返回原串、`_find_partial_tag_start` 只看尾部 `MAX_TAG_HINT_LENGTH`。

#### 18:51 `60c3ced` 首个闭合标签截断(严重)

`_find_matching_block` 取**第一个**闭合标签。当 `Write` 的文件内容里含有协议自身标签
(`</|DSML|tool_calls>`)时,调用会在 173 字符中的第 115 字符处被截断 ——
整个调用丢失,原始标记泄漏成助手消息。

改为 `_candidate_block_spans` 生成器:逐个尝试可能的块结束位置,**第一个能真正解析成工具调用的候选胜出**。
配合 `_escape_protocol_tags_inside_values`,在解析前转义参数值内部的结构标签。

**这个缺陷是写 e2e 测试时发现的,不是读代码发现的。**

#### 18:57 `4fc30a6` agent 循环探针

`tools/agent_loop_probe.py`:按 Claude Code 的形态驱动多轮循环(声明工具集、回放 `tool_use`、
本地执行工具),把每一轮归类为:有调用 / 纯文本 / 空回合 / 参数不可解析 / 缺必需参数 /
未声明工具 / 工具报错 / 干净执行。

输出端到端可用调用率和失败模式分布 —— 这是解析器单测给不出的数字。

#### 20:06 `1f24195` 移除测试残留 + 镜像排除密钥

两件事:

1. 探针的假模型模板字符串未格式化,`%s` 原样落盘成 `%s/greeting.py` 目录,被误提交进 `4fc30a6`。
2. `Dockerfile` 是 `COPY . .` 且没有 `.dockerignore` —— **`.env`(真实 refresh token + API key)
   会被烤进每一个镜像层**。

两者都已修复。新增 `.dockerignore` 排除 `.env`、`tests/`、`tools/`。

#### 20:24 `a4b1a82` 探针自检不再污染仓库

给假模型一个真实路径(临时目录),自检不再往仓库根目录写文件。
新增 `tools/apply_bundle_update.py`:把 git bundle 的**追踪文件**合并进部署目录,`.env` 天然不在名单上。

#### 21:22 `ab7fb8f` 【核心修复】增量与快照不一致

**这是本轮唯一让工具调用真正可用的改动。**

根因(用真实抓包定位,`tools/dump_snapshots.py` 输出):

```
28 个快照,前 27 个是纯增量
拼接前 27 个长度 = 492,最后一条快照长度 = 492,完全相等
```

上游在散文阶段发**增量**,进入工具调用后切成**全量快照**。

老代码记录的是「可见文本长度」——即 `StreamingToolParser` 缓冲后存活下来的部分。
可见长度会独立于原始快照变化(解析器找工具块时会扣住尾部,恢复时又会补上 `\n` 分隔符、
对重叠片段去重)。按旧偏移切片,就把无关碎片拼进了助手消息。

后果比预想更重:解析器在扫描时会**改写**内容(`</|DSML|invoke<|DSML|invoke` 被合并、
多了 `\n`),所以磨碎的文本本身已不再包含完整调用 —— 三条并行线索在此汇合。

修法:改为对**原始累计内容**做前缀校验,只在「新快照是旧的扩展」时发出增量。
非扩展的快照直接跳过(客户端已有该文本,重发只会重复或丢内容)。

回归测试:新增 2 个,`translator.py` 被 `git stash` 后测试**确实失败**,修复后通过。

**实测对比(真实 upstream,glm-5.3,流式):**

| 环境 | 改前 | 改后 |
|---|---|---|
| 本地 | 0/12 | 14/15 (93.3%) |
| 线上部署 | 0/12 | 12/12 轮有调用,13 个调用全部执行成功 |

唯一失败是模型**用散文作答**,不是解析问题。

#### 21:26 `a040c2f` 探针抗非 UTF-8

`run_tool` 用 `text=True` 解码 Bash 输出,遇非 UTF-8 字节直接 `UnicodeDecodeError` 中断测量,
且假设两个管道都是 `str`。改为宽松解码 + 空值兜底。

---

### 12.1 部署记录

服务器:`114.115.179.97`,宝塔面板,Docker 默认路径 `/www/dk_project/`。
项目:`/www/dk_project/wwwroot/glm2api`,Nginx 反代 `glm.didikaiche.cn`。

#### 20:38 第一次部署

- 容器从 `Up 3 months` 重建成功
- 模型数 **78 → 82**,`glm-5.3` 从无到有
- 真实对话 `UPGRADE_OK`
- 工具调用仍为 `0/6` —— 因为此时 diff bug 未修

#### 21:2x 第二次部署(带 diff 修复)

- 构建 → 重建 → 健康检查 200
- 线上探针 12/12 轮工具命中

---

### 12.2 注意事项

#### 部署必须分两步,不能合并

```bash
docker build --progress=plain -t glm2api-glm2api:latest .   # 第一步
docker compose up -d                                        # 第二步,不要加 --build
```

`docker compose up -d --build` 会调 `buildx bake`,**在这台机器上静默挂死**:
24 分钟只用了 6 秒 CPU,零输出,而 `docker ps` 仍显示旧容器正常运行 ——
看起来一切正常,实际什么都没发生。

#### 每次部署必须验证,否则不知道有没有生效

```bash
docker ps --filter name=glm2api --format '{{.Names}}\t{{.Status}}'   # 启动时间必须是刚刚
curl -s -H "Authorization: Bearer $KEY" http://127.0.0.1:8000/v1/models | grep -c glm-5.3
```

判断升级是否生效的硬指标:模型数 82、`glm-5.3` 存在。上次就是靠 `Up 3 months` 才发现容器压根没重建。

#### 每次合并前记录 `.env` 指纹

```bash
md5sum .env      # 合并前
md5sum .env      # 合并后,必须一致
```

当前指纹 `81c6eee4f2f53574523af3de0678468e`。`apply_bundle_update.py` 只覆盖 git 追踪文件,
`.env` 不在名单上,但指纹比对是最后一道保险。

#### 代码传输用 bundle,不用 git pull

服务器无法读取私有仓库(SSH key 未加入 GitHub),且服务器到 github.com 的连接不稳定。
流程:`git bundle create` → `scp` → 服务器解包 → 合并。

#### 公开仓库无密钥(已验证)

对**所有 refs 的 53 个 blob(含全部历史)** 扫描,拿 `.env` 的 7 个真实值逐项比对:

```
JWT 形态 token (eyJhbGciOi)   ->  none
服务端 API key (sk-glm2api)   ->  none
.env 这个路径                  ->  从未被提交过
didikaiche / 114.115.179.97 / /www/dk_project / yuemipro / zsydd  ->  全部 none
```

`.env` 在 `.gitignore` 中;`.dockerignore` 也已排除。

#### 已知未修(经决定暂缓)

`glm_auth.py:272` 的 `_persist_env_refresh_token`:上游轮换 refresh_token 时会**回写 `.env`**。
容器内会写到 `/app/.env`(容器内部副本),随容器销毁丢失,而进程环境变量仍是旧 token,
下次重启可能认证失败,症状是「token 莫名其妙过期」。

非必要不修 —— `.env.example` 默认 `GLM_USE_GUEST_REFRESH_TOKEN=false` 且正常回退到环境变量,
一直运行正常。若将来出现上述症状,是这里。

#### 协议层的硬边界(不是缺陷,是缺失)

- **1M 上下文不可能走这条路**:上游请求体没有 token 上限字段,仓库无截断与 token 计数,`usage` 是硬编码的。
  这是协议层缺失,不是没实现。
- **上游原生工具调用不存在**:`SERVER_SIDE_TOOL_NAMES` 只能为空;上游 `tool_calls` 内容块只承载 GLM 自家的
  `browser` 搜索工具。三方独立实现加 ChatALL 客户端均确认。
- **模型名是装饰性的**:实测 `glm-flash`、`glm-totally-made-up-xyz` 与 `glm-5.3` 返回完全一致
  (上游请求体不含 model 字段)。`/v1/models` 的清单由本地常量决定,改它只是改招牌。

---

### 12.3 新支持的能力

| 能力 | 说明 |
|---|---|
| glm-5.3 及 82 个模型变体 | 含 `-think` / `-search` 组合后缀 |
| 工具调用可用 | 0% → 93~100%(真实上游) |
| JSON 漂移恢复 | ```` ```json ```` / `<tool_call>` / `Bash({...})` 形态回收 |
| 截断恢复 | `max_tokens` 截断后仍产出可用调用 |
| 逐字载荷保真 | 换行、制表符、协议标签文本原样往返 |
| 空回合消除 | 永不出现既无文本也无调用的回合 |
| 大参数线性流式 | 400KB 参数 0.011s |
| 私有代码安全更新 | bundle + 仅覆盖追踪文件的合并脚本 |
| 可靠性可测量 | agent 循环探针,含失败模式分类与 `--mock` 自检 |

---

### 12.4 当前测试状态

```
单元测试        66 passed
e2e(真实 HTTP 链路)  18 passed, 0 failed
缺陷复现套件     20 passed(基线时 18 项失败)
探针 --mock 自检  3/3
```

`repro_defects.py` 覆盖 8 组缺陷:内容保真、参数内协议标签、空容器、截断、JSON 漂移、
未声明工具、结果转义、吞吐。
