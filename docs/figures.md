# Diagrams — mini-sglang-metax

Mermaid source for all architecture and flow diagrams used in docs and README.

---

## 1. System Architecture

```mermaid
graph TD
    User([User / Client])

    subgraph "mini-sglang process group"
        API[API Server\nFastAPI / OpenAI-compat]
        TOK[Tokenizer Worker]
        DETOK[Detokenizer Worker]

        subgraph "TP Workers  ×N  via NCCL/MCCL"
            SCH0[Scheduler\nRank 0  master]
            SCH1[Scheduler\nRank 1..N-1]
            ENG0[Engine\nGPU 0]
            ENG1[Engine\nGPU 1..N-1]
        end
    end

    User -->|HTTP POST /v1/chat/completions| API
    API -->|ZMQ| TOK
    TOK -->|tokens| SCH0
    SCH0 <-->|NCCL/MCCL broadcast| SCH1
    SCH0 --> ENG0
    SCH1 --> ENG1
    ENG0 -->|output token| SCH0
    SCH0 -->|ZMQ| DETOK
    DETOK -->|text| API
    API -->|SSE stream| User
```

---

## 2. platform / device_type Decoupling

```mermaid
flowchart LR
    subgraph Startup
        D[get_accelerator_platform\nMACa_PATH / torch.__version__]
        D -->|MetaX detected| P_MX[platform = metax\ndevice_type = cuda]
        D -->|NVIDIA detected| P_NV[platform = nvidia\ndevice_type = cuda]
    end

    subgraph "Kernel routing  per operator"
        P_MX -->|attention| TN[torch_native\neager matmul]
        P_MX -->|MoE FFN| MM[MetaxMoe\npure PyTorch]
        P_MX -->|TP comm| MC[MCCL\ntorch.distributed]
        P_MX -->|KV store| IC[index_copy_]

        P_NV -->|attention| FA[FlashAttention\nFlashInfer]
        P_NV -->|MoE FFN| FK[sgl_kernel\nfused_moe CUDA]
        P_NV -->|TP comm| NC[NCCL / PyNCCL]
        P_NV -->|KV store| SK[store_cache\nCUDA kernel]
    end
```

---

## 3. MetaxMoe M1 — Expert Routing Flow

```mermaid
flowchart TD
    IN["hidden_states\n[T, hidden_dim]"]
    IN --> GATE["Router\n(Linear gate)"]
    GATE --> SF["softmax + topk\n→ topk_weights, topk_ids\n[T, topk]"]

    SF --> LOOP

    subgraph LOOP["for e in range(E)"]
        MASK["token_mask = topk_ids == e\n→ select tokens routed to expert e"]
        GATHER["x = hidden_states[token_mask]"]
        W1["gate_up = x @ w1[e].T\ngate, up = chunk(2)"]
        ACT["x_mid = silu(gate) × up"]
        W2["x_out = x_mid @ w2[e].T"]
        WAcc["weighted accumulate\noutput[token_mask] += x_out × weight"]
        MASK --> GATHER --> W1 --> ACT --> W2 --> WAcc
    end

    LOOP --> OUT["output\n[T, hidden_dim]"]
```

---

## 4. KV Cache: Naive vs Radix

```mermaid
flowchart LR
    subgraph "Naive Cache"
        N1["Req A: [SYS][User1]"] -->|allocate fresh| NA["KV pages A"]
        N2["Req B: [SYS][User2]"] -->|allocate fresh| NB["KV pages B\n(SYS recomputed!)"]
    end

    subgraph "Radix Cache (RadixAttention)"
        R1["Req A: [SYS][User1]"] -->|prefix match| RS["[SYS] node\n(cached ✓)"]
        RS --> RA["[User1] leaf"]
        R2["Req B: [SYS][User2]"] -->|prefix match| RS
        RS --> RB["[User2] leaf"]
    end
```

---

## 5. Chunked Prefill vs Naive

```mermaid
gantt
    title GPU Timeline — Prefill 32K tokens + 4 Decode steps
    dateFormat X
    axisFormat %s

    section Naive
    Prefill 32K          :p1, 0, 8
    Decode 1             :d1, 8, 9
    Decode 2             :d2, 9, 10
    Decode 3             :d3, 10, 11
    Decode 4             :d4, 11, 12

    section Chunked Prefill (chunk=8K)
    Prefill chunk 1      :c1, 0, 2
    Decode interleave 1  :e1, 2, 3
    Prefill chunk 2      :c2, 3, 5
    Decode interleave 2  :e2, 5, 6
    Prefill chunk 3      :c3, 6, 8
    Decode interleave 3  :e3, 8, 9
    Prefill chunk 4      :c4, 9, 11
    Decode interleave 4  :e4, 11, 12
```

---

## 6. Overlap Scheduling

```mermaid
sequenceDiagram
    participant CPU as CPU Scheduler
    participant GPU as GPU Worker

    Note over CPU,GPU: Traditional (no overlap)
    GPU->>CPU: batch N done
    CPU->>CPU: schedule batch N+1 (Radix lookup, etc.)
    CPU->>GPU: launch batch N+1

    Note over CPU,GPU: SGLang Overlap Scheduling
    GPU->>GPU: computing batch N
    CPU->>CPU: scheduling batch N+1 in parallel
    GPU->>CPU: batch N done (N+1 already ready)
    CPU->>GPU: launch batch N+1 immediately
```

---

## 7. Benchmark: Throughput vs Concurrency（实测）

> Qwen3-30B-A3B.w8a8 | 8×MetaX C500 | SGLang 0.5.13+maca3.8.1.0

```mermaid
xychart-beta
    title "Output Throughput vs Concurrency"
    x-axis "Concurrency" [1, 4, 8, 16, 32]
    y-axis "Throughput (tok/s)" 0 --> 360
    line [10.8, 42.3, 77.6, 115.7, 0]
    line [11.6, 43.4, 82.2, 173.0, 0]
    line [11.3, 41.8, 84.6, 171.7, 336.9]
```

*注：prefill-heavy 和 decode-heavy 只测到 conc=16（混合场景则测到 conc=32）*

### 数据表（ASCII 可视化）

```
           Output Throughput (tok/s) — 实测
                                              Prefill-heavy (in=1024,out=16)  ■
 337 ┤                            ░░░          Decode-heavy  (in=64, out=256)  ▒
     │                           ░░░░░         Mixed PD      (in=512, out=128) ░
 250 ┤                          ░░░░░░░
     │
 173 ┤                    ▒▒▒░░░
 172 ┤                   ▒▒▒▒░░░░
     │
 116 ┤             ■■■░░░
  85 ┤          ■■■▒▒▒░░░
  78 ┤         ■■■▒▒▒░░░░
     │
  43 ┤    ■▒░░░
  42 ┤    ■▒▒░░
  11 ┤ ■▒░
     └──────────────────────────────────
        1    4    8    16           32
```
