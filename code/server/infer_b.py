"""Paper B, VLM-B on server1: letter probabilities for land-use evidence cards (vLLM, LoRA adapter).

    python infer_b.py --model <dir> --input infer.jsonl --images <dir> --output out.jsonl --lora <adapter dir>

Each input line: {"id": str, "text": str, "image": "<file in --images>"}. One generated token at temperature 0 with
thinking off; the log-probabilities of the letters A-F in the top-20 are written (missing letters get -inf).
"""
import argparse
import base64
import json
import math
import time
from pathlib import Path

LETTERS = list("ABCDEF")
SYSTEM = "You are a land-use analyst who interprets satellite imagery of Vietnam."


def uri(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--headroom-gib", type=float, default=0.6, help="memory left free for the desktop and the vision encoder")
    a = ap.parse_args()
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    items = [json.loads(l) for l in open(a.input, encoding="utf-8")]
    images = Path(a.images)
    convs = [[{"role": "system", "content": SYSTEM},
              {"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri(images / it["image"])}},
                                           {"type": "text", "text": it["text"]}]}] for it in items]
    free, total = torch.cuda.mem_get_info()  # GPU 0 also drives the desktop: size vLLM to what is actually free
    gpu_mem = round(min(0.92, (free - a.headroom_gib * 2 ** 30) / total), 3)
    print(f"free {free / 2 ** 30:.1f} of {total / 2 ** 30:.1f} GiB -> gpu_memory_utilization {gpu_mem}", flush=True)
    t0 = time.time()
    llm = LLM(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=gpu_mem, max_num_seqs=a.max_num_seqs,
              max_num_batched_tokens=8192, mm_processor_kwargs={"max_pixels": 1024 * 1024},
              limit_mm_per_prompt={"image": 1, "video": 0}, enable_lora=True, max_lora_rank=64, seed=42,
              compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16]})
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20, seed=42)
    t1 = time.time()
    outs = llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False,
                    lora_request=LoRARequest("paperB", 1, a.lora))
    t2 = time.time()
    with open(a.output, "w", encoding="utf-8") as f:
        for it, o in zip(items, outs):
            c = o.outputs[0]
            lp = {L: -math.inf for L in LETTERS}
            for cand in (c.logprobs[0] if c.logprobs else {}).values():
                tok = (cand.decoded_token or "").strip()
                if tok in lp and cand.logprob > lp[tok]:
                    lp[tok] = cand.logprob
            f.write(json.dumps({"id": it["id"], "text_out": c.text, "logprobs": lp}) + "\n")
    print(json.dumps({"items": len(items), "load_s": round(t1 - t0, 1), "run_s": round(t2 - t1, 1)}))


if __name__ == "__main__":
    main()
