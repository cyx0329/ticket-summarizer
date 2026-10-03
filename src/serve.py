"""Serve the fine-tuned model with vLLM (OpenAI-compatible) and run a
small latency/throughput benchmark.

Usage:
    # first merge the LoRA adapter into the base model weights:
    python - <<'EOF'
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    m = PeftModel.from_pretrained(base, "outputs/adapter").merge_and_unload()
    m.save_pretrained("outputs/merged"); tok.save_pretrained("outputs/merged")
    EOF

    python src/serve.py --model outputs/merged --port 8000
"""
import argparse
import subprocess
import sys
import time

import requests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/merged")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--benchmark-requests", type=int, default=20)
    args = ap.parse_args()

    server = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", args.model, "--port", str(args.port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        url = f"http://localhost:{args.port}/v1/completions"
        for _ in range(120):  # wait for readiness
            try:
                requests.get(f"http://localhost:{args.port}/health",
                             timeout=2)
                break
            except requests.ConnectionError:
                time.sleep(5)
        else:
            raise RuntimeError("vLLM server did not become ready")

        prompt = ("Summarize as a customer notification: "
                  "[INC-12345] payments-api 5xx spike in us-west-2, "
                  "mitigated via traffic shift, lasted 35 minutes.")
        latencies = []
        start = time.time()
        for _ in range(args.benchmark_requests):
            t0 = time.time()
            r = requests.post(url, json={"prompt": prompt,
                                        "max_tokens": 128}, timeout=120)
            r.raise_for_status()
            latencies.append(time.time() - t0)
        total = time.time() - start
        latencies.sort()
        print(f"requests: {len(latencies)}  "
              f"throughput: {len(latencies) / total:.1f} req/s")
        print(f"p50 latency: {latencies[len(latencies)//2]*1000:.0f} ms  "
              f"p95 latency: {latencies[int(len(latencies)*0.95)]*1000:.0f} ms")
        print(f"sample output: {r.json()['choices'][0]['text'][:200]}")
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
