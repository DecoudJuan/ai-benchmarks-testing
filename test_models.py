import asyncio
import os
import sys

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

import litellm
litellm.set_verbose = False
litellm.suppress_debug_info = True

MODELS = {
    "claude-haiku":   "openrouter/anthropic/claude-3.5-haiku",
    "claude-sonnet":  "openrouter/anthropic/claude-sonnet-4-5",
    "claude-opus":    "openrouter/anthropic/claude-opus-4-5",
    "gpt-4o-mini":    "openai/gpt-4o-mini",
    "gpt-4o":         "openai/gpt-4o",
    "llama-3.1-8b":   "openrouter/meta-llama/llama-3.1-8b-instruct",
    "llama-3.1-70b":  "openrouter/meta-llama/llama-3.1-70b-instruct",
    "llama-3.3-70b":  "openrouter/meta-llama/llama-3.3-70b-instruct",
    "gemini-flash":   "openrouter/google/gemini-2.0-flash-001",
    "gemini-pro":     "openrouter/google/gemini-2.5-pro-preview-03-25",
    "mistral-nemo":   "openrouter/mistralai/mistral-nemo",
    "mixtral-8x7b":   "openrouter/mistralai/mixtral-8x7b-instruct",
    "qwen-2.5-72b":   "openrouter/qwen/qwen-2.5-72b-instruct:nitro",
    "deepseek-r1":    "openrouter/deepseek/deepseek-r1",
    "deepseek-v3":    "openrouter/deepseek/deepseek-chat-v3-0324",
    "phi-4":          "openrouter/microsoft/phi-4",
}

MSG = [{"role": "user", "content": "Reply with only the letter A."}]


async def test(key, model_id):
    try:
        resp = await litellm.acompletion(model=model_id, messages=MSG, max_tokens=5, temperature=0)
        out = (resp.choices[0].message.content or "").strip()
        return key, True, out
    except Exception as e:
        return key, False, str(e)[:120]


async def main():
    print("Testing all models...\n")
    tasks = [test(k, v) for k, v in MODELS.items()]
    results = await asyncio.gather(*tasks)

    ok = [r for r in results if r[1]]
    fail = [r for r in results if not r[1]]

    print(f"{'Model':<20} {'Status':<6}  {'Output / Error'}")
    print("-" * 90)
    for key, status, msg in results:
        icon = "OK  " if status else "FAIL"
        print(f"{key:<20} {icon}   {msg}")

    print(f"\n{len(ok)}/{len(results)} models working")
    if fail:
        print(f"\nFailing: {', '.join(r[0] for r in fail)}")


if __name__ == "__main__":
    asyncio.run(main())
