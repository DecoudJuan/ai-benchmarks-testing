import asyncio, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from dotenv import load_dotenv; load_dotenv()
import litellm; litellm.set_verbose = False; litellm.suppress_debug_info = True

CANDIDATES = {
    "sonnet-a": "openrouter/anthropic/claude-3.5-sonnet-20241022",
    "sonnet-b": "openrouter/anthropic/claude-3-5-sonnet-20241022",
    "sonnet-c": "openrouter/anthropic/claude-sonnet-4-5",
    "opus-a":   "openrouter/anthropic/claude-3-opus-20240229",
    "opus-b":   "openrouter/anthropic/claude-opus-4-5",
}
MSG = [{"role": "user", "content": "Reply with only the letter A."}]

async def test(key, mid):
    try:
        r = await litellm.acompletion(model=mid, messages=MSG, max_tokens=5, temperature=0)
        return key, True, (r.choices[0].message.content or "").strip()
    except Exception as e:
        return key, False, str(e)[:100]

async def main():
    results = await asyncio.gather(*[test(k, v) for k, v in CANDIDATES.items()])
    for k, ok, m in results:
        print(f"{'OK  ' if ok else 'FAIL'} {k:<12} {m}")

asyncio.run(main())
