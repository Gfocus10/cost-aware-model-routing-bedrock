"""models.py - one function that can call any model tier and report cost.

Everything else in the project (judge, router, eval runner) builds on call_model().
Uses the Bedrock Converse API because it has the same request/response shape for
Nova and Claude, so we don't write different code per model family.
"""
import json
import os
import time
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

load_dotenv(override=True)

ROOT = Path(__file__).resolve().parent.parent
PRICES = json.loads((ROOT / "prices.json").read_text())

TIER_ENV_VARS = {
    "cheap": "TIER_CHEAP_ID",
    "mid": "TIER_MID_ID",
    "strong": "TIER_STRONG_ID",
}

# Retries with backoff handle occasional throttling without crashing a long eval run.
_client = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-east-1"),
    config=Config(retries={"max_attempts": 5, "mode": "standard"}),
)


AUTH_ERROR_CODES = {"ExpiredTokenException", "ExpiredToken", "UnrecognizedClientException",
                    "InvalidSignatureException", "InvalidClientTokenId"}


def is_credentials_error(e: Exception) -> bool:
    """True when the failure means 'your AWS credentials are expired or invalid' (lab session ended)."""
    if isinstance(e, NoCredentialsError):
        return True
    return isinstance(e, ClientError) and e.response.get("Error", {}).get("Code") in AUTH_ERROR_CODES


CREDENTIALS_HELP = ("\nAWS credentials are expired or invalid. Restart the lab, copy the NEW access key, "
                    "secret key AND session token into .env, then run the same command again - "
                    "finished questions are kept and only the missing ones are redone.")


def model_id_for(tier: str) -> str:
    """Look up the model ID configured for a tier (cheap / mid / strong)."""
    if tier not in TIER_ENV_VARS:
        raise ValueError(f"Unknown tier '{tier}'. Use one of {list(TIER_ENV_VARS)}")
    model_id = os.getenv(TIER_ENV_VARS[tier])
    if not model_id:
        raise RuntimeError(f"{TIER_ENV_VARS[tier]} is empty. Set it in your .env file.")
    return model_id


def compute_cost(tier: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of one call, from token counts and the prices.json table."""
    p = PRICES[tier]
    return (
        input_tokens / 1_000_000 * p["input_per_million"]
        + output_tokens / 1_000_000 * p["output_per_million"]
    )


def call_model(tier: str, prompt: str, max_tokens: int = 300, system: str | None = None) -> dict:
    """Send one prompt to a tier and return text, token counts, latency and cost.

    temperature=0 keeps answers as repeatable as possible, so differences between
    tiers come from the models, not from randomness.
    """
    kwargs = {
        "modelId": model_id_for(tier),
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.0},
    }
    if system:
        kwargs["system"] = [{"text": system}]

    start = time.perf_counter()
    response = _client.converse(**kwargs)
    wall_ms = (time.perf_counter() - start) * 1000

    # The reply can contain several content blocks; keep only the text ones.
    blocks = response["output"]["message"]["content"]
    text = "".join(b["text"] for b in blocks if "text" in b).strip()

    usage = response["usage"]
    in_tok, out_tok = usage["inputTokens"], usage["outputTokens"]

    return {
        "tier": tier,
        "model_id": kwargs["modelId"],
        "text": text,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "latency_ms": response.get("metrics", {}).get("latencyMs", round(wall_ms)),
        "cost_usd": compute_cost(tier, in_tok, out_tok),
        "stop_reason": response.get("stopReason"),
    }


if __name__ == "__main__":
    # Step 4 checkpoint: call every tier once with the same prompt.
    test_prompt = "In one sentence, what is a return policy?"
    total = 0.0
    for tier in ("cheap", "mid", "strong"):
        try:
            r = call_model(tier, test_prompt, max_tokens=80)
            total += r["cost_usd"]
            print(f"\n[{tier}] {r['model_id']}")
            print(f"  answer : {r['text']}")
            print(f"  tokens : {r['input_tokens']} in / {r['output_tokens']} out")
            print(f"  latency: {r['latency_ms']} ms")
            print(f"  cost   : ${r['cost_usd']:.6f}")
        except Exception as e:  # show the real error so we can fix access/ID problems
            print(f"\n[{tier}] FAILED: {type(e).__name__}: {e}")
    print(f"\nTotal checkpoint cost: ${total:.6f}")
