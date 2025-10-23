#%%
import os
import time
import random
from openai import AzureOpenAI  # use the Azure client from the v1.x SDK

MAX_RETRY = 5

def chat_gpt(
        user_prompt=None, 
        system_prompt=None, 
        n_used=1,
        logprobs=False,      # kept for API compatibility; not used by Azure Chat Completions
        seed=None,
        llm_name=None,       # kept for compatibility
        engine_used='GPT5-mini'
    ):
    """
    Calls Azure OpenAI Chat Completions using deployment names selected by `engine_used`.
    All secrets and endpoints are read from environment variables.
    """

    # --- Read configuration from environment variables ---
    # Prefer Azure-standard names; fall back to your original names if present.
    API_VERSION = (
        os.getenv("AZURE_OPENAI_API_VERSION")
        or os.getenv("API_VERSION")
        or "2024-02-15-preview"
    )
    API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("API_BASE")
    API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("API_KEY")

    if not API_BASE or not API_KEY:
        raise EnvironmentError(
            "Missing Azure OpenAI config. Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY "
            "(optionally AZURE_OPENAI_API_VERSION)."
        )

    # Allow overriding deployment names via env; default to your prior values.
    DEPLOYMENT_GPT5       = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT5", "gpt-5-na")
    DEPLOYMENT_GPT5_MINI  = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT5_MINI", "gpt-5-mini-na")
    DEPLOYMENT_DEFAULT    = os.getenv("AZURE_OPENAI_DEPLOYMENT_DEFAULT", DEPLOYMENT_GPT5)

    if engine_used == 'GPT5':
        mdl = DEPLOYMENT_GPT5
    elif engine_used == 'GPT5-mini':
        mdl = DEPLOYMENT_GPT5_MINI
    else:
        mdl = DEPLOYMENT_DEFAULT

    client = AzureOpenAI(
        azure_endpoint=API_BASE,
        api_key=API_KEY,
        api_version=API_VERSION,
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})

    # Optional extras
    kwargs = {}
    # If you ever want to toggle JSON mode from the call site, add json_mode=True to the signature.
    # Keeping the behavior identical to your original (json_mode=False by default).
    # if json_mode:
    #     kwargs["response_format"] = {"type": "json_object"}
    if seed is not None:
        kwargs["seed"] = seed

    # --- simple retry with exponential backoff ---
    last_err = None
    for attempt in range(MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=mdl,  # this is the *deployment name* in Azure
                messages=messages if messages else [{"role": "user", "content": user_prompt or system_prompt}],
                reasoning_effort="minimal",
                n=n_used,
                **kwargs
            )
            # basic sanity check like your original
            _ = response.choices[0].message.content  # will raise if missing
            return response
        except Exception as e:
            last_err = e
            # jittered backoff
            sleep_time = min(2.5, 1.0 + random.random()) * (1.7 ** attempt)
            time.sleep(sleep_time)

    # If we got here, all retries failed
    raise RuntimeError(f"chat_gpt failed after {MAX_RETRY} attempts. Last error: {last_err}")
