# Security scope

- This is a **closed-world benchmark**. All inspect calls are served by local `query_oracle`; only the explicit OpenAI model invocation uses the internet.
- No real Hugging Face endpoints, external networking tools, unauthorized system access, shell tools, or exploit tooling are exposed to the model.
- Store API credentials in `.env` / environment variables only. `.env` and `.env.*` except `.env.example` are ignored by Git. No actual key is included.
- Never push the sealed test set before evaluating the model: `data/private/` is excluded by `.gitignore`. The report includes dataset hash rather than world contents.
- Avoid putting sensitive real-world facts into prompts; synthetic random XOR relations are provided instead.
- Budget guardrails: `--backend mock` default (offline); live use requires explicit `--backend openai`; default 8 tasks; configurable `--max-api-calls`.
- Errors or invalid query IDs must not create unrestricted fallback tools or silently substitute ground-truth answers.
- The API call uses `store=False` to disable storage for this API request where supported. Review your organization settings and OpenAI data policies separately.
