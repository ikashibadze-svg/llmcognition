.PHONY: test smoke generate audit

test:
	python -m pytest -q

smoke:
	python -m llmcognition smoke

generate:
	python -m llmcognition generate --n 100

audit:
	python -m llmcognition audit

# Deliberately NO implicit live API target; use explicit --backend openai command.
