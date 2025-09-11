.PHONY: init activate
init:
	uv init
	uv venv

activate:
	@echo "Run this command in your shell:"
	@echo "source .venv/bin/activate"