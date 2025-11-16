Our code is built upon the GraphRAG-Bench repository. We evaluate the different LLM models
on different GraphRAG methods using the src/framework/main.py file changing the configs in src/framework/Option/merged_config.yaml.
For evaluating the different LLM models we use VLLM package command to deploy a local LLM server. The baselines for each llm model is evaluated
using the baseline_evaluate.py.

