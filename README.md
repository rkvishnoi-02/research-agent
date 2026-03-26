# Research Agent

LangGraph-first market research agent that enforces:

- emotional, decision-rich query generation
- verbatim quote capture with context snippets
- a hard human-test for quote quality
- validation against paraphrased or generic evidence

## Install

```bash
pip install -e .
```

## Environment

For live web runs, set:

```bash
export TAVILY_API_KEY="..."
export FIRECRAWL_API_KEY="..."
```

Optional:

```bash
export APIFY_API_TOKEN="..."
```

If `APIFY_API_TOKEN` is missing or invalid, the agent falls back to direct Reddit discovery instead of stopping.

## Run

```bash
research-agent-run \
  --request "Research raw buyer frustration and switching language for a sales engagement platform." \
  --product "Synvo" \
  --audience "SDR teams at mid-market SaaS companies" \
  --industry "B2B SaaS" \
  --company-size "100-500 employees" \
  --mode live
```

`live` mode is tuned for public-web evidence and aims to return a usable packet.

`strict` mode keeps the full architecture thresholds and will escalate more often when evidence gaps remain.

## Colab

```python
!git clone https://github.com/rkvishnoi-02/research-agent.git
%cd /content/research-agent
!pip install -e .
```

Then set your keys and run:

```python
import os
os.environ["TAVILY_API_KEY"] = "..."
os.environ["FIRECRAWL_API_KEY"] = "..."
```

```python
!research-agent-run --request "Research raw buyer frustration and switching language for a sales engagement platform." --product "Synvo" --audience "SDR teams at mid-market SaaS companies" --industry "B2B SaaS" --company-size "100-500 employees" --mode live
```

## Tests

```bash
python -m pytest
```
