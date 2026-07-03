# Deep Research Workflow

A multi-step research workflow built with LangGraph that takes any topic and produces a well-structured, citation-heavy research report — basically the kind of thing you'd want if you needed a solid briefing on something but didn't have six hours to read through everything yourself.

## What it does

You give it a topic, it gives you back a full markdown report with sections, sources, and actual structure. The pipeline runs through four stages:

1. **Context Gathering** — generates precision search queries to pull in relevant source material
2. **Synthesis** — filters and condenses raw search results into a dense factual briefing
3. **Planning** — designs the report architecture (intro, body sections, conclusion) with clear scope boundaries so sections don't overlap
4. **Writing** — multiple workers write individual sections in parallel, each grounded in fresh search evidence with inline citations

The whole thing is wired up as a LangGraph state graph with a fixed execution order. The planner fans out tasks to multiple parallel workers, and the reducer stitches everything back together in order and saves the final report as a markdown file.

## How it works

```
Topic → Context Gatherer → Synthesizer → Planner → Workers (parallel) → Reducer → .md file
```

Each step feeds its output into the next through shared state. Checkpointing is enabled so you can inspect intermediate results at any node.

## Tech stack

- **LangGraph** for workflow orchestration and state management
- **LangChain** + **ChatOllama** as the LLM interface
- **Tavily** for web search
- **Pydantic** for structured output schemas

## Setup

1. Clone the repo

```bash
git clone https://github.com/adkdev200-ops/research_workflow
cd research_workflow
```

2. Create a virtual environment and install dependencies

```bash
python -m venv myenv
source myenv/bin/activate
pip install langchain langchain-ollama langchain-community langgraph pydantic python-dotenv
```

3. Set up your `.env` file with API keys

```
TAVILY_API_KEY=your_tavily_key
```

4. Open `research_workflow.ipynb` and run the cells

The model is currently set to `minimax-m3:cloud` via Ollama — swap it out for whatever model you prefer.

## Usage

Just change the topic in the last cell:

```python
for item in research_model.stream({'title': 'Your research topic here'}, config=config, stream_mode='updates'):
    print(item)
```

The final report gets saved as a markdown file named after your topic (e.g., `your_research_topic_here.md`).

## Project structure

```
├── research_workflow.ipynb            # Main notebook — the full pipeline
├── context_gatherer_system_prompt.txt # Prompt for query generation
├── synthesizer_system_prompt.txt      # Prompt for search result synthesis
├── planner_system_prompt.txt          # Prompt for report architecture planning
├── worker_system_prompt.txt           # Prompt for section writing
├── .env                               # API keys (not tracked)
└── output/                            # Generated research reports
```

## Notes

- The system prompts are stored as separate `.txt` files so they're easy to tweak without touching the notebook
- Each worker does its own independent web search based on planner-assigned queries, so sections get fresh evidence rather than all pulling from the same initial results
- The worker prompt enforces strict evidence discipline — no hallucinated facts, no invented quotes, mandatory inline citations
- Report quality depends heavily on the LLM you're using. Stronger models = better planning and writing
