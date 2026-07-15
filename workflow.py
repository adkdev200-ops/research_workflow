from langchain_ollama import ChatOllama
from typing  import Annotated, TypedDict
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessageChunk
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.types import Send
import operator
from langchain_tavily import TavilySearch
from pathlib import Path
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

def generate_workflow(num_results):
    CONTEXT_GATHERER_SYSTEM_PROMPT = None
    SYNTHESIZER_SYSTEM_PROMPT = None
    PLANNER_SYSTEM_PROMPT = None
    WORKER_SYSTEM_PROMPT = None


    with open('context_gatherer_system_prompt.txt', "r") as f:
        CONTEXT_GATHERER_SYSTEM_PROMPT = f.read()

    with open('synthesizer_system_prompt.txt', "r") as f:
        SYNTHESIZER_SYSTEM_PROMPT = f.read()

    with open('planner_system_prompt.txt', "r") as f:
        PLANNER_SYSTEM_PROMPT = f.read()

    with open('worker_system_prompt.txt', "r") as f:
        WORKER_SYSTEM_PROMPT = f.read()  
        

    model = ChatOllama(model = 'minimax-m3:cloud')
    search_tool = TavilySearch(max_results= num_results)



    class GeneralResearcherSchema(BaseModel):
        queries : list[str]

        
    class Task(BaseModel):
        id : int
        title : str
        description : str
        search_queries : list[str]

    class PlannerSchema(BaseModel):
        plans : list[Task]

    class ResearchState(TypedDict):
        title : str
        initial_search_queries : str
        planner_input : str
        plans : list[Task]
        sections : Annotated[list[str], operator.add]



    generalresearcher  = model.with_structured_output(GeneralResearcherSchema)

    def context_gatherer(state):

        out = generalresearcher.invoke([SystemMessage(content = CONTEXT_GATHERER_SYSTEM_PROMPT), HumanMessage(content= state['title'])])
        return {'initial_search_queries': out.queries}




    def synthesizer(state):
        results = []

        for item in state['initial_search_queries']:
            results.append(search_tool.invoke(item))
        clean_results = []
        for result_group in results:
            for r in result_group:
                if isinstance(r, dict) and 'content' in r:
                    clean_results.append(r['content'])
                elif isinstance(r, str):
                    clean_results.append(r)
        combined_text = '\n---\n'.join(clean_results)

        sythesizer_output = model.invoke([SystemMessage(content = SYNTHESIZER_SYSTEM_PROMPT), HumanMessage(content = f"""RESEARCH TOPIC: {state['title']}

        SEARCH RESULTS:
        {combined_text}""")])

        return{'planner_input' : sythesizer_output.content}


    planner_model = model.with_structured_output(PlannerSchema)

    def planner(state):

        response = planner_model.invoke([SystemMessage(content = PLANNER_SYSTEM_PROMPT), HumanMessage(content = f"""RESEARCH TOPIC: {state['title']}

        SYNTHESIZED CONTEXT:
        {state['planner_input']}""")])

        return {'plans' : response.plans}

    def fanout(state):
        all_plans = state['plans']
        return [Send('worker', {'topic' : state['title'], 'id' : task.id, 'title': task.title, 'description' : task.description, 'queries' : task.search_queries, 'plans' : all_plans}) for task in all_plans]

    def worker(payload : dict):
        idx = payload['id']
        topic = payload['topic']
        title = payload['title']
        desc = payload['description']
        queries = payload['queries']
        plans = payload['plans']

        raw_results = []
        for query in queries:
            raw_results.extend(search_tool.invoke(query))

        evidence = []
        for r in raw_results:
            if isinstance(r, dict) and 'content' in r:
                evidence.append(r['content'])
            elif isinstance(r, str):
                evidence.append(r)
        evidence_text = '\n---\n'.join(evidence)

        plan_summary = '\n'.join([f"Section {p.id}: {p.title}" for p in plans])

        response = model.invoke([SystemMessage(content = WORKER_SYSTEM_PROMPT), HumanMessage(content = f"""RESEARCH TOPIC: {topic}

    YOUR ASSIGNED SECTION: {title}
    SECTION DESCRIPTION: {desc}

    FULL REPORT PLAN (for context only — write ONLY your assigned section):
    {plan_summary}

    SEARCH EVIDENCE:
    {evidence_text}""")])

        return {'sections' : [{'idx' : idx ,'content' : response.content}]}


    def reducer(state):
        title = state['title']

        ordered_sections = sorted(
            state['sections'],
            key=lambda x: x['idx']
        )

        body = "\n\n".join(
            section['content']
            for section in ordered_sections
        )

        final_md = f"# {title}\n\n{body}\n"

        file_name = title.lower().replace(" ", "_") + ".md"
        output_path = Path(file_name)
        output_path.write_text(final_md, encoding="utf-8")

        return {"final": final_md}

    graph = StateGraph(ResearchState)
    graph.add_node('context_gatherer', context_gatherer)
    graph.add_node('synthesizer', synthesizer)
    graph.add_node('planner', planner)
    graph.add_node('worker', worker)
    graph.add_node('reducer', reducer)



    graph.add_edge(START, 'context_gatherer')
    graph.add_edge('context_gatherer', 'synthesizer')
    graph.add_edge('synthesizer', 'planner')
    graph.add_conditional_edges('planner', fanout, ['worker'])
    graph.add_edge('worker', 'reducer')
    graph.add_edge('reducer',END )

    checkpointer = InMemorySaver()

    research_model = graph.compile(checkpointer= checkpointer)
    return research_model




