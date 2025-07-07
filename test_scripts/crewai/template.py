import os
import yaml
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from structured_outputs.template_agent import Agent1Output

llm = LLM(
    model="openai/o3-mini",
    temperature=0.4,
    # max_tokens=550,
    top_p=0.9,
    frequency_penalty=0.5,
    presence_penalty=0.5,
    stop=["END"],
    # seed=42
)

def load_prompts():
    try:
        with open('prompts/template_agent.yaml', 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise Exception("prompts.yaml file not found")
    except yaml.YAMLError as e:
        raise Exception(f"Error parsing YAML file: {e}")

def template_agent(input):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    prompts = load_prompts()

    agent1 = Agent(
        role=prompts['agents']['agent1']['role'],
        goal=prompts['agents']['agent1']['goal'].format(input=input),
        backstory=prompts['agents']['agent1']['backstory'],
        verbose=True,
        allow_delegation=False,
        tools=[],
        llm=llm
    )

    task1 = Task(
        description=prompts['agents']['agent1']['description'],
        agent=agent1,
        expected_output=prompts['agents']['agent1']['expected_output'],
        output_pydantic=Agent1Output
    )

    crew = Crew(
        agents=[agent1],
        tasks=[task1],
    )

    results = crew.kickoff()
    agent1 = results.tasks_output[0].pydantic

    # Write agent1 output to file
    with open('Outputs/crewai/template_agent.json', 'w') as f:
        f.write(str((crew.usage_metrics)))
        f.write(agent1.model_dump_json(indent=2))

    return agent1