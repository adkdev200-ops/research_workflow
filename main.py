from workflow import generate_workflow

results = int(input("Enter number of searches the agent can do on a topic:\n"))
topic = input("Enter the topic of research:")

config = {'configurable' : {'thread_id' : "thread_1"}} 

research_model  = generate_workflow(num_results= results)

for item in research_model.stream({'title' : topic} , config= config, stream_mode= 'updates'):
    print(item)

print(f"Research saved as {topic}.md")