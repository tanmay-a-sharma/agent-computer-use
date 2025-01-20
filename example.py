import os
from os_computer_use.langgraph_agent import create_agent, AgentState, run_agent
from dotenv import load_dotenv
from os_computer_use.logging import Logger

logger = Logger()


def main():

    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    # Create the agent and get the graph app
    print("Creating agent...")
    app, sandbox_agent = create_agent()

    # Initialize the state
    initial_state = AgentState(messages=[], sandbox=sandbox_agent, next="process")

    # Example tasks for the agent
    tasks = [
        "Open Google Chrome",
        "In Chrome, navigate to google.com",
        "Type 'Python programming' in the Google search bar",
        "Press enter to search",
    ]

    # Run each task
    current_state = initial_state
    for task in tasks:
        print(f"\nExecuting task: {task}")
        current_state = run_agent(task, app, current_state)
        # Print the last message from the agent
        if current_state["messages"]:
            print(f"Agent response: {current_state['messages'][-1].content}")


if __name__ == "__main__":
    main()
