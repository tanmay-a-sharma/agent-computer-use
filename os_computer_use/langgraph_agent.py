from typing import Dict, TypedDict, Annotated, Sequence
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from os_computer_use.sandbox_agent import SandboxAgent
from os_computer_use.streaming import Sandbox

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    sandbox: Annotated[SandboxAgent, "The sandbox agent for computer interaction"]
    next: Annotated[str, "The next node to route to"]
    agent_scratchpad: Annotated[Sequence[BaseMessage], "The agent's scratchpad"]

class TypeTextInput(BaseModel):
    text: str

class ClickInput(BaseModel):
    query: str

class SendKeyInput(BaseModel):
    name: str

class RunCommandInput(BaseModel):
    command: str

class OpenAppInput(BaseModel):
    app_name: str

class NavigateToUrlInput(BaseModel):
    url: str

def create_tools(sandbox_agent: SandboxAgent) -> list[Tool]:
    """Create tools that wrap the sandbox agent's capabilities"""
    tools = []
    
    # Type text tool
    tools.append(
        Tool(
            name="type_text",
            description="Type text into the current focused element",
            func=lambda text: sandbox_agent.type_text(text),
            args_schema=TypeTextInput
        )
    )
    
    # Click tool
    tools.append(
        Tool(
            name="click",
            description="Click on an element matching the given query",
            func=lambda query: sandbox_agent.click(query),
            args_schema=ClickInput
        )
    )
    
    # Send key tool
    tools.append(
        Tool(
            name="send_key",
            description="Send a keyboard key press",
            func=lambda name: sandbox_agent.send_key(name),
            args_schema=SendKeyInput
        )
    )
    
    # Run command tool
    tools.append(
        Tool(
            name="run_command",
            description="Run a shell command",
            func=lambda command: sandbox_agent.run_command(command),
            args_schema=RunCommandInput
        )
    )
    
    # Open app tool
    tools.append(
        Tool(
            name="open_app",
            description="Open an application",
            func=lambda app_name: sandbox_agent.open_app(app_name),
            args_schema=OpenAppInput
        )
    )

    # Navigate to URL tool
    tools.append(
        Tool(
            name="navigate_to_url",
            description="Navigate to a URL in Chrome",
            func=lambda url: sandbox_agent.navigate_to_url(url),
            args_schema=NavigateToUrlInput
        )
    )
    
    return tools

def create_agent(model: str = "gpt-4-1106-preview"):
    """Create a LangGraph agent that can interact with the computer"""
    
    # Initialize sandbox and agent
    sandbox = Sandbox()
    sandbox_agent = SandboxAgent(sandbox)
    
    # Create tools
    tools = create_tools(sandbox_agent)
    
    # Create LLM
    llm = ChatOpenAI(
        model=model,
        temperature=0,
        streaming=True
    )
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant that can control a computer through various tools.
        Your goal is to help users accomplish tasks by interacting with the computer interface.
        You have access to tools that let you type, click, send keystrokes, and run commands.
        Always think step by step and use the appropriate tool for each action.
        If you're unsure about an element's location, try to describe it as precisely as possible."""),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    
    # Define the processing function
    def process_message(state: AgentState) -> AgentState:
        """Process a message in the graph"""
        messages = state["messages"]
        if not messages:
            return state
        
        # Get the last message
        last_message = messages[-1]
        
        # If it's a human message, run it through the agent
        if isinstance(last_message, HumanMessage):
            # Run the agent
            response = agent_executor.invoke({
                "messages": messages[:-1],
                "input": last_message.content,
                "agent_scratchpad": state.get("agent_scratchpad", [])
            })
            
            # Add the AI's response to messages
            new_messages = list(messages)
            new_messages.append(AIMessage(content=response["output"]))
            
            return {
                **state,
                "messages": new_messages,
                "next": "end"
            }
        
        return {**state, "next": "end"}
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add the nodes
    workflow.add_node("process", process_message)
    workflow.add_node("end", lambda x: x)  # End node just returns the state
    
    # Add the edges
    workflow.set_entry_point("process")
    workflow.add_edge("process", "end")
    
    # Compile the graph
    app = workflow.compile()
    
    return app, sandbox_agent

def run_agent(input_text: str, app: Graph, state: AgentState) -> AgentState:
    """Run the agent with a given input"""
    # Add the input message to the state
    messages = list(state["messages"])
    messages.append(HumanMessage(content=input_text))
    new_state = {**state, "messages": messages}
    
    # Run the graph
    result = app.invoke(new_state)
    
    return result
