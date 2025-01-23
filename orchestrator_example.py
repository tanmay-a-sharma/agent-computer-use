import os
from dotenv import load_dotenv
from os_computer_use.orchestrator_agent import OrchestratorAgent
from os_computer_use.sandbox_agent import SandboxAgent
from os_computer_use.langgraph_agent import create_tools

def main():
    """
    Example showing how the OrchestratorAgent analyzes and understands LangChain tools
    to better plan and execute tasks.
    """
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    
    # Create the sandbox agent for computer interaction
    sandbox_agent = SandboxAgent()
    
    # Create the orchestrator
    orchestrator = OrchestratorAgent()
    
    # Add and analyze each tool
    tools = create_tools(sandbox_agent)
    print("\nAnalyzing available tools:")
    print("=" * 50)
    
    for tool in tools:
        print(f"\nAdding tool: {tool.name}")
        orchestrator.add_tool(tool)
        
        # Show the analysis for this tool
        analysis = orchestrator.tool_analysis[tool.name]
        print(f"Capabilities: {', '.join(analysis['capabilities'])}")
        print(f"Requirements: {', '.join(analysis['requirements'])}")
        if analysis['parameters']:
            print("Parameters:")
            for param, info in analysis['parameters'].items():
                print(f"  - {param}: {info['description']} ({'required' if info['required'] else 'optional'})")
    
    # Show how this knowledge is used in planning
    print("\nPlanning knowledge generated for tools:")
    print("=" * 50)
    print(orchestrator.get_tool_knowledge_prompt())
    
    # Example task
    task = "Get me a summary of Java programming."
    print("\nPlanning for task:", task)
    print("=" * 50)
    
    plan = orchestrator.plan(task)
    print("\nGenerated Plan:")
    for i, step in enumerate(plan.get("steps", []), 1):
        print(f"\nStep {i}:")
        print(f"Description: {step['description']}")
        print(f"Tool: {step['tool']}")
        print(f"Requirements: {step.get('requirements', [])}")
        print(f"Parameters: {step['parameters']}")
        print(f"Success Criteria: {step['success_criteria']}")
        print(f"Fallback: {step.get('fallback', 'None specified')}")

if __name__ == "__main__":
    main()
