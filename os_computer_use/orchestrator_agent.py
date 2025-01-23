from typing import List, Dict, Any
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class OrchestratorAgent:
    """
    An agent responsible for planning and coordinating steps using available LangChain tools.
    This agent analyzes the initial prompt, breaks it down into subtasks, and determines
    which tools would be most appropriate for each step.
    """

    def __init__(self, model_name: str = "gpt-4-1106-preview"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.tools: List[BaseTool] = []
        self.tool_descriptions: Dict[str, str] = {}
        self.tool_analysis: Dict[str, Dict[str, Any]] = {}

    def analyze_tool(self, tool: BaseTool) -> Dict[str, Any]:
        """Analyze a LangChain tool to understand its capabilities and requirements"""
        analysis = {
            "name": tool.name,
            "description": tool.description,
            "capabilities": [],
            "requirements": [],
            "effects": [],
            "parameters": {}
        }

        # Analyze args schema if available
        if hasattr(tool, 'args_schema'):
            schema = tool.args_schema.schema()
            analysis["parameters"] = {
                field: {
                    "type": field_info.get("type", "unknown"),
                    "required": field in schema.get("required", []),
                    "description": field_info.get("description", ""),
                } for field, field_info in schema.get("properties", {}).items()
            }

        # Infer capabilities from description and name
        desc_lower = tool.description.lower()
        if "browser" in desc_lower or "chrome" in desc_lower:
            analysis["capabilities"].append("browser_interaction")
        if "type" in desc_lower or "input" in desc_lower:
            analysis["capabilities"].append("text_input")
        if "click" in desc_lower:
            analysis["capabilities"].append("mouse_interaction")
        if "read" in desc_lower or "get" in desc_lower:
            analysis["capabilities"].append("content_extraction")

        # Infer requirements and effects
        if "browser" in analysis["capabilities"]:
            analysis["requirements"].append("browser_must_be_open")
        if "text_input" in analysis["capabilities"]:
            analysis["requirements"].append("target_element_must_be_focused")

        return analysis

    def add_tool(self, tool: BaseTool) -> None:
        """Add a tool to the orchestrator's available toolset and analyze it"""
        self.tools.append(tool)
        self.tool_descriptions[tool.name] = tool.description
        self.tool_analysis[tool.name] = self.analyze_tool(tool)

    def get_tool_knowledge_prompt(self) -> str:
        """Generate a detailed prompt section about tool knowledge"""
        tool_knowledge = ["Available Tools and Their Capabilities:"]

        for name, analysis in self.tool_analysis.items():
            tool_info = [f"\n{name}:"]
            tool_info.append(f"Description: {analysis['description']}")

            if analysis['capabilities']:
                tool_info.append(f"Capabilities: {', '.join(analysis['capabilities'])}")
            if analysis['requirements']:
                tool_info.append(f"Requirements: {', '.join(analysis['requirements'])}")
            if analysis['parameters']:
                params = [f"- {param}: {info['description']} ({'required' if info['required'] else 'optional'})"
                         for param, info in analysis['parameters'].items()]
                tool_info.append("Parameters:\n" + "\n".join(params))

            tool_knowledge.append("\n".join(tool_info))

        return "\n\n".join(tool_knowledge)

    def create_planning_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for the planning phase"""
        return ChatPromptTemplate.from_messages([
            ("system", f"""You are an orchestrator agent that understands high-level user requests and breaks them down into actionable steps using available tools.

            {self.get_tool_knowledge_prompt()}

            Tool Dependencies and Requirements:
            1. Browser Interaction Chain:
               - Opening browser must happen before navigation
               - Navigation must happen before page interaction
               - Element interaction requires proper element state

            2. Input/Output Chain:
               - Reading content requires page to be loaded
               - Text input requires element focus
               - Clicking requires element to be visible

            Your Planning Process:
            1. Analyze the user's high-level goal
            2. Break down what information or actions are needed
            3. For each step:
               - Match required action with tool capabilities
               - Verify all tool requirements are met
               - Plan for potential failures
            4. Sequence steps based on tool dependencies

            Respond with a JSON structure containing:
            - steps: list of steps, each with:
                - description: what needs to be done
                - tool: name of the tool to use
                - parameters: required parameters for the tool
                - requirements: list of conditions that must be true
                - success_criteria: specific conditions that indicate success
                - fallback: what to do if the step fails
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

    def create_agent(self) -> AgentExecutor:
        """Create the LangChain agent with the planning prompt"""
        prompt = self.create_planning_prompt()
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools)

    def plan(self, task: str) -> Dict[str, Any]:
        """
        Create a plan for accomplishing the given task using available tools.
        Returns a structured plan with steps and tool assignments.
        """
        agent = self.create_agent()
        # Convert tool descriptions to a formatted string
        tool_desc_str = "\n".join(
            [f"- {name}: {desc}" for name, desc in self.tool_descriptions.items()]
        )

        result = agent.invoke(
            {"input": task, "tool_descriptions": tool_desc_str, "chat_history": []}
        )

        return result

    def execute_plan(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute a plan created by the plan() method.
        Returns the results of each step.
        """
        results = []
        for step in plan.get("steps", []):
            tool_name = step.get("tool")
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if tool:
                try:
                    # Add a small delay between actions to allow for UI updates
                    import time

                    time.sleep(1)

                    result = tool.invoke(step.get("parameters", {}))
                    success = True

                    # Check success criteria if specified
                    if "success_criteria" in step:
                        # TODO: Implement success criteria checking
                        pass

                    results.append(
                        {
                            "step": step["description"],
                            "success": success,
                            "result": result,
                        }
                    )
                except Exception as e:
                    results.append(
                        {"step": step["description"], "success": False, "error": str(e)}
                    )
            else:
                results.append(
                    {
                        "step": step["description"],
                        "success": False,
                        "error": f"Tool {tool_name} not found",
                    }
                )
        return results
