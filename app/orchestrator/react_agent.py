"""
ReAct Agent Implementation for Discord Message Orchestration

This module implements a LangGraph ReAct agent that replaces the rule-based
orchestration logic with an AI agent that can reason about Discord messages
and decide how to respond appropriately.
"""

import logging
from datetime import UTC, datetime
from typing import Dict, List, Literal, cast, Any
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from app.orchestrator.react_context import OrchestrationContext
from app.orchestrator.react_state import OrchestrationState, OrchestrationInput
from app.orchestrator.react_tools import ORCHESTRATOR_TOOLS
from app.signals import Signal as BaseSignal


logger = logging.getLogger(__name__)


def load_chat_model(model_name: str) -> ChatOpenAI:
    """
    Load the chat model for orchestration decisions.
    
    Args:
        model_name: Model name in format "provider/model" (e.g., "openai/gpt-4o-mini")
        
    Returns:
        Configured ChatOpenAI instance
    """
    # Extract provider and model
    if "/" in model_name:
        provider, model = model_name.split("/", 1)
    else:
        provider = "openai"
        model = model_name
    
    if provider.lower() != "openai":
        logger.warning(f"Unsupported provider {provider}, defaulting to OpenAI")
    
    return ChatOpenAI(
        model=model,
        temperature=0.1,  # Low temperature for consistent decision-making
    )


async def call_orchestration_model(
    state: OrchestrationState,
    runtime: Runtime[OrchestrationContext]
) -> Dict[str, List[AIMessage]]:
    """
    Call the LLM to make orchestration decisions.
    
    This function prepares the prompt with message classification data,
    calls the model, and processes the response for orchestration decisions.
    
    Args:
        state: Current orchestration state
        runtime: Runtime context with configuration
        
    Returns:
        Dictionary containing the model's response message
    """
    try:
        # Initialize the model with tool binding
        model = load_chat_model(runtime.context.model).bind_tools(ORCHESTRATOR_TOOLS)
        
        # Format the system prompt
        system_message = runtime.context.system_prompt.format(
            system_time=datetime.now(tz=UTC).isoformat()
        )
        
        # If this is the first call, create the initial human message with classification data
        messages = list(state.messages) if state.messages else []
        
        if not messages and state.classification:
            # Create initial message with classification information
            classification_summary = _format_classification_for_prompt(
                state.classification, 
                state.context, 
                state.signal
            )
            
            initial_message = HumanMessage(content=classification_summary)
            messages = [initial_message]
        
        # Prepare the full conversation
        conversation = [
            {"role": "system", "content": system_message},
            *messages
        ]
        
        # Get the model's response
        response = cast(AIMessage, await model.ainvoke(conversation))
        
        # Handle max steps reached
        if state.is_last_step and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Maximum reasoning steps reached. Taking no action for this message."
                    )
                ]
            }
        
        # Return the model's response
        return {"messages": [response]}
        
    except Exception as e:
        logger.error(f"Error in orchestration model call: {e}")
        return {
            "messages": [
                AIMessage(
                    content=f"Error in orchestration decision: {str(e)}. Taking no action."
                )
            ]
        }


def _format_classification_for_prompt(
    classification: Dict[str, Any], 
    context: Dict[str, Any], 
    signal: BaseSignal
) -> str:
    """
    Format classification data into a human-readable prompt for the agent.
    
    Args:
        classification: Message classification results
        context: Additional message context
        signal: Original signal data
        
    Returns:
        Formatted string for the agent to analyze
    """
    # Extract key information
    message_type = classification.get('message_type', 'unknown')
    confidence = classification.get('confidence', 0.0)
    sentiment = classification.get('sentiment', 'neutral')
    toxicity = classification.get('toxicity', 0.0)
    requires_response = classification.get('requires_response', False)
    
    # Get Discord message info
    discord_msg = context.get('discord_message', {})
    user_id = discord_msg.get('user_id', 'unknown')
    channel_id = discord_msg.get('channel_id', 'unknown')
    message_content = discord_msg.get('content', '')
    
    prompt = f"""Please analyze this classified Discord message and decide if a response is needed:

## Message Classification
- **Type**: {message_type}
- **Confidence**: {confidence:.2f}
- **Sentiment**: {sentiment}  
- **Toxicity Level**: {toxicity:.2f}
- **Requires Response**: {requires_response}

## Message Context
- **User ID**: {user_id}
- **Channel ID**: {channel_id}
- **Content**: "{message_content}"

## Your Decision
Based on this classification, should we respond to this message? If yes, what type of response is appropriate?

Use the `send_discord_response` tool if you decide a response is warranted, or explain why no response is needed."""

    return prompt


def route_orchestration_output(state: OrchestrationState) -> Literal["__end__", "tools"]:
    """
    Determine the next step in the orchestration process.
    
    Args:
        state: Current orchestration state
        
    Returns:
        Next node to execute ("__end__" or "tools")
    """
    last_message = state.messages[-1] if state.messages else None
    
    if not isinstance(last_message, AIMessage):
        logger.error(f"Expected AIMessage, got {type(last_message).__name__}")
        return "__end__"
    
    # If no tool calls, we're done
    if not last_message.tool_calls:
        return "__end__"
    
    # Execute the requested tools
    return "tools"


# Build the ReAct orchestration graph
def create_orchestration_graph():
    """Create and return the LangGraph ReAct orchestration graph."""
    
    builder = StateGraph(
        OrchestrationState, 
        input_schema=OrchestrationInput,
        context_schema=OrchestrationContext
    )
    
    # Add nodes
    builder.add_node(call_orchestration_model)
    builder.add_node("tools", ToolNode(ORCHESTRATOR_TOOLS))
    
    # Set entry point
    builder.add_edge("__start__", "call_orchestration_model")
    
    # Add conditional routing
    builder.add_conditional_edges(
        "call_orchestration_model",
        route_orchestration_output
    )
    
    # Return to model after tool execution
    builder.add_edge("tools", "call_orchestration_model")
    
    # Compile the graph
    graph = builder.compile(name="Discord Orchestration Agent")
    
    return graph


# Global graph instance
_orchestration_graph = None


def get_orchestration_graph():
    """Get the global orchestration graph instance."""
    global _orchestration_graph
    if _orchestration_graph is None:
        _orchestration_graph = create_orchestration_graph()
    return _orchestration_graph