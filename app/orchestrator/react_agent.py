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

from app.config import get_settings
from app.orchestrator.react_context import OrchestrationContext
from app.orchestrator.react_state import OrchestrationState, OrchestrationInput
from app.orchestrator.react_tools import ORCHESTRATOR_TOOLS
from app.signals import Signal as BaseSignal


logger = logging.getLogger(__name__)


def load_chat_model(model_name: str) -> ChatOpenAI:
    """
    Load the chat model for orchestration decisions.
    
    Args:
        model_name: Model name in format "provider/model" (e.g., "openai/gpt-5-mini")
        
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
    
    # Get settings to access the API key
    settings = get_settings()
    
    return ChatOpenAI(
        model=model,
        api_key=settings.llm_api_key,
        temperature=1.0,  # GPT-5 only supports default temperature value of 1.0
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
        try:
            logger.info(f"System prompt before formatting: {repr(runtime.context.system_prompt[:200])}...")
            system_message = runtime.context.system_prompt.format(
                system_time=datetime.now(tz=UTC).isoformat()
            )
            logger.info(f"System prompt formatted successfully")
        except Exception as format_error:
            logger.error(f"Error formatting system prompt: {format_error}")
            logger.error(f"System prompt full content: {repr(runtime.context.system_prompt)}")
            # Try to identify the problematic format code
            import re
            matches = re.findall(r'{[^}]*}', runtime.context.system_prompt)
            logger.error(f"All format placeholders found: {matches}")
            raise
        
        # If this is the first call, create the initial human message with classification data
        messages = list(state.messages) if state.messages else []
        
        if not messages and state.classification:
            # Create initial message with classification information
            try:
                logger.info(f"Creating initial message from classification: {state.classification}")
                classification_summary = _format_classification_for_prompt(
                    state.classification, 
                    state.context, 
                    state.signal
                )
                logger.info(f"Classification summary created successfully")
                initial_message = HumanMessage(content=classification_summary)
                messages = [initial_message]
                logger.info(f"Initial message created successfully")
            except Exception as classification_error:
                logger.error(f"Error creating classification message: {classification_error}")
                logger.error(f"Classification data: {state.classification}")
                logger.error(f"Context data: {state.context}")
                raise
        
        # Prepare the full conversation
        conversation = [
            {"role": "system", "content": system_message},
            *messages
        ]
        
        # Get the model's response
        try:
            logger.info(f"About to invoke model with {len(conversation)} messages")
            logger.info(f"System message length: {len(system_message)}")
            if messages:
                logger.info(f"First message content preview: {messages[0].content[:200]}...")
            response = cast(AIMessage, await model.ainvoke(conversation))
            logger.info(f"Model response received successfully")
            
            # Log detailed response information
            logger.info(f"🧠 LLM RESPONSE ANALYSIS:")
            logger.info(f"   Response ID: {getattr(response, 'id', 'no-id')}")
            logger.info(f"   Response type: {type(response).__name__}")
            if hasattr(response, 'content') and response.content:
                logger.info(f"   Content length: {len(response.content)}")
                logger.info(f"   Content preview: {repr(response.content[:200])}")
            if hasattr(response, 'tool_calls') and response.tool_calls:
                logger.info(f"   Tool calls requested: {len(response.tool_calls)}")
                for i, tool_call in enumerate(response.tool_calls):
                    tool_name = tool_call.get('name', 'unknown')
                    tool_args = tool_call.get('args', {})
                    logger.info(f"     Tool call {i+1}: {tool_name}")
                    logger.info(f"       Arguments: {tool_args}")
            else:
                logger.info(f"   Tool calls: NONE (agent decided no action needed)")
                
        except Exception as model_error:
            logger.error(f"Error during model invocation: {model_error}")
            logger.error(f"Model invocation error type: {type(model_error)}")
            if hasattr(model_error, '__traceback__'):
                import traceback
                logger.error(f"Model invocation traceback: {traceback.format_exc()}")
            raise
        
        # Handle max steps reached
        if state.is_last_step and response.tool_calls:
            logger.warning(f"Maximum reasoning steps reached. Response had {len(response.tool_calls)} tool calls that will not be executed")
            for i, tool_call in enumerate(response.tool_calls):
                logger.warning(f"  Tool call {i+1}: {tool_call.get('name', 'unknown')} - {tool_call}")
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Maximum reasoning steps reached. Taking no action for this message."
                    )
                ]
            }
        
        # Log the model response details
        logger.info(f"Model response received - has_tool_calls: {bool(response.tool_calls)}")
        if response.tool_calls:
            logger.info(f"Model wants to execute {len(response.tool_calls)} tool(s):")
            for i, tool_call in enumerate(response.tool_calls):
                tool_name = tool_call.get('name', 'unknown')
                tool_args = tool_call.get('args', {})
                logger.info(f"  Tool {i+1}: {tool_name} with args: {tool_args}")
        else:
            logger.info(f"Model response (no tools): {response.content[:200]}...")
        
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
    toxicity_raw = classification.get('toxicity', 0.0)
    
    # Handle toxicity - convert 'none' string to 0.0
    if isinstance(toxicity_raw, str):
        toxicity = 0.0 if toxicity_raw.lower() in ['none', 'null', ''] else 0.0
    else:
        toxicity = float(toxicity_raw) if toxicity_raw is not None else 0.0
    
    # Ensure confidence is numeric
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    
    # Get Discord message info - check both context and signal
    discord_msg = context.get('discord_message', {})
    
    # If not in context, get from signal
    if not discord_msg and signal:
        user_id = getattr(signal, 'user_id', 'unknown')
        channel_id = getattr(signal, 'channel_id', 'unknown') 
        message_content = getattr(signal, 'content', '')
    else:
        user_id = discord_msg.get('user_id', 'unknown')
        channel_id = discord_msg.get('channel_id', 'unknown')
        message_content = discord_msg.get('content', '')
    
    # Build the prompt using f-string to avoid format() conflicts
    try:
        logger.info("About to format prompt with f-string")
        prompt = f"""Please analyze this classified Discord message and decide if a response is needed:

## Message Classification
- **Type**: {message_type}
- **Confidence**: {confidence:.2f}
- **Sentiment**: {sentiment}  
- **Toxicity Level**: {toxicity:.2f}

## Message Context
- **User ID**: {user_id}
- **Channel ID**: {channel_id}
- **Content**: "{message_content}"

## Your Decision
Based on this classification and the aggregation trigger that brought this to your attention, should we respond to this message? If yes, what type of response is appropriate?

Use the `send_discord_response` tool if you decide a response is warranted, or explain why no response is needed."""
        logger.info("F-string prompt formatted successfully")
        return prompt
    except Exception as fstring_error:
        logger.error(f"Error in f-string formatting: {fstring_error}")
        logger.error(f"Variables: message_type={message_type}, confidence={confidence}, sentiment={sentiment}, toxicity={toxicity}")
        logger.error(f"Variables: user_id={user_id}, channel_id={channel_id}")
        logger.error(f"Variables: message_content={repr(message_content)}")
        raise


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
    
    # Log routing decision
    if not last_message.tool_calls:
        logger.info("No tool calls in response - ending orchestration workflow")
        return "__end__"
    
    # Log tool execution decision
    logger.info(f"Routing to tools node to execute {len(last_message.tool_calls)} tool call(s)")
    for i, tool_call in enumerate(last_message.tool_calls):
        tool_name = tool_call.get('name', 'unknown')
        logger.info(f"  Will execute tool {i+1}: {tool_name}")
    
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