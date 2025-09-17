"""
LangChain Tools for ReAct Agent Orchestrator

This module provides LangChain-compatible tools that wrap our existing
system functionality for use with the ReAct agent.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Type
from datetime import datetime, timezone
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.orchestrator.tools import create_discord_responder_tool, ResponseTemplate
from app.signals import Signal as BaseSignal


logger = logging.getLogger(__name__)


# Global context store for current ReAct execution
_current_execution_context: Optional[Dict[str, Any]] = None


def set_execution_context(signal: BaseSignal, context: Dict[str, Any]) -> None:
    """Set the current execution context for tools to access."""
    global _current_execution_context
    _current_execution_context = {
        'signal': signal,
        'context': context
    }


def get_execution_context() -> Optional[Dict[str, Any]]:
    """Get the current execution context."""
    return _current_execution_context


def clear_execution_context() -> None:
    """Clear the execution context after processing."""
    global _current_execution_context
    _current_execution_context = None


class DiscordResponseInput(BaseModel):
    """Input schema for Discord response tool."""
    response_content: str = Field(description="The message content to send to Discord")


class KnowledgebaseInput(BaseModel):
    """Input schema for knowledgebase query tool."""
    query: str = Field(description="The search query to find relevant knowledge")


# Simple knowledgebase - replace these strings with your actual knowledge content
KNOWLEDGEBASE_ENTRIES = [
    # Core DayZ basics
    "DayZ is a survival game where players must find food, water, and shelter while avoiding zombies and other players.",
    "DayZ servers use the Central Loot Economy (CLE), so items despawn and respawn over time; if you don’t find an item, check back later.",
    "In DayZ, always carry bandages or rags to stop bleeding from zombie attacks or player encounters.",

    # Moderation / bot behavior (retain from pilot)
    "Social credit scores are earned by helping other players and being positive in chat.",
    "Social credit scores are reduced by toxic behavior, griefing, or spam messages.",
    "Users with low social credit scores may need additional guidance and support.",
    "The bot responds to questions, complaints, and situations requiring moderation.",
    "Users can ask about game mechanics, server rules, or general help topics.",
    "Response cooldowns prevent spam and give users time to read previous responses.",
    "High toxicity messages are handled by automated moderation systems.",

    # Refer-a-Friend program
    "Refer-a-Friend: players may refer an unlimited number of friends; both accounts must have 10+ hours played on the server.",
    "Refer-a-Friend rewards: referrer receives $200,000; the referred friend receives $150,000.",
    "To claim a referral, open a support ticket in #make-a-ticket and include both players’ in-game names and Steam64 IDs (admins can help locate these).",
    "Referral board resets each wipe; after a wipe you may re-refer friends who were previously referred.",

    # Starter kit & quality-of-life
    "Each player gets one starter kit per wipe; press ‘L’ in-game to claim it while in a safe area.",
    "Windows troubleshooting: press Win+R, open %localappdata%/DayZ, and you may delete folders like ‘crash-<year>’, ‘DayZ-X64’, and ‘Script’—do NOT delete all files.",
    "If issues persist after troubleshooting, open a support ticket and an admin will assist.",

    # The Wrench item
    "Item ‘Wrench’: flips a vehicle back onto its wheels and can nudge a stuck vehicle (sometimes); available at the General Trader.",
    "Recommendation: store a Wrench in your vehicle so you can recover from flips or minor terrain/fence snags.",

    # In-game drug crafting (mod—gameplay only)
    "In-game drug-making guide (mod only, not real): see the Recipes & Crafting Notion page provided by the server.",

    # Workstations & tools (mod)
    "Most workstations and similar tools have a Kit version for easy placement; they can typically be dismantled back into a Kit using a Screwdriver.",
    "Static workstation variants cannot be moved; the non-static Pill Press cannot be dismantled but can be picked up and placed on solid surfaces.",
    "Lab Station: used for meth manufacturing; each craft carries a chance of a theatrical ‘chemical explosion’ that KOs the player for ~30s but still completes the batch.",
    "Production Table: used for packaging higher-tier products and crafting bricks.",
    "Pill Press: used for Ecstasy manufacturing per the server’s guide.",
    "Chemical Barrel (cocaine manufacturing): attach 8 coca leaves (all dried or all fresh), 1 Sulfuric Acid, 1 Gasoline Jerry Can, and use Baking Soda as the required tool.",
    "Chemical Barrel yields: dried coca leaves produce a larger output than fresh leaves.",
    "Opium recipes (Chemical Barrel): 3×10 Opium + Acetic Anhydride + Gasoline (Jerry Can as tool) = Black Tar Heroin Knots; add Baking Soda to produce Brown Heroin Piles.",
    "Production Table—brick recipes: 2×10 Brown Heroin Piles + Duct Tape = 1 Brick of Brown; 5×10 Black Tar + Duct Tape = 1 Brick of Black Tar; 2×10 Fentanyl Piles + Duct Tape = 1 Brick of Fentanyl.",
    "Grow Tent: single-slot portable cultivation; dismantle to a Kit with a Screwdriver.",
    "Grow Pot: single-slot portable cultivation; dismantle to a Kit with a Screwdriver.",
    "Drying Rack: 24 slots for cannabis bud branches and 8 slots for coca leaves; default drying time is 10 minutes; dried items drop to the ground below—stay nearby to collect.",
    "Storage Pallet: visualizes your stack of bricks via many attachment slots; provides no cargo capacity.",

    # Reporting players / rule breaks
    "To report a player or rule break, open a ticket and select ‘report a player’; do not discuss incidents publicly in chat or Discord.",
    "Set up gameplay recording per #recording-recommendation; video evidence significantly improves admin ability to help; without video, assistance may be limited.",
    "Use the suggestions channel, community discussions, or DM for feedback; persistent negativity or attempts to discourage play may result in bans.",

    # Raid alarm system
    "Raid Alarm: assemble three required components with other players to build a tower that alerts your private Discord group (pings optional).",
    "Only the player who set up the Raid Alarm can access the edit menu or dismantle the tower; the item includes an in-game setup help button.",

    # Economy, vendors, events, and world features (current state)
    "Daily rewards now include a chance to win cash (e.g., $100k or $500k).",
    "ATMs are active across the map and can be robbed for $50k–$300k; paychecks are enabled.",
    "Vendors: magazines larger than 30-round cannot be purchased; purchasable attachments are limited to Tier 1–2.",
    "Armor and helmet systems have been reworked; updated vests and helmets display clearer tier labels.",
    "A new helicopter has been added.",
    "Town stashes are widely distributed (roughly 10–20 per town) with unique loot.",
    "Gold content: gold safes/stashes are active; the Gold Vendor is available; secure container upgrades can be purchased at the Gold Vendor.",
    "Softsiding raiding is enabled.",
    "Killfeed changes: Discord killfeed removed (trial); an obscured in-game killfeed is enabled (trial).",
    "Weather & lighting: darker nights have returned; weather is closer to natural.",
    "Store robberies: cash registers can be robbed using a screwdriver or a hammer.",
    "Weapon balance updates: .357 damage reduced; 5.56×45 damage increased for more consistent PvP.",
    "KOTH events are re-added (WIP) and KOTH chest capacity is increased (+20 gun slots).",

    # Recent fixes and maintenance highlights
    "Recent maintenance included fixes to medical spawn rates, stamina, building allowance, vendor issues, NBC quest, and multiple Cement Mixer placement issues.",
    "Keys & key rooms: key rooms have been added at Hrabice and Overgrown; lootable shirts/jackets nearby can spawn keys; keys now indicate their usable location.",
    "‘Drug Runners’ activity provides a new way to make money (e.g., via odd pill-related tasks).",

    # Medical system (concise, complete lookups)
    "First Aid Kits: MFAK (8×30 HP), AFAK (6×25 HP), IFAK (4×20 HP), Grizzly (8×20 HP), Saleva (4×15 HP), CarKit (2×15 HP), Ai2 (1×10 HP).",
    "Hemostatics: EME (6 uses, ×6 faster than vanilla bandage), Calok-B (3, ×5), Army BD (2, ×3), CAT (1, ×8), Esmarch (1, ×8), B-2U (2, ×2), BD-8 (8, normal), BD-10 (10, normal), BD-12 (12, normal).",
    "Medical—Other: Golden Star (10 uses; shock-damage invulnerability 10 min), Ibuprofen (15; 8 min), Vaselin (6; 5 min), Augmentin (1; cures all diseases), Ololo (50; boosts immunity), AquaTabs (25; purifies water), Analgin (12; cold relief + shock invulnerability 3 min).",
    "Surgical & Splints: Surv12 (15 uses; fixes broken bones, removes all bleeding, +10 HP), CMS Kit (5 uses; fixes broken bones, removes all bleeding, +10 HP), Splint and AlumSplint treat leg fractures.",
    "Injectors: Morphine (3 min), Adrenaline (3 min), NorAdrenaline (morphine + adrenaline for 3 min), Zagustin (treats all bleeding), Meldonin (treats bleeding + blood recovery), Propital (1 HP/s for 120 s + analgesic), PNB-16 (1 HP/s for 120 s + +500 hydration and energy), ETG Change (4 HP/s for 30 s), Obdolbos V1 (20 HP instantly + 3 HP/s for 10 s + NorAdrenaline + +500 hydration/energy/blood), Obdolbos V2 (30 HP instantly + 5 HP/s for 10 s + NorAdrenaline + +1000 hydration/energy/blood), BoneRepair (quick bone healing), SJ-1 (stamina recovery ×1.8 for 8 min), SJ-6 (×2.5 for 10 min), SJ-9 (×3.5 for 15 min), SJ-12 (×5.0 for 15 min), Perfotoran (treats POX + removes bleeding), AHF1M (25 HP instantly + removes bleeding), Zakusil (+5000 hydration & energy), P22 (regenerates 50% of current HP + analgesic 5 min), xTG-12 (heals all diseases).",
]


class KnowledgebaseTool(BaseTool):
    """
    Simple knowledgebase tool that searches through predefined knowledge entries.
    
    This tool allows the ReAct agent to query a knowledgebase for relevant information
    to help answer user questions or provide context for responses.
    """
    
    name: str = "knowledgebase"
    description: str = """
    Query the knowledgebase for relevant information. Use this tool to find knowledge
    about game mechanics, rules, social credit system, or other topics that might
    help you provide better responses to users.
    
    Parameters:
    - query: A search term or question to find relevant knowledge entries
    """
    
    args_schema: Type[BaseModel] = KnowledgebaseInput
    
    async def _run(
        self,
        query: str,
        **kwargs: Any,
    ) -> str:
        """
        Search the knowledgebase for relevant entries.
        
        Args:
            query: Search query string
            
        Returns:
            String containing matching knowledge entries
        """
        try:
            logger.info(f"🔍 Knowledgebase query: {repr(query)}")
            
            # Simple string matching - convert query to lowercase for case-insensitive search
            query_lower = query.lower()
            matching_entries = []
            
            for entry in KNOWLEDGEBASE_ENTRIES:
                # Check if any word in the query appears in the entry
                if any(word in entry.lower() for word in query_lower.split()):
                    matching_entries.append(entry)
            
            if matching_entries:
                result = "Found relevant knowledge:\n" + "\n".join(f"• {entry}" for entry in matching_entries)
                logger.info(f"📚 Knowledgebase found {len(matching_entries)} entries for: {query}")
            else:
                result = f"No knowledge found for query: {query}"
                logger.info(f"📚 Knowledgebase found no entries for: {query}")
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Knowledgebase query failed: {str(e)}"
            logger.error(f"🔍 Knowledgebase ERROR: {error_msg}")
            return error_msg
    
    async def _arun(
        self,
        query: str,
        **kwargs: Any,
    ) -> str:
        """Async version of knowledgebase query."""
        return await self._run(query, **kwargs)


class DiscordResponseTool(BaseTool):
    """
    LangChain tool that wraps our DiscordResponderTool for the ReAct agent.
    
    This tool allows the ReAct agent to send responses to Discord channels.
    It automatically determines where to send the response based on the user's
    last message context from their profile.
    """
    
    name: str = "send_discord_response"
    description: str = """
    Send a response message to a Discord channel. Use this tool when you decide to respond 
    to a user's message. The tool automatically figures out where to send the response
    based on the user's most recent message.
    
    Parameters:
    - response_content: The content of the response message to send
    """
    
    args_schema: Type[BaseModel] = DiscordResponseInput
    
    async def _run(
        self,
        response_content: str,
        **kwargs: Any,
    ) -> str:
        """
        Send a Discord response message.
        
        This method gets the user context from the current ReAct state and automatically
        determines where to send the response based on their last message.
        
        Args:
            response_content: Message content to send
            
        Returns:
            String confirmation of the action taken
        """
        # Required imports for Discord message sending
        import asyncio
        
        try:
            logger.info(f"🤖 ReAct agent TOOL EXECUTION START: send_discord_response")
            logger.info(f"   Response content: {repr(response_content)}")
            
            # Get execution context
            exec_context = get_execution_context()
            
            if exec_context and exec_context.get('signal'):
                signal = exec_context['signal']
                signal_id = signal.signal_id
                logger.info(f"   Got signal context: {signal_id}")
                
                # Extract channel and user info from the signal context and author
                channel_id = signal.context.get('channel_id')
                user_id = signal.author.get('user_id')
                message_id = signal.context.get('message_id')
                
                if channel_id and user_id:
                    logger.info(f"   Target user: {user_id}")
                    logger.info(f"   Target channel: {channel_id}")
                    logger.info(f"   Reply to message: {message_id}")
                    
                    # Use the existing DiscordResponderTool to send the actual Discord message
                    from app.orchestrator.tools import create_discord_responder_tool, ResponseMode
                    discord_tool = create_discord_responder_tool()
                    
                    # Send the Discord message directly using the responder tool
                    result = await discord_tool.run(
                        signal=signal,
                        text=response_content,
                        mode=ResponseMode.REPLY  # Reply to the original message
                    )
                    
                    logger.info(f"📤 Discord message sent successfully")
                    logger.info(f"   Response ID: {result.get('response_id', 'unknown')}")
                    logger.info(f"   Discord API Response: {result.get('api_response', {}).get('id', 'unknown')}")
                    
                    return f"✅ Discord response sent: {response_content[:100]}..."
                    
                else:
                    error_msg = f"❌ Cannot send Discord response: missing channel_id ({channel_id}) or user_id ({user_id})"
                    logger.error(error_msg)
                    return error_msg
            else:
                error_msg = "❌ Cannot send Discord response: no execution context available"
                logger.error(error_msg)
                return error_msg
            
        except Exception as e:
            error_msg = f"❌ Failed to send Discord response: {str(e)}"
            logger.error(f"🤖 ReAct agent TOOL EXECUTION ERROR: {error_msg}")
            logger.error(f"   Exception details: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"   Full traceback: {traceback.format_exc()}")
            return error_msg
    
    async def _arun(
        self,
        response_content: str,
        **kwargs: Any,
    ) -> str:
        """Async version with comprehensive logging."""
        logger.info(f"🤖 ReAct agent ASYNC TOOL EXECUTION: send_discord_response")
        return await self._run(response_content, **kwargs)


# List of available tools for the ReAct agent
ORCHESTRATOR_TOOLS = [DiscordResponseTool(), KnowledgebaseTool()]