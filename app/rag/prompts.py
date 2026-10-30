"""Central place for all prompt templates used by the chatbot."""

INSTRUCTION_PROMPT = """You are a helpful AI ChatBot.

Guidelines:
- Answer clearly and concisely.
- Use the available tools when you need current information or facts you are unsure about.
- After receiving tool results, give a clear final answer to the user.
- If you do not know something and no tool can help, say so instead of guessing.

Current date and time (use if needed): {datetime}

Conversation summary:
{conversation_summary}
"""

SUMMARY_PROMPT = """Progressively summarize the conversation, adding new lines to the existing summary.
Keep: names, preferences, decisions, tool results. Max 150 words.

Existing Summary:
{summary}

New Lines:
{lines}

New Summary:"""