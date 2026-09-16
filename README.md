# Vera Bot — magicpin AI Challenge Submission

## Approach
This submission implements a robust, stateful AI merchant assistant ("Vera") using FastAPI and an LLM-powered composer. Key architectural decisions include:

1. **Two-Tier LLM Routing:** High-value triggers (e.g., `research_digest`, `perf_dip`) route to a premium model, while routine nudges (e.g., `dormant_with_vera`) route to a faster model to optimize latency and cost.
2. **Contextual Prompt Dispatch:** 22 distinct trigger-specific prompt templates ensure that messages are highly relevant, specific, and leverage the right compulsion levers (e.g., urgency for compliance, social proof for milestones).
3. **Multi-turn Conversation State:** The bot tracks conversation history, intent state, and consecutive auto-replies in memory, allowing it to adapt dynamically.
4. **Rule-based & LLM Hybrid Handlers:**
   - **Auto-reply Detection:** A robust regex-based tracker maintains a merchant-level counter, employing a 3-tier escalation strategy (send gentle nudge → wait 24h → end conversation).
   - **Hostile Handling:** Immediately terminates the conversation gracefully.
   - **Intent Transitions:** Switches from qualifying to action mode immediately upon detecting positive intent.
   - **Fallback Composer:** Ensures the bot always responds, even if the LLM API times out or returns malformed JSON.

## Tradeoffs
- **In-Memory State:** Conversation and context states are held in memory. This ensures sub-millisecond state retrieval for the 30-second timeout budget but would require migration to Redis/PostgreSQL for a true horizontally scalable production deployment.
- **Regex vs. LLM for Intent:** Auto-reply and hostile detection rely heavily on regex patterns. This is lightning-fast and deterministic but might miss nuanced, novel phrasings that a dedicated classifier LLM would catch.
- **JSON Parsing:** The LLM output relies on regex extraction of JSON. While mostly stable, complex markdown wrapping from some models requires careful stripping.

## Additional Context Needed
To further improve Vera, the following would be helpful:
- **Expanded Conversation Logs:** More examples of edge-case merchant responses (e.g., partial interest, language mixing) to refine the `respond` prompt.
- **Conversion Tracking:** Knowing which compulsion levers historically led to actual GBP updates or magicpin profile changes for specific merchant categories.
- **Deeper Category Taboos:** A more comprehensive list of regulatory and voice taboos per category to ensure 100% compliance in generated content.
