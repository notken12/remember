# remember

Relive, recollect, remember.

## CLI Usage

The Remember CLI provides the same functionality as the MentraOS app, allowing you to interact with the memory assistant through different session types.

### Prerequisites

1. Set the `BACKEND_URL` environment variable to your backend server URL
2. Optionally set `WAKE_WORD` (defaults to "jarvis")

```bash
export BACKEND_URL="http://localhost:8000"
export WAKE_WORD="jarvis"
```

### Running the CLI

```bash
# Show help
bun run src/cli.ts help

# Start an ESI (Episodic Structured Interview) session
bun run src/cli.ts esi start

# Chat with the ESI session
bun run src/cli.ts esi chat "Tell me about your childhood"

# Chat directly with the assistant
bun run src/cli.ts assistant "What did we discuss yesterday?"

# Check current session status
bun run src/cli.ts status

# Start interactive mode for continuous conversation
bun run src/cli.ts interactive
```

### Session Types

- **ESI (Episodic Structured Interview)**: Structured memory interviews that automatically transition to SR sessions
- **SR (Spaced Retrieval)**: Follow-up sessions for memory reinforcement
- **Assistant**: Direct chat with the memory assistant

### Interactive Mode

For the best experience, use interactive mode:

```bash
bun run src/cli.ts interactive
```

This provides a continuous chat interface where you can switch between session types and commands seamlessly.
