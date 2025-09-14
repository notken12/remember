#!/usr/bin/env bun
import { randomUUID } from "crypto"

// Load configuration from environment variables
const BACKEND_URL = process.env.BACKEND_URL
const WAKE_WORD = process.env.WAKE_WORD || "jarvis"

if (!BACKEND_URL) {
    console.error("BACKEND_URL environment variable is required")
    process.exit(1)
}

interface SessionState {
    esiSessionId: string | null
    srSessionId: string | null
    assistantSessionId: string | null
}

class RememberCLI {
    private sessionState: SessionState = {
        esiSessionId: null,
        srSessionId: null,
        assistantSessionId: null
    }

    async startEsiSession(): Promise<string> {
        const sessionId = randomUUID().toString()
        const searchParams = new URLSearchParams({ session_id: sessionId })
        const url = `${BACKEND_URL}/esi_session?${searchParams.toString()}`

        console.log("🔄 Starting new ESI session...")

        try {
            const response = await fetch(url, { method: "POST" })
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`)
            }

            this.sessionState.esiSessionId = sessionId
            console.log(`✅ ESI session started with ID: ${sessionId}`)
            return sessionId
        } catch (error) {
            console.error("❌ Failed to start ESI session:", error)
            throw error
        }
    }

    async chatWithEsi(sessionId: string, message: string): Promise<string> {
        const chatParams = new URLSearchParams({
            session_id: sessionId,
            user_message: message
        })
        const chatUrl = `${BACKEND_URL}/esi_chat?${chatParams.toString()}`

        console.log(`💬 Sending message to ESI session: "${message}"`)

        try {
            const response = await fetch(chatUrl, { method: "POST" })
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`)
            }

            const responseText = await response.text()
            console.log(`🤖 ESI Response: ${responseText}`)

            // Check if ESI session should end and SR session should start
            if (responseText.includes("end_esi_session")) {
                console.log("🔄 ESI session ended, starting SR session...")
                this.sessionState.esiSessionId = null
                await this.startSrSession()
            }

            return responseText
        } catch (error) {
            console.error("❌ Failed to chat with ESI:", error)
            throw error
        }
    }

    async startSrSession(): Promise<string> {
        const sessionId = randomUUID().toString()
        const searchParams = new URLSearchParams({ session_id: sessionId })
        const srUrl = `${BACKEND_URL}/sr_session?${searchParams.toString()}`

        console.log("🔄 Starting SR session...")

        try {
            const response = await fetch(srUrl, { method: "POST" })
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`)
            }

            this.sessionState.srSessionId = sessionId
            console.log(`✅ SR session started with ID: ${sessionId}`)
            return sessionId
        } catch (error) {
            console.error("❌ Failed to start SR session:", error)
            throw error
        }
    }

    async chatWithSr(sessionId: string, message: string): Promise<string> {
        const chatParams = new URLSearchParams({
            session_id: sessionId,
            user_message: message
        })
        const chatUrl = `${BACKEND_URL}/sr_chat?${chatParams.toString()}`

        console.log(`💬 Sending message to SR session: "${message}"`)

        try {
            const response = await fetch(chatUrl, { method: "POST" })
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`)
            }

            const responseText = await response.text()
            console.log(`🤖 SR Response: ${responseText}`)

            // Check if SR session should end
            if (responseText.includes("end_sr_session")) {
                console.log("✅ SR session ended")
                this.sessionState.srSessionId = null
            }

            return responseText
        } catch (error) {
            console.error("❌ Failed to chat with SR:", error)
            throw error
        }
    }

    async chatWithAssistantSession(sessionId: string, message: string): Promise<string> {
        const chatParams = new URLSearchParams({
            session_id: sessionId,
            user_message: message
        })
        const chatUrl = `${BACKEND_URL}/assistant_chat?${chatParams.toString()}`

        console.log(`💬 Sending message to Assistant session: "${message}"`)

        try {
            const response = await fetch(chatUrl, { method: "POST" })
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`)
            }

            const responseText = await response.text()
            console.log(`🤖 Assistant Response: ${responseText}`)
            return responseText
        } catch (error) {
            console.error("❌ Failed to chat with Assistant:", error)
            throw error
        }
    }

    async chatWithAssistant(message: string): Promise<string> {
        const chatParams = new URLSearchParams({ user_message: message })
        const chatUrl = `${BACKEND_URL}/assistant_chat?${chatParams.toString()}`

        console.log(`💬 Sending message to Assistant: "${message}"`)

        try {
            const response = await fetch(chatUrl, { method: "POST" })
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`)
            }

            const responseText = await response.text()
            console.log(`🤖 Assistant Response: ${responseText}`)
            return responseText
        } catch (error) {
            console.error("❌ Failed to chat with Assistant:", error)
            throw error
        }
    }

    async sendTranscription(text: string, isFinal: boolean = true): Promise<void> {
        const searchParams = new URLSearchParams({
            text: text,
            is_final: isFinal.toString()
        })
        const url = `${BACKEND_URL}/current_transcription?${searchParams.toString()}`

        try {
            await fetch(url, { method: "POST" })
            console.log(`📝 Transcription sent: "${text}" (final: ${isFinal})`)
        } catch (error) {
            console.error("❌ Failed to send transcription:", error)
            throw error
        }
    }

    printStatus(): void {
        console.log("\n📊 Current Session Status:")
        console.log(`ESI Session: ${this.sessionState.esiSessionId || "None"}`)
        console.log(`SR Session: ${this.sessionState.srSessionId || "None"}`)
        console.log(`Assistant Session: ${this.sessionState.assistantSessionId || "None"}`)
    }

    printHelp(): void {
        console.log(`
🔮 Remember CLI - Interactive Memory Assistant

Commands:
  esi start                    Start a new ESI (Episodic Structured Interview) session
  esi chat <message>           Send a message to the current ESI session
  sr start                     Start a new SR (Spaced Retrieval) session  
  sr chat <message>            Send a message to the current SR session
  assistant chat <message>     Send a message to the current Assistant session
  assistant <message>          Chat directly with the assistant (no session)
  transcription <text>         Send transcription text to backend
  status                       Show current session status
  interactive                  Start interactive mode
  help                         Show this help message
  exit                         Exit the CLI

Examples:
  remember esi start
  remember esi chat "Tell me about your childhood"
  remember assistant chat "What did we discuss yesterday?"
  remember assistant "Quick question without session"
  remember interactive

Environment Variables:
  BACKEND_URL                  Backend server URL (required)
  WAKE_WORD                    Wake word for assistant (default: "jarvis")
`)
    }

    async interactiveMode(): Promise<void> {
        console.log("🚀 Starting interactive mode. Type 'help' for commands or 'exit' to quit.")

        const readline = await import("readline")
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
            prompt: "remember> "
        })

        rl.prompt()

        rl.on('line', async (input: string) => {
            const trimmed = input.trim()
            if (!trimmed) {
                rl.prompt()
                return
            }

            try {
                await this.handleCommand(trimmed.split(' '))
            } catch (error) {
                console.error("❌ Error:", error)
            }

            rl.prompt()
        })

        rl.on('close', () => {
            console.log("\n👋 Goodbye!")
            process.exit(0)
        })
    }

    async handleCommand(args: string[]): Promise<void> {
        const [command, subcommand, ...rest] = args

        switch (command) {
            case 'esi':
                if (subcommand === 'start') {
                    await this.startEsiSession()
                } else if (subcommand === 'chat') {
                    if (!this.sessionState.esiSessionId) {
                        console.log("⚠️  No active ESI session. Starting one...")
                        await this.startEsiSession()
                    }
                    const message = rest.join(' ')
                    if (!message) {
                        console.log("❌ Please provide a message to send")
                        return
                    }
                    await this.chatWithEsi(this.sessionState.esiSessionId!, message)
                } else {
                    console.log("❌ Unknown ESI command. Use 'start' or 'chat <message>'")
                }
                break

            case 'sr':
                if (subcommand === 'start') {
                    await this.startSrSession()
                } else if (subcommand === 'chat') {
                    if (!this.sessionState.srSessionId) {
                        console.log("⚠️  No active SR session. Starting one...")
                        await this.startSrSession()
                    }
                    const message = rest.join(' ')
                    if (!message) {
                        console.log("❌ Please provide a message to send")
                        return
                    }
                    await this.chatWithSr(this.sessionState.srSessionId!, message)
                } else {
                    console.log("❌ Unknown SR command. Use 'start' or 'chat <message>'")
                }
                break

            case 'assistant':
                if (subcommand === 'chat') {
                    const message = rest.join(' ')
                    if (!message) {
                        console.log("❌ Please provide a message to send")
                        return
                    }
                    if (!this.sessionState.assistantSessionId) {
                        console.log("⚠️  No active Assistant session. Starting one...")
                        this.sessionState.assistantSessionId = randomUUID().toString()
                    }
                    await this.chatWithAssistantSession(this.sessionState.assistantSessionId!, message)
                } else {
                    // Backwards compatibility: treat any other subcommand as a direct message
                    const assistantMessage = [subcommand, ...rest].join(' ')
                    if (!assistantMessage) {
                        console.log("❌ Please provide a message for the assistant")
                        return
                    }

                    // Check for wake word
                    if (assistantMessage.toLowerCase().includes(WAKE_WORD)) {
                        console.log(`🎯 Wake word "${WAKE_WORD}" detected!`)
                    }

                    await this.chatWithAssistant(assistantMessage)
                }
                break

            case 'transcription':
                const transcriptionText = [subcommand, ...rest].join(' ')
                if (!transcriptionText) {
                    console.log("❌ Please provide transcription text")
                    return
                }
                await this.sendTranscription(transcriptionText)
                break

            case 'status':
                this.printStatus()
                break

            case 'interactive':
                await this.interactiveMode()
                break

            case 'help':
                this.printHelp()
                break

            case 'exit':
                console.log("👋 Goodbye!")
                process.exit(0)
                break

            default:
                console.log(`❌ Unknown command: ${command}`)
                console.log("Type 'help' for available commands")
        }
    }
}

async function main() {
    const cli = new RememberCLI()
    const args = process.argv.slice(2)

    if (args.length === 0) {
        cli.printHelp()
        return
    }

    try {
        await cli.handleCommand(args)
    } catch (error) {
        console.error("❌ Error:", error)
        process.exit(1)
    }
}

// Run the CLI
if (import.meta.main) {
    main()
}
