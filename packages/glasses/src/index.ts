import { AppServer, AppSession } from "@mentra/sdk"
import { createClient } from '@supabase/supabase-js'
import { randomUUID } from "crypto"
// Create Supabase client
const supabase = createClient(process.env.SUPABASE_URL!, process.env.SUPABASE_SERVICE_ROLE_KEY!)

// Load configuration from environment variables
const PACKAGE_NAME = process.env.PACKAGE_NAME || "com.example.myfirstmentraosapp"
const PORT = parseInt(process.env.PORT || "3000")
const MENTRAOS_API_KEY = process.env.MENTRAOS_API_KEY

if (!MENTRAOS_API_KEY) {
  console.error("MENTRAOS_API_KEY environment variable is required")
  process.exit(1)
}

class MyMentraOSApp extends AppServer {
  protected override async onSession(session: AppSession, sessionId: string, userId: string): Promise<void> {
    session.logger.info(`New session: ${sessionId} for user ${userId}`)

    let currentEsiSessionId: string | null = null
    let aiTalking = false

    session.events.onButtonPress(async (e) => {
      session.logger.info(`Button pressed: ${e.buttonId}`)
      currentEsiSessionId = randomUUID().toString()
      const searchParams = new URLSearchParams({ session_id: currentEsiSessionId })
      const url = process.env.BACKEND_URL + "/esi_session?" + searchParams.toString()
      session.logger.info(`Starting ESI session`)
      aiTalking = true
      await fetch(url, {
        method: "POST",
      })
      aiTalking = false
      session.logger.info(`ESI session started`)
    })

    session.events.onTranscription(async (data) => {
      // session.logger.info(`Transcription: ${data.text}`)
      const searchParams = new URLSearchParams({ text: data.text, is_final: data.isFinal.toString() })
      const url = process.env.BACKEND_URL + "/current_transcription?" + searchParams.toString()
      await fetch(url, {
        method: "POST",
      })
      session.logger.info(`Transcription sent to backend: ${data.text}`)
      if (data.isFinal && !aiTalking && currentEsiSessionId) {
        const searchParams = new URLSearchParams({ session_id: currentEsiSessionId, user_message: data.text })
        const url = process.env.BACKEND_URL + "/esi_chat?" + searchParams.toString()
        aiTalking = true
        await fetch(url, {
          method: "POST",
        })
        aiTalking = false
      }
    })

    const statusUnsubscribe = session.camera.onManagedStreamStatus((data) => {
      session.logger.info(`Managed stream status: ${JSON.stringify(data, null, 2)}`)
    })

    const photoInterval = setInterval(async () => {
      // const photo = await session.camera.requestPhoto({ saveToGallery: true })
    }, 5000)

    // Log when the session is disconnected
    session.events.onDisconnected(() => {
      session.logger.info(`Session ${sessionId} disconnected.`)
      clearInterval(photoInterval)
      statusUnsubscribe()
    })
  }
}

// Create and start the app server
const server = new MyMentraOSApp({
  packageName: PACKAGE_NAME,
  apiKey: MENTRAOS_API_KEY,
  port: PORT,
})

server.start().catch(err => {
  console.error("Failed to start server:", err)
})
