import { AppServer, AppSession } from "@mentra/sdk"
import { createClient } from '@supabase/supabase-js'
import { randomUUID } from "crypto"
// Create Supabase client
const supabase = createClient(process.env.SUPABASE_URL!, process.env.SUPABASE_SERVICE_ROLE_KEY!)

// Load configuration from environment variables
const PACKAGE_NAME = process.env.PACKAGE_NAME || "com.example.myfirstmentraosapp"
const PORT = parseInt(process.env.PORT || "3000")
const MENTRAOS_API_KEY = process.env.MENTRAOS_API_KEY
const WAKE_WORD = process.env.WAKE_WORD || "jarvis"

if (!MENTRAOS_API_KEY) {
  console.error("MENTRAOS_API_KEY environment variable is required")
  process.exit(1)
}

class MyMentraOSApp extends AppServer {
  protected override async onSession(session: AppSession, sessionId: string, userId: string): Promise<void> {
    session.logger.info(`New session: ${sessionId} for user ${userId}`)

    let currentEsiSessionId: string | null = null
    let currentSRSessionId: string | null = null
    let wakeWordDetected = false
    let aiTalking = false
    let accumulatedTranscription = ""

    session.events.onButtonPress(async (e) => {
      session.logger.info(`Button pressed: ${e.buttonId}`)

      // If we have accumulated transcription and an active ESI session, submit it via button
      if (accumulatedTranscription) {
        if (wakeWordDetected || accumulatedTranscription.toLowerCase().includes(WAKE_WORD)) {
          session.logger.info(`Button-submitting accumulated transcription to Assistant chat: ${accumulatedTranscription}`)
          const chatParams = new URLSearchParams({ user_message: accumulatedTranscription })
          const chatUrl = process.env.BACKEND_URL + "/assistant_chat?" + chatParams.toString()
          aiTalking = true
          const response = await fetch(chatUrl, {
            method: "POST",
          })
          const responseText = await response.text()
          aiTalking = false
          session.logger.info(`Accumulated transcription button-submitted to Assistant chat`)
        }
        else if (currentEsiSessionId) {
          session.logger.info(`Button-submitting accumulated transcription to ESI session: ${accumulatedTranscription}`)
          const chatParams = new URLSearchParams({ session_id: currentEsiSessionId, user_message: accumulatedTranscription })
          const chatUrl = process.env.BACKEND_URL + "/esi_chat?" + chatParams.toString()
          aiTalking = true
          const response = await fetch(chatUrl, {
            method: "POST",
          })
          const responseText = await response.text()
          if (responseText.includes("end_esi_session")) {
            session.logger.info(`ESI session ended`)
            currentEsiSessionId = null
            // Start the SR session
            currentSRSessionId = randomUUID().toString()
            const searchParams = new URLSearchParams({ session_id: currentSRSessionId })
            const srUrl = process.env.BACKEND_URL + "/sr_session?" + searchParams.toString()
            await fetch(srUrl, {
              method: "POST",
            })
            session.logger.info(`SR session started`)
          }
          aiTalking = false
          session.logger.info(`Accumulated transcription button-submitted to ESI session`)
        } else if (currentSRSessionId) {
          session.logger.info(`Button-submitting accumulated transcription to SR session: ${accumulatedTranscription}`)
          const chatParams = new URLSearchParams({ session_id: currentSRSessionId, user_message: accumulatedTranscription })
          const chatUrl = process.env.BACKEND_URL + "/sr_chat?" + chatParams.toString()
          aiTalking = true
          const response = await fetch(chatUrl, {
            method: "POST",
          })
          const responseText = await response.text()
          if (responseText.includes("end_sr_session")) {
            session.logger.info(`SR session ended`)
            currentSRSessionId = null
          }
          aiTalking = false
          session.logger.info(`Accumulated transcription button-submitted to SR session`)
        }

        // Clear the accumulated transcription after successful submission
        accumulatedTranscription = ""
        session.logger.info(`Accumulated transcription cleared after button submission`)
        return
      }

      // If no accumulated transcription or no active session, start a new ESI session
      currentEsiSessionId = randomUUID().toString()
      const searchParams = new URLSearchParams({ session_id: currentEsiSessionId })
      const url = process.env.BACKEND_URL + "/esi_session?" + searchParams.toString()
      session.logger.info(`Starting new ESI session`)
      aiTalking = true
      await fetch(url, {
        method: "POST",
      })
      aiTalking = false
      session.logger.info(`ESI session started`)
    })

    session.events.onTranscription(async (data) => {
      // session.logger.info(`Transcription: ${data.text}`)

      // Accumulate transcription text for button submission
      if (data.isFinal) {
        accumulatedTranscription += (accumulatedTranscription ? " " : "") + data.text
        session.logger.info(`Accumulated transcription: ${accumulatedTranscription}`)

        // Check for wake word to start assistant session
        if (data.text.toLowerCase().includes(WAKE_WORD) && !wakeWordDetected) {
          session.logger.info(`Wake word detected, querying Assistant`)
          wakeWordDetected = true
          const searchParams = new URLSearchParams({ user_message: accumulatedTranscription })
          const url = process.env.BACKEND_URL + "/assistant_chat?" + searchParams.toString()
          aiTalking = true
          accumulatedTranscription = ""
          await fetch(url, {
            method: "POST",
          })
          aiTalking = false
          wakeWordDetected = false
          session.logger.info(`Assistant query started`)
          return
        }
      }

      const searchParams = new URLSearchParams({ text: data.text, is_final: data.isFinal.toString() })
      const url = process.env.BACKEND_URL + "/current_transcription?" + searchParams.toString()
      await fetch(url, {
        method: "POST",
      })
      session.logger.info(`Transcription sent to backend: ${data.text}`)

      // Submit to ESI session on final transcription (automatic submission)
      // if (data.isFinal && !aiTalking && currentEsiSessionId) {
      //   session.logger.info(`Auto-submitting final transcription to ESI session: ${data.text}`)
      //   const chatParams = new URLSearchParams({ session_id: currentEsiSessionId, user_message: data.text })
      //   const chatUrl = process.env.BACKEND_URL + "/esi_chat?" + chatParams.toString()
      //   aiTalking = true
      //   await fetch(chatUrl, {
      //     method: "POST",
      //   })
      //   aiTalking = false
      //   session.logger.info(`Final transcription auto-submitted to ESI session`)

      //   // Clear the accumulated transcription since it was just submitted
      //   accumulatedTranscription = ""
      //   session.logger.info(`Accumulated transcription cleared after auto-submission`)
      // }
    })

    // Log when the session is disconnected
    session.events.onDisconnected(() => {
      session.logger.info(`Session ${sessionId} disconnected.`)
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
