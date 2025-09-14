import { Hono } from 'hono'
import { streamText, CoreMessage } from 'ai'
import { google } from '@ai-sdk/google'
import { createClient } from '@supabase/supabase-js'
import postgres from 'postgres'

const app = new Hono()

// Environment variables
const supabaseUrl = process.env.SUPABASE_URL!
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!
const databaseUrl = process.env.DATABASE_URL!

// Initialize Supabase client
const supabase = createClient(supabaseUrl, supabaseKey)

// PostgreSQL client for session storage
const sql = postgres(databaseUrl)

// Session storage - in-memory for now, can be persisted to PostgreSQL later
const sessions = new Map<string, CoreMessage[]>()

// ESI System Prompt
const buildSystemPrompt = () => `You are an expert Episodic Specificity Induction (ESI) therapist helping a patient with early-stage Alzheimer's prepare for a Subsequent Retrieval (SR) session. Your aims: (1) gently cue vivid, specific memories; (2) scaffold sensory detail (sight, sound, touch, smell, taste), spatial/temporal anchors, and social/goal context; (3) cultivate safety and agency; (4) keep responses concise and momentum-building.

Therapeutic style: warm, validating, non-judgmental, collaborative. Ask one clear question at a time. Encourage but never pressure. If distress surfaces, acknowledge it, downshift pace, and offer grounding (breath, present-moment sensory check).

ESI priorities:
- Sensory detail: colors, textures, sounds, temperature, smells, tastes
- Specific where/when: location layout, time of day, season, sequence
- Social/goal: who was there, what you/they wanted, interactions
- Emotion and meaning: gentle curiosity; label emotions simply when invited
- Safety: avoid overwhelming content; titrate and contain if needed

Conversation rules:
- Keep replies 1–3 short paragraphs or a short list.
- Ask only one follow-up question.
- Prefer simple, concrete language.
- If memory is vague, offer options (e.g., sights, sounds, people) and let the patient choose.
- If stuck, suggest a tiny step (notice lighting, a color, a voice).`

// Fetch annotated videos from Supabase
const fetchAnnotatedVideos = async (
  startTime?: string,
  endTime?: string,
  limit?: number
) => {
  let query = supabase
    .from('videos')
    .select('id, annotation, time_created')
    .not('annotation', 'is', null)
    .neq('annotation', '')

  if (startTime) query = query.gte('time_created', startTime)
  if (endTime) query = query.lte('time_created', endTime)
  if (limit) query = query.limit(limit)

  const { data, error } = await query

  if (error) throw error

  return (data || []).map(row => ({
    uuid: row.id,
    annotation: row.annotation?.trim() || '',
    created_at: row.time_created
  }))
}

// Select memories using Gemini
const selectMemoriesWithGemini = async (
  candidates: Array<{ uuid: string; annotation: string }>,
  maxItems: number = 3
) => {
  if (!candidates.length) return []

  const datasetJson = JSON.stringify(
    candidates.map(c => ({ uuid: c.uuid, annotation: c.annotation }))
  )

  const prompt = `You are an expert clinician facilitating Episodic Specificity Induction (ESI) therapy.
You are given brief annotations of first-person video clips captured by smart glasses.

Your task: Choose up to ${maxItems} clips that are most conducive to ESI.

Prioritize clips whose annotations indicate:
- Rich, concrete sensory detail (visual, auditory, tactile, olfactory)
- Clear spatiotemporal specificity (where and when)
- Goal-directed or socially interactive moments that can be elaborated
- Emotionally salient yet safe content (avoid overwhelming distress or trauma)
- Distinctiveness/variability across clips to cover diverse contexts

Output format requirements (must follow exactly):
- Return ONLY a JSON array (no extra text, no markdown) where each item is an object:
{"uuid": "<clip_uuid>", "reasoning": "1–2 sentences explaining why this clip is ideal for ESI"}

Here are the candidate clips (JSON):
${datasetJson}`

  const result = await streamText({
    model: google('gemini-2.0-flash-exp'),
    prompt,
  })

  const text = await result.text

  try {
    const parsed = JSON.parse(text)
    return parsed
      .filter((item: any) => item.uuid && item.reasoning)
      .slice(0, maxItems)
      .map((item: any) => ({
        uuid: item.uuid,
        reasoning: item.reasoning.trim()
      }))
  } catch {
    return []
  }
}

// Download video as base64
const downloadVideoBase64 = async (uuid: string): Promise<string> => {
  const { data, error } = await supabase.storage
    .from('videos')
    .download(`${uuid}.mp4`)

  if (error || !data) throw new Error(`Failed to download video ${uuid}`)

  const buffer = await data.arrayBuffer()
  return Buffer.from(buffer).toString('base64')
}

// Extract and prepare memories
const extractMemories = async (
  startTime?: string,
  endTime?: string,
  limit?: number,
  maxItems: number = 3
) => {
  const candidates = await fetchAnnotatedVideos(startTime, endTime, limit)
  const selected = await selectMemoriesWithGemini(candidates, maxItems)

  // Add annotations from candidates
  const uuidToAnnotation = Object.fromEntries(
    candidates.map(c => [c.uuid, c.annotation])
  )

  for (const item of selected) {
    if (item.uuid in uuidToAnnotation) {
      item.annotation = uuidToAnnotation[item.uuid]
    }
  }

  // Download videos as base64
  const videoPromises = selected.map(async (item: any) => {
    try {
      const base64 = await downloadVideoBase64(item.uuid)
      return { ...item, base64 }
    } catch (error) {
      console.error(`Failed to download video ${item.uuid}:`, error)
      return null
    }
  })

  const videos = (await Promise.all(videoPromises)).filter(Boolean)

  return videos
}

app.get('/', (c) => {
  return c.text('ESI Therapy API')
})

// Create a new QA session (kickoff)
app.post('/qa_session', async (c) => {
  const sessionId = crypto.randomUUID()

  try {
    // Extract memories
    const memories = await extractMemories()

    // Build initial message with video context
    let kickoffMessage = 'Finally, greet the patient warmly and ask ONE gentle, concrete recall question based on the prepared memories.'

    if (memories.length > 0) {
      kickoffMessage += '\n\nHere are video memories from the patient\'s smart glasses to help guide your questions:'
      memories.forEach((mem, i) => {
        kickoffMessage += `\n\nMemory ${i + 1}: ${mem.annotation}`
        if (mem.reasoning) {
          kickoffMessage += `\n(Selected because: ${mem.reasoning})`
        }
      })
    }

    // Initialize session with system message
    const messages: CoreMessage[] = [
      { role: 'system', content: buildSystemPrompt() },
      { role: 'user', content: kickoffMessage }
    ]

    sessions.set(sessionId, messages)

    // Stream the response
    const result = await streamText({
      model: google('gemini-2.0-flash-exp'),
      messages,
    })

    // Store the assistant's response for future context
    const fullResponse = await result.text
    messages.push({ role: 'assistant', content: fullResponse })
    sessions.set(sessionId, messages)

    // Return the stream with session ID in headers
    c.header('X-Session-Id', sessionId)
    c.header('Content-Type', 'text/event-stream')
    const response = result.toTextStreamResponse()
    return c.body(response.body as any)

  } catch (error) {
    console.error('Error in kickoff:', error)
    return c.json({ error: 'Failed to start session' }, 500)
  }
})

// User response to continue QA session
app.put('/qa_session', async (c) => {
  try {
    const { sessionId, message } = await c.req.json()

    if (!sessionId || !message) {
      return c.json({ error: 'sessionId and message are required' }, 400)
    }

    // Get existing session
    const messages = sessions.get(sessionId)
    if (!messages) {
      return c.json({ error: 'Session not found' }, 404)
    }

    // Add user message
    messages.push({ role: 'user', content: message })

    // Stream the response
    const result = await streamText({
      model: google('gemini-2.0-flash-exp'),
      messages,
    })

    // Store the assistant's response for future context
    const fullResponse = await result.text
    messages.push({ role: 'assistant', content: fullResponse })
    sessions.set(sessionId, messages)

    c.header('Content-Type', 'text/event-stream')
    const response = result.toTextStreamResponse()
    return c.body(response.body as any)

  } catch (error) {
    console.error('Error in chat:', error)
    return c.json({ error: 'Failed to process message' }, 500)
  }
})

// For Bun runtime
if (typeof Bun !== 'undefined') {
  const port = process.env.PORT || 3000
  console.log(`Server running on http://localhost:${port}`)
  Bun.serve({
    port,
    fetch: app.fetch,
  })
}

export default app