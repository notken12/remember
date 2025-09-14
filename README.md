# remember

## Inspiration

82,000,000,000. That’s how many man-hours are dedicated for informal caregiving helping people with neurodegenerative disorders. Alzheimer’s and other associated diseases impact an absurd amount of individuals across the globe, more than 56,000,000 to be precise. Alzheimer’s is a haunting disease. The erasure of the ability to form memories leads to confusion, frustration, and the loss of “self-concept.” Eventually, all things will wither, and everything you’ve worked so hard to see throughout your life, watching your grandkids grow up, take their first steps, start to exhibit some of the same behaviors you did when you were younger, everything shatters. 

Despite the magnitude of this problem, the technology used to address Alzheimer’s is still primitive. It has been shown that autobiographical recollection is one of the most useful ways to combat Alzheimer’s, but diary-ing isn’t enough. 80% of all memory is visual, and without supporting media, studies show that long-term memory formation is futile. 

When we heard of this issue, an obvious solution came to mind; if diarying doesn’t work, why not record your day on a camera? This solution was actually implemented in 2003 with Microsoft’s Sensecam, and it was revolutionary for patients with Alzheimer’s, showing exponential increases in remembrance, and garnering nearly 1,000 citations. However, the technology was primitive and expensive. Limited storage led to image-only capture, and the mechanism for automatic image capture was through analyzing changes in sensors. Moreover, the only form of review was looking at the raw images, there was no prompting. 

Since Sensecam, almost no one has dared to take on Alzheimer's in the visual space. With the rise of smart glasses and artificial intelligence, we saw a massive hole in this space. The rest is in the details: using SoTA Alzheimer’s research to design specific training curricula using artificial intelligence, designing multi-agent workflows that facilitate this research, and creating engaging conversation using Cerebras’ blazing fast inference.

##What it does

Our system can primarily be split up into 3 agents, that together facilitate the autobiographical rehabilitation process for the patient, and/or increase quality of life.

### Episodic Specificity Induction (ESI) Agent:

The ESI agent’s primary role is to “jump start” the “episodic construction network” to be active and ready before the main Spaced Retrieval (SR) tasks (see https://pmc.ncbi.nlm.nih.gov/articles/PMC6361685/). In ESI, the patient has no fixed task, but is shown memories from the past and asked to speak and reminisce about them, with light levels of persuasion to elucidate on their experiences. 

The ESI agent is a multi-agent system. First, there’s a “candidate extractor” agent, which takes in memories from a database and decides which ones are worthy to use for ESI specific training. In particular, videos that tend to elicit large amounts of reminiscence are prioritized. 

Next, these memories are sent to the ESI agent, and the ESI agent is tasked with providing an interactive conversation with the patient about the memory, i.e. how they felt, what they remember seeing and smelling, etc. As shown in the ESI literature provided, people with Alzheimer’s get frustrated easily if they don’t remember something, so the agent is designed to be as gentle as possible.

### Spaced Retrieval (SR) Agent:

After ESI, an SR curriculum is run to help reinforce the patient’s memory (https://pmc.ncbi.nlm.nih.gov/articles/PMC7174872/). The core of SR is asking a question at exponentially spaced-apart interval sizes (i.e. 30 seconds, 1 minute, 2 minutes, 4 minutes). Remarkably, just 16 minutes of SR is shown to preserve recollection for weeks, even months. In SR, if a patient gets a question wrong after a particular time interval, the correct answer is immediately displayed and the difficulty of that question drops down one notch as to not discourage the patient (as shown helps in literature), and to also prevent error encoding. However, if the patient gets the problem correct, the difficulty is increased a notch.

Since SR is a more involved and rigid curriculum, the multi-agent system that supports it is naturally much more involved than ESI. 

Just like ESI, we first have a candidate extractor that extracts relevant memories that are relevant to SR tasks. There are actually many sample SR questions in the paper which we used to populate the system prompt of the SR question generation model. The questions asked for SR are “cue-based,” i.e. they provide some hint about the situation to jog the relevant areas of the brain, and are shown to be more effective (they use the implicit cognitive functions of the brain to push memories to long-term, and they haven’t been as damaged yet by Alzheimer’s). 

After the questions are generated, they are processed via a complex priority queue structure. The core idea of our implementation is that it’s inefficient to wait long durations of time without asking questions, so we need to “interleave” the questions with one another to minimize total session time. To do this, we pop a question from the priority queue if its time has arrived, and if there are currently no questions to be popped, we have a small talk agent to make the patient feel comfortable. After the patient answers a question, we feed the answer into a verifier model that appends to the priority queue at different durations depending on the criteria listed above.

### General Assistant Agent:

The general assistant agent’s functionality is to tell the user about a particular memory if they have a question. More generally, if the user has lost something or forgotten where something is, they can ask the general assistant and it can query its vast database of knowledge to answer the question.

The workflow for the general agent is simple. A query is processed, and the model searches through its memories, looking for similarities with existing memories. It then outputs the memory most closely related with the query (if it’s related enough), along with an answer. 

Example: Where’s my wallet?
Example Answer: On top of the dresser drawer in the living room. <Show image>

A useful note is that all of these agents support tool calls to interact with the camera display, displaying particular memories. This is useful for ESI/SR if probing/prompting is necessary to elicit memory activation in the patient, and it’s useful in the general assistant agent to point out where a particular object/item is.

## How we built it

Our project consisted of:
- A MentraOS app
- A python backend that runs our agents
- A Next.js web UI exclusively for demos that shows the POV of a user

As the user goes through their day, the glasses record memories as videos and upload them to Supabase Object Storage, and then the backend runs Gemini 2.5 Flash to generate annotations for the content of the videos, which are stored in Supabase Tables. 

We allow the user to start a therapy session by clicking a button on their glasses. The agent, which we built using LangGraph, retrieves the day’s memories and determines which memories are the most interesting to the user, based on factors such as novelty of the experience and the presence of loved ones through facial recognition (to be implemented in the future). We gave the model tools to show an image of or a few seconds of the video memory to the user, which helps stimulate their visual memory and increases results based on the SenseCam study. 

To stream the conversation to the Next.js web demo UI, we needed to use a Pub-Sub architecture with Redis. When the agent outputs tokens or tool calls, the chunks are broadcast to a pub-sub channel. The Next.js UI subscribes to this pub-sub channel through an SSE endpoint, allowing all LLM outputs to be displayed. We also show live transcriptions of the user’s voice through this stream, achieving a feature-complete POV of the user.

Though most users would choose to engage with our therapy experience in a quiet environment, the event venue is quite loud, making it difficult to speak clearly to the agent. To make our demo reliable, we created a CLI that lets us type in what the user is saying to the agent for the sake of the demo, going through the exact same pub-sub channel as the glasses and essentially mocking the input. 

## Individual Contributions
We were a team of three so we had constant communication and worked together extensively. However, here are the general areas in which we each focused. 

### Ken: I mainly worked on the MentraOS app, front-end HUD, streaming, and integrating everything (glasses, frontend, backend, database, etc.) together using API routes and pub-sub. I did a deep dive into MentraOS’s abilities and limitations and used what I learned to architect the overall structure of our project and the frameworks we used.

### Vishal: I designed the overall algorithms and approach we took. I focused on building the Spaced Retrieval (SR) agent and helped with the HUD. 

### Samarth: I built both the Episodic Specificity Induction (ESI) and general assistant agents. Specifically, I specced out the agent workflows, ported functionality to Langgraph, and used Cerebras to speed up inference, yielding smooth conversation. I also created the base backend infrastructure for video storage and chat messages/sessions.
Challenges we ran into

We chose to use LangGraph for its ability to automatically persist agent graph state such as conversation history to our Postgres database. However, we experienced significant issues with the library’s builtin Postgres “checkpointer” and it often failed to write into the database. We ultimately decided to write our own message database schema and code to save messages after each LLM inference and tool call result, as well as retrieve session data when chatting in existing sessions. 

The only way to access the video feed from the smart glasses was to use RTMP streaming. Due to the early beta nature of MentraOS, this was very unreliable. Its managed stream option had a 15 second delay and automatically shut off after around 2 minutes, and streaming to custom RTMP servers did not work. We worked around this by building a mock of the user’s perspective for our demo using a web UI that accesses the browser’s camera and overlaying a mock HUD of the conversation history the user would see. Wi-Fi was also unreliable due to the high bandwidth usage causing instability at the venue, which we worked around by using cellular hotspots. 

The venue is also quite loud, which makes it difficult to talk to the AI agents we built. We worked around this by creating a CLI exclusively for our demo where we type in what the demonstrator is asking the glasses.

We initially faced difficulties with the latency. Specifically, processing all that video data and dealing with the latency of the glasses added up. To combat this, we implemented extensive pre-processing when the videos are added to Supabase with Gemini 2.5 Flash with detailed transcriptions and knowledge management. Finally, we switched to cerebras for the audio inference to dramatically reduce latency overall. 

Finally, we had difficulty processing the messy data. We imported many, many videos into Supabase (since we didn’t have time to use the glasses extensively and record a bunch of videos), so that we could demonstrate difficult retrieval and memory management capabilities. However, dealing with these large amounts of messy internet data was difficult. We used parallelized multiagent filtering to decide which videos were useful for each task. 

## Accomplishments that we're proud of

We’re proud that we got such complex multi-agent systems (with tool calls!) working in the first place. We worked on the SR agent for more than 8 hours straight, showing the complexity and depth of the code. We’re infinitely indebted to LangGraph for providing such a great way to do multiagent orchestration. 

We’re also proud that we put such a significant amount of time into researching what issues actually afflict people with Alzheimer’s and what the most effective ways to solve them are. A wise man once said, coding is just the expression of ideas. We’re very pleased that we put so much time into thinking of the ideas, so the code came out well. 

## What we learned

The main thing we learned through this project was how to pivot our ideas quickly when they failed on us. The smart glasses we received were not able to support streaming reliably, and we struggled with integrating them since they were such a new technology. However, by properly segmenting our work, making API endpoints and robust code (thanks Ken), we were able to circumvent these issues by building last minute solutions that didn’t rely so heavily on dependencies. 

We also learned the importance of making every design choice intentionally geared towards the user. From the LLM prompts to diving deep in the weeds of cognitive research, we tried our best to tune our product to genuinely be useful for a person with Alzheimer’s, and it definitely paid off given our outcome.

## What's next for our project

There are many avenues, both algorithmic and conceptual, which we can explore. Here are a couple:

- Using vector search/knowledge graph for memory retrieval. Currently, we’re just shoving everything into context, which is fine short term since Google Gemini 2.5 Pro miraculously has 45 minutes of video. However, this is not sustainable in the long term. 
- Emergency response. The glasses have many useful features, including calculating head positioning among other things. This can be used to detect patient falls and dangerous scenarios where calling EMS is necessary. 
- Scale to more types of neurodegenerative disease. SR and ESI have been shown to be successful for mild and moderate Alzheimer’s and amnesia patients. We can explore other ND and see if these techniques extend.
- Scaling to large amounts of video using techniques like LongVLM (https://arxiv.org/abs/2404.03384). 

## Links

Slides: https://docs.google.com/presentation/d/1gzFsC1lWWLn0gXMJ1s5Le_rHjwm8zjtBF0E-HNu_sDU/edit?usp=sharing
