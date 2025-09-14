'use client';

import { useEffect, useRef, useState } from 'react';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { produce } from 'immer'

import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { readUIMessageStream, UIMessage, UIMessageChunk } from 'ai';
import { cn } from '@/lib/utils';
import Image from 'next/image';
import { createClient } from '@supabase/supabase-js';

export default function DemoPage() {
    const [messages, setMessages] = useState<UIMessage[]>([])
    const streamRef = useRef<ReadableStream<UIMessageChunk> | null>(null)
    const sseRef = useRef<EventSource | null>(null)
    const isCleaningUpRef = useRef(false)
    const [transcription, setTranscription] = useState('')
    const [transcriptionIsFinal, setTranscriptionIsFinal] = useState(false)

    useEffect(() => {
        console.log('loading sse')
        const sse = new EventSource(process.env.NEXT_PUBLIC_BACKEND_URL! + "/web_stream");
        sseRef.current = sse;

        let controller: ReadableStreamDefaultController<UIMessageChunk> | null = null

        const createAndProcessStream = async (): Promise<void> => {
            console.log('Creating new ReadableStream...');
            // Create the ReadableStream and store reference to prevent garbage collection
            const stream = new ReadableStream<UIMessageChunk>({
                start(c) {
                    controller = c
                },
                cancel() {
                    console.log('ReadableStream cancelled')
                    // Only close SSE if we're actually cleaning up, not if stream is just being cancelled
                    if (isCleaningUpRef.current && sseRef.current && sseRef.current.readyState !== EventSource.CLOSED) {
                        sseRef.current.close();
                    }
                },
            });

            streamRef.current = stream;

            // Process the stream in a separate async function
            await processUIMessageStream(stream);

            console.log('Stream processing completed');
        };

        sse.onopen = async () => {
            console.log('SSE connection opened');

            sse.onmessage = (event) => {
                console.log('SSE message received:', event)
                try {
                    // Parse the event data and enqueue it
                    const data = JSON.parse(event.data);
                    if (data.type === 'data-transcription') {
                        setTranscription(data.data.text)
                        setTranscriptionIsFinal(data.data.is_final)
                        console.log('transcription: ', data.data.text)
                        if (data.data.is_final) {
                            console.log('FINAL TRANSCRIPTION')
                            // setMessages(prev => [...prev, { id: crypto.randomUUID().toString(), "role": "user", "parts": [{ "type": "text", text: data.data.text }] }])
                            // // Close current controller if it exists
                            // if (controller) {
                            //     controller.close();
                            //     controller = null;
                            // }
                            // // Create and process new stream
                            // createAndProcessStream();
                            return
                        }
                        return
                    } else if (data.type === "finish") {
                        console.log('FINISH message received, creating new stream...');
                        // Close current controller if it exists
                        if (controller) {
                            controller.close();
                            controller = null;
                        }
                        // Create and process new stream
                        createAndProcessStream();
                        return; // Don't enqueue finish message to the old controller
                    } else {
                        setTranscription("")
                        setTranscriptionIsFinal(false)
                    }

                    // Only enqueue if we have an active controller
                    if (controller) {
                        controller.enqueue(data);
                    }
                } catch (error) {
                    console.log('Error parsing SSE data:', error);
                }
            };
            sse.onerror = (error) => {
                console.log('SSE error:', error);
                controller?.error(error);
            };

            // Create initial stream
            await createAndProcessStream();
        };

        const processUIMessageStream = async (stream: ReadableStream<UIMessageChunk>) => {
            try {
                for await (const uiMessage of readUIMessageStream({ stream })) {
                    setMessages([uiMessage])
                }
            } catch (error) {
                console.log('Error processing UI message stream:', error);
                // Don't close the stream on error, just log it
            }
        };

        return () => {
            console.log('Cleaning up SSE connection')
            isCleaningUpRef.current = true;
            if (sseRef.current && sseRef.current.readyState !== EventSource.CLOSED) {
                sseRef.current.close();
            }
            sseRef.current = null;
            streamRef.current = null;
        }
    }, [])
    return (
        <div className="absolute inset-0">
            <WebcamBackground />
            <div className="absolute inset-0 p-4">
                <ul className="max-h-full overflow-y-auto">
                    {messages.map((msg, msgIndex) => {
                        if (!msg.parts.some(p => !p.type.startsWith('data-'))) return null
                        return (
                            <li
                                key={msg.id || `msg-${msgIndex}`}
                                className={cn(
                                    "p-3 rounded-lg backdrop-blur-sm",
                                    {
                                        "bg-blue-500/40 text-white": msg.role === "user",
                                        "bg-white/40 text-black": msg.role === "assistant"
                                    }
                                )}
                            >
                                {msg.parts.map((part, partIndex) => {
                                    switch (part.type) {
                                        case "text":
                                            return <p key={partIndex} className="whitespace-pre-wrap">{part.text}</p>
                                        case "tool-select_video":
                                            return (
                                                <Video
                                                    key={partIndex}
                                                    uuid={part.input?.video_uuid}
                                                    controls
                                                    autoPlay
                                                />
                                            )
                                        case "tool-show_image":
                                            return (
                                                <Video
                                                    key={partIndex}
                                                    uuid={part.input?.video_uuid}
                                                    muted
                                                    preload="metadata"
                                                />
                                            )
                                        case "tool-show_video_slice":
                                            return (
                                                <VideoSlice
                                                    key={partIndex}
                                                    uuid={part.input?.video_uuid} start={part.input?.start} end={part.input?.end} />
                                            )
                                        case "tool-end_esi_session":
                                            return <p key={partIndex}>ESI session complete! Now let{"'"}s move on to spaced repetition.</p>
                                        case "tool-end_sr_session":
                                            return <p key={partIndex}>SR session complete! You{"'"}re all done, congratulations!</p>
                                        default:
                                            if (part.type.startsWith('data-')) return null
                                            return <div key={partIndex} className="text-xs opacity-70">{JSON.stringify(part)}</div>
                                    }
                                })}
                            </li>
                        )
                    })}
                    <li className={cn("text-foreground", { "opacity-70": !transcriptionIsFinal })}>
                        {transcription}
                    </li>
                </ul>
            </div>
        </div>
    )
}

const supabase = createClient(process.env.NEXT_PUBLIC_SUPABASE_URL!, process.env.NEXT_PUBLIC_SUPABASE_API_KEY!)

function Video({ uuid, ...restProps }: { uuid: string } & React.ComponentProps<'video'>) {
    const [url, setUrl] = useState<string | null>(null)
    useEffect(() => {
        (async () => {
            const { data } = await supabase.from('videos').select('video_path').eq("id", uuid).single().throwOnError()
            setUrl(supabase.storage.from('videos').getPublicUrl(data.video_path).data.publicUrl)
        })()
    }, [uuid])
    return <video src={url}
        className="max-w-96 max-h-96"
        {...restProps}
    />
}

function VideoSlice({ uuid, start, end }: { uuid: string, start: number, end: number }) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const [isLoaded, setIsLoaded] = useState(false);
    const [url, setUrl] = useState<string | null>(null)

    useEffect(() => {
        (async () => {
            const { data } = await supabase.from('videos').select('video_path').eq("id", uuid).single().throwOnError()
            setUrl(supabase.storage.from('videos').getPublicUrl(data.video_path).data.publicUrl)
        })()

        const video = videoRef.current;
        if (!video) return;

        const handleLoadedMetadata = () => {
            // Set the video to start at the specified start time
            video.currentTime = start;
            setIsLoaded(true);
        };

        const handleLoadedData = () => {
            // Auto-play once the video data is loaded
            video.play().catch(error => {
                console.log('Error playing video:', error);
            });
        };

        const handleTimeUpdate = () => {
            // Stop the video when it reaches the end time
            if (video.currentTime >= end) {
                video.pause();
                video.currentTime = start; // Reset to start for potential replay
            }
        };

        const handleSeeked = () => {
            // Ensure we don't seek outside our allowed range
            if (video.currentTime < start) {
                video.currentTime = start;
            } else if (video.currentTime > end) {
                video.currentTime = end;
                video.pause();
            }
        };

        // Add event listeners
        video.addEventListener('loadedmetadata', handleLoadedMetadata);
        video.addEventListener('loadeddata', handleLoadedData);
        video.addEventListener('timeupdate', handleTimeUpdate);
        video.addEventListener('seeked', handleSeeked);

        return () => {
            // Cleanup event listeners
            video.removeEventListener('loadedmetadata', handleLoadedMetadata);
            video.removeEventListener('loadeddata', handleLoadedData);
            video.removeEventListener('timeupdate', handleTimeUpdate);
            video.removeEventListener('seeked', handleSeeked);
        };
    }, [start, end]);

    return (
        <video
            ref={videoRef}
            src={url}
            className="max-w-96 max-h-96"
            controls
            preload="metadata"
            muted
        />
    );
}

export function WebcamBackground() {
    const videoRef = useRef<HTMLVideoElement>(null);
    const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
    const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
    const [isSelectVisible, setIsSelectVisible] = useState<boolean>(true);

    // Load available devices on component mount
    useEffect(() => {
        const loadDevices = async () => {
            try {
                // Request permission first to get device labels
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                stream.getTracks().forEach(track => track.stop()); // Stop the temporary stream

                const allDevices = await navigator.mediaDevices.enumerateDevices();
                const videoDevices = allDevices.filter(device => device.kind === 'videoinput');

                setDevices(videoDevices);

                // Set iPhone Camera or first device as default
                if (videoDevices.length > 0 && !selectedDeviceId) {
                    setSelectedDeviceId((videoDevices.find(d => d.label.includes("iPhone Camera")) ?? videoDevices[0]).deviceId);
                }

                console.log('Available video input devices:');
                videoDevices.forEach((device, index) => {
                    console.log(`${index + 1}. ${device.label || `Camera ${index + 1}`} (${device.deviceId})`);
                });
            } catch (error) {
                console.error('Error loading devices:', error);
            }
        };

        loadDevices();
    }, [selectedDeviceId]);

    // Add keyboard event listener for 'c' key toggle
    useEffect(() => {
        const handleKeyPress = (event: KeyboardEvent) => {
            if (event.key === 'c' || event.key === 'C') {
                setIsSelectVisible(prev => !prev);
            }
        };

        window.addEventListener('keydown', handleKeyPress);

        return () => {
            window.removeEventListener('keydown', handleKeyPress);
        };
    }, []);

    // Start webcam when device is selected
    useEffect(() => {
        if (!selectedDeviceId) return;

        const startWebcam = async () => {
            try {
                const constraints = {
                    video: { deviceId: { exact: selectedDeviceId } },
                    audio: false,
                };

                const stream = await navigator.mediaDevices.getUserMedia(constraints);

                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                }
            } catch (error) {
                console.error('Error accessing webcam:', error);
            }
        };

        startWebcam();

        // Cleanup function to stop the stream
        return () => {
            const video = videoRef.current;
            if (video?.srcObject) {
                const stream = video.srcObject as MediaStream;
                stream.getTracks().forEach(track => track.stop());
            }
        };
    }, [selectedDeviceId]);

    return (
        <div className="absolute inset-0 bg-black">
            {/* Device Selection Dropdown */}
            <Dialog open={isSelectVisible} onOpenChange={setIsSelectVisible}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Camera Settings</DialogTitle>
                    </DialogHeader>
                    <div>
                        <Select value={selectedDeviceId} onValueChange={setSelectedDeviceId}>
                            <SelectTrigger className="bg-background">
                                <SelectValue placeholder="Select a camera" />
                            </SelectTrigger>
                            <SelectContent className="">
                                {devices.map((device) => (
                                    <SelectItem key={device.deviceId} value={device.deviceId} className="hover:bg-white/20 focus:bg-white/20">
                                        {device.label || `Camera ${devices.indexOf(device) + 1}`}
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>
                </DialogContent>
            </Dialog>

            {/* Webcam Video */}
            <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
            />
        </div>
    );
}
