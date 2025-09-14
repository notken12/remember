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

export default function DemoPage() {
    const [messages, setMessages] = useState<UIMessage[]>([])
    const streamRef = useRef<ReadableStream<UIMessageChunk> | null>(null)
    const sseRef = useRef<EventSource | null>(null)
    const isCleaningUpRef = useRef(false)

    useEffect(() => {
        console.log('loading sse')
        const sse = new EventSource(process.env.NEXT_PUBLIC_BACKEND_URL! + "/web_stream");
        sseRef.current = sse;

        sse.onopen = () => {
            console.log('SSE connection opened');

            // Create the ReadableStream and store reference to prevent garbage collection
            const stream = new ReadableStream<UIMessageChunk>({
                start(controller) {
                    sse.onmessage = (event) => {
                        console.log('SSE message received:', event)
                        try {
                            // Parse the event data and enqueue it
                            const data = JSON.parse(event.data);
                            controller.enqueue(data);
                        } catch (error) {
                            console.error('Error parsing SSE data:', error);
                        }
                    };

                    sse.onerror = (error) => {
                        console.error('SSE error:', error);
                        controller.error(error);
                    };
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
            processUIMessageStream(stream);
        };

        sse.onerror = (error) => {
            console.error('SSE connection error:', error);
        };

        const processUIMessageStream = async (stream: ReadableStream<UIMessageChunk>) => {
            try {
                for await (const uiMessage of readUIMessageStream({ stream })) {
                    console.log('UI Message received:', uiMessage)
                    setMessages(produce(prev => {
                        // Avoid duplicates by checking if message already exists
                        const index = prev.findIndex(m => m.id === uiMessage.id);
                        if (index == -1) {
                            // @ts-expect-error deep
                            prev.push(uiMessage)
                        } else {
                            prev[index] = uiMessage
                        }
                        return prev
                    }));
                }
            } catch (error) {
                console.error('Error processing UI message stream:', error);
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
            <ol className="m-4 max-h-full overflow-y-auto">
                {messages.map((msg, msgIndex) => {
                    return (
                        <li
                            key={msg.id || `msg-${msgIndex}`}
                            className={cn(
                                "p-3 rounded-lg backdrop-blur-sm",
                                {
                                    "bg-blue-500/80 text-white ml-8": msg.role === "user",
                                    "bg-white/80 text-black mr-8": msg.role === "assistant"
                                }
                            )}
                        >
                            {msg.parts.map((part, partIndex) => {
                                switch (part.type) {
                                    case "text":
                                        return <p key={partIndex} className="whitespace-pre-wrap">{part.text}</p>
                                    default:
                                        return <div key={partIndex} className="text-xs opacity-70">{JSON.stringify(part)}</div>
                                }
                            })}
                        </li>
                    )
                })}
            </ol>
        </div>
    )
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
