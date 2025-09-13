'use client';

import { useEffect, useRef, useState } from 'react';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';

export default function DemoPage() {
    return (
        <div className="absolute inset-0">
            <WebcamBackground />
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
