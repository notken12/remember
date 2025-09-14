'use client';

import { useState, useRef, useEffect } from "react";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface Video {
    id: string;
    video_path: string;
    videoURL: string;
    time_created: string;
    title?: string;
    annotation: string;
}

interface VideoGridProps {
    videos: Video[];
}

function VideoThumbnail({ video, onClick }: { video: Video; onClick: () => void }) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const [isLoaded, setIsLoaded] = useState(false);

    useEffect(() => {
        const videoElement = videoRef.current;
        if (!videoElement) return;

        const handleLoadedMetadata = () => {
            // Seek to 1 second or 10% of duration, whichever is smaller
            const seekTime = Math.min(1, videoElement.duration * 0.1);
            videoElement.currentTime = seekTime;
        };

        const handleSeeked = () => {
            setIsLoaded(true);
        };

        const handleError = () => {
            console.error('Video loading error for:', video.videoURL);
            setIsLoaded(false);
        };

        videoElement.addEventListener('loadedmetadata', handleLoadedMetadata);
        videoElement.addEventListener('seeked', handleSeeked);
        videoElement.addEventListener('error', handleError);

        return () => {
            videoElement.removeEventListener('loadedmetadata', handleLoadedMetadata);
            videoElement.removeEventListener('seeked', handleSeeked);
            videoElement.removeEventListener('error', handleError);
        };
    }, [video.videoURL]);

    return (
        <div
            className="group cursor-pointer rounded-lg overflow-hidden shadow-md shadow-gray-300 hover:shadow-lg hover:shadow-gray-400 transition-shadow duration-200"
            onClick={onClick}
        >
            <div className="relative aspect-video">
                <video
                    ref={videoRef}
                    src={video.videoURL}
                    className={`w-full h-full object-cover group-hover:scale-105 transition-transform duration-200 opacity-100
                        `}
                    muted
                    preload="metadata"
                />
                {/* {!isLoaded && (
                    <div className="absolute inset-0 bg-[#ddd] flex items-center justify-center">
                        <div className="w-8 h-8 border-2 border-gray-400 border-t-transparent rounded-full animate-spin"></div>
                    </div>
                )} */}
                <div className="absolute inset-0 group-hover:bg-opacity-20 transition-all duration-200 flex items-center justify-center">
                    <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                        <svg className="w-12 h-12 text-white" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                        </svg>
                    </div>
                </div>
            </div>
            <div className="p-2">
                <p className="text-sm text-gray-600 truncate">
                    {video.annotation}
                </p>
            </div>
        </div>
    );
}

export default function VideoGrid({ videos }: VideoGridProps) {
    const [selectedVideo, setSelectedVideo] = useState<Video | null>(null);
    const [isDialogOpen, setIsDialogOpen] = useState(false);

    const handleThumbnailClick = (video: Video) => {
        setSelectedVideo(video);
        setIsDialogOpen(true);
    };

    return (
        <>
            {/* Video Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                {videos.map(video => (
                    <VideoThumbnail
                        key={video.videoURL}
                        video={video}
                        onClick={() => handleThumbnailClick(video)}
                    />
                ))}
            </div>

            {/* Video Dialog */}
            <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
                <DialogContent className="max-w-4xl w-full">
                    <DialogHeader>
                        <DialogTitle>
                            Memory from {selectedVideo ? new Date(selectedVideo.time_created).toLocaleDateString() : ''}
                        </DialogTitle>
                    </DialogHeader>
                    {selectedVideo && (
                        <div className="aspect-video w-full">
                            <video
                                src={selectedVideo.videoURL}
                                controls
                                autoPlay
                                className="w-full h-full rounded-md"
                            />
                        </div>
                    )}
                    <DialogFooter>
                        <p className="text-xs">
                            {selectedVideo?.annotation}
                        </p>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </>
    );
}
