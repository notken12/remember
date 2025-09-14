import { createClient } from "@supabase/supabase-js";
import VideoGrid from "@/components/VideoGrid";

interface Video {
    id: string;
    video_path: string;
    videoURL: string;
    time_created: string;
    title?: string;
    description?: string;
}

export default async function AgentPage() {
    const supabase = createClient(process.env.SUPABASE_URL!, process.env.SUPABASE_API_KEY!)
    const { data } = await supabase.from('videos').select('*').throwOnError();
    const videos: Video[] = data.map(v => {
        return {
            ...v,
            videoURL: supabase.storage.from('videos').getPublicUrl(v.video_path).data.publicUrl
        }
    })

    return (
        <div className="h-full container mx-auto p-6">
            <h1 className="scroll-m-20 text-3xl font-semibold tracking-tight mb-8">Memories</h1>
            <VideoGrid videos={videos} />
        </div>
    )
}