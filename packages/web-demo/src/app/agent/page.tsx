import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";

export default function AgentPage() {
    return (
        <div className="h-full">
            <ResizablePanelGroup direction="horizontal">
                <ResizablePanel defaultSize={75} className="py-6 px-4">
                    <h1 className="scroll-m-20 text-3xl font-semibold tracking-tight">Memories</h1>
                </ResizablePanel>
                <ResizableHandle />
                <ResizablePanel defaultSize={25} className="py-6 px-4">
                    <h1 className="scroll-m-20 text-3xl font-semibold tracking-tight">Agent</h1>
                </ResizablePanel>
            </ResizablePanelGroup>
        </div>
    )
}