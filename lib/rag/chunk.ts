export interface Chunk {
    heading: string;
    text: string;
}

interface Options {
    maxChars?: number;
}

export function chunkText(content: string, { maxChars = 700 }: Options = {}): Chunk[] {
    const blocks = content
        .replace(/\r\n/g, "\n")
        .split(/\n{2,}/)
        .map((b) => b.trim())
        .filter(Boolean);

    const chunks: Chunk[] = [];
    let heading = "";
    let buffer: string[] = [];
    let size = 0;

    const flush = () => {
        if (buffer.length) chunks.push({ heading, text: buffer.join("\n\n") });
        buffer = [];
        size = 0;
    };

    for (const block of blocks) {
        const headingMatch = /^#+\s*(.+)/.exec(block.split("\n")[0]);
        if (headingMatch) {
            flush();
            heading = headingMatch[1].trim();
            const rest = block.split("\n").slice(1).join("\n").trim();
            if (!rest) continue;
            pushBlock(rest);
            continue;
        }
    }
    flush();
    return chunks;

    function pushBlock(block: string) {
        if (block.length > maxChars) {
            flush();
            for (const piece of splitLong(block, maxChars)) chunks.push({ heading, text: piece });
            return;
        }
        if (size + block.length > maxChars) flush();
        buffer.push(block);
        size += block.length + 2;
    } 
}

function splitLong(text: string, maxChars: number): string[] {
    const sentences = text.match(/[^.!?]+[.!?]+/g) ?? [text];
    const pieces: string[] = [];
    let current = "";
    let lastSentence = "";

    for (const s of sentences) {
        if (current && current.length + s.length > maxChars) {
            pieces.push(current.trim());
            current = lastSentence;
        }
        current += s;
        lastSentence = s;
    }
    if (current.trim()) pieces.push(current.trim());
    return pieces;
}

export function embeddingInput(title: string, chunk: Chunk): string {
    return [title, chunk.heading, chunk.text].filter(Boolean).join("\n");
}