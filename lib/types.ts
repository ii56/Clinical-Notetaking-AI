export type Speaker = "clinical" | "patient" | "other";

export interface Turn {
    id: number;
    speaker: Speaker;
    text: string;
    start?: number;
    end?: number;
}

export interface Patient {
    id: string;
    name: string;
    dob: string;
    sex: string;
    allergies: string[];
    medications: string[];
    conditions: string[];
    created_at: string;
}

export interface RetrievedSource {
    label: string;
    chunkId: number;
    docId: string;
    title: string;
    heading: string;
    kind: string;
    
}