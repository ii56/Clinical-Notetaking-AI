import { extend } from "zod/mini";

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
    scope: "patient" | "kb";
    text: string;
    dense: number;
    bm25: number;
    fused: number;
    pinned?: boolean;
}

export interface Claim {
    text: string;
    evidence_turns: number[];
    chart_sources: string[];
    knowledge_sources: string[];
    grounded?: boolean;
    manual?: boolean;
}

export interface Alert {
    severity: "info" | "caution" | "warning";
    message: string;
    evidence_turns: number[];
    chart_sources: string[];
    knowledge_sources: string[];
}

export interface SuggestedCode {
    code: string;
    description: string;
    rationale: string;
    knowledge_sources: string[];
}

export interface NoteDraft {
    subjective: Claim[];
    objective: Claim[];
    assessment: Claim[];
    plan: Claim[];
    alerts: Claim[];
    suggested_code: SuggestedCode[];
    open_questions: string[];
    warnings?: string[];
    model?: string;
    generated_at?: string;
}

export const SOAP_SECTIONS = ["subjective", "objective", "assessment", "plan"] as const;
export type SoapSection = (typeof SOAP_SECTIONS)[number];

export interface Encounter {
    id: string;
    patient_id: string;
    title: string;
    status: "draft" | "final";    
    transcript: Turn[];
    note: NoteDraft | null;
    sources: RetrievedSource[];
    queries: string[];
    created_at: string;
    updated_at: string;
    signed_at: string | null;
}

export interface KbDocument {
    id: string;
    title: string;
    kind: string;
    content: string;
    chunk_count: number;
    created_at: string;
}

export interface SearchHit extends Omit<RetrievedSource, "label"> {
    rank: number;
}

export interface AskResult {
    answer: string;
    sources: RetrievedSource[];
    cited: string[];
}