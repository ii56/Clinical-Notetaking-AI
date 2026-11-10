import "server-only";
import { randomUUID } from "node:crypto";
import { db, getMeta, setMeta, transaction } from "../db";
import { buildBm25, type Bm25Index } from "./bm25";
import { chunkText, embeddingInput } from "./chunk";
import { EMBEDDING_MODEL, embed, fromBlob, toBlob } from "./embed";

export interface IndexedChunk {
  id: number;
  documentId: string;
  scope: "patient" | "kb";
  patientId: string | null;
  kind: string;
  title: string;
  heading: string;
  text: string;
  vec: Float32Array;
}

interface RagIndex {
  chunks: IndexedChunk[];
  bm25: Bm25Index;
}

const g = globalThis as unknown as { __ragIndex?: RagIndex | null; __ragReindex?: Promise<void> };

function invalidate() {
  g.__ragIndex = null;
}

export interface NewDocument {
  scope: "patient" | "kb";
  patientId?: string | null;
  kind: string;
  title: string;
  content: string;
  id?: string;
  createdAt?: string;
}

export async function addDocument(doc: NewDocument): Promise<{ id: string; chunks: number }> {
  const id = doc.id ?? randomUUID();
  const chunks = chunkText(doc.content);
  if (chunks.length === 0) throw new Error("Document has no text to index.");
  const vectors = await embed(chunks.map((c) => embeddingInput(doc.title, c)));

  transaction(() => {
    const conn = db();
    conn
      .prepare(
        "INSERT INTO documents (id, scope, patient_id, kind, title, content, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
      )
      .run(id, doc.scope, doc.patientId ?? null, doc.kind, doc.title, doc.content, doc.createdAt ?? new Date().toISOString());
    const insert = conn.prepare(
      "INSERT INTO chunks (document_id, scope, patient_id, ord, heading, text, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
    );
    chunks.forEach((c, i) => {
      insert.run(id, doc.scope, doc.patientId ?? null, i, c.heading, c.text, toBlob(vectors[i]));
    });
    setMeta("embedding_model", EMBEDDING_MODEL);
  });
  invalidate();
  return { id, chunks: chunks.length };
}

export function deleteDocument(id: string): void {
  db().prepare("DELETE FROM documents WHERE id = ?").run(id); // chunks cascade
  invalidate();
}

export async function syncPatientProfile(patient: {
  id: string;
  name: string;
  dob: string;
  sex: string;
  allergies: string[];
  medications: string[];
  conditions: string[];
}): Promise<void> {
  const list = (items: string[], empty: string) =>
    items.length ? items.map((i) => `- ${i}`).join("\n") : `- ${empty}`;
  const content = [
    `# Chart summary`,
    `${patient.name}, born ${patient.dob}, ${patient.sex}.`,
    `# Allergies`,
    list(patient.allergies, "No known drug allergies recorded"),
    `# Current medications`,
    list(patient.medications, "None recorded"),
    `# Problem list`,
    list(patient.conditions, "None recorded"),
  ].join("\n\n");

  const existing = db()
    .prepare("SELECT id FROM documents WHERE patient_id = ? AND kind = 'profile'")
    .all(patient.id) as { id: string }[];
  const added = await addDocument({
    scope: "patient",
    patientId: patient.id,
    kind: "profile",
    title: "Chart summary",
    content,
  });
  for (const row of existing) if (row.id !== added.id) deleteDocument(row.id);
}

export function listKbDocuments() {
  return db()
    .prepare(
      `SELECT d.id, d.title, d.kind, d.content, d.created_at,
              (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) AS chunk_count
       FROM documents d WHERE d.scope = 'kb' ORDER BY d.created_at DESC, d.title`,
    )
    .all() as { id: string; title: string; kind: string; content: string; created_at: string; chunk_count: number }[];
}

export function listPatientDocuments(patientId: string) {
  return db()
    .prepare(
      `SELECT id, kind, title, created_at FROM documents
       WHERE patient_id = ? AND kind != 'profile' ORDER BY created_at DESC`,
    )
    .all(patientId) as { id: string; kind: string; title: string; created_at: string }[];
}

export function countChunks(): { kb: number; patient: number } {
  const rows = db().prepare("SELECT scope, COUNT(*) AS n FROM chunks GROUP BY scope").all() as {
    scope: string;
    n: number;
  }[];
  const get = (s: string) => rows.find((r) => r.scope === s)?.n ?? 0;
  return { kb: get("kb"), patient: get("patient") };
}

async function ensureEmbeddingsCurrent(): Promise<void> {
  const stored = getMeta("embedding_model");
  if (!stored || stored === EMBEDDING_MODEL) return;
  if (!g.__ragReindex) {
    g.__ragReindex = (async () => {
      const rows = db()
        .prepare(
          `SELECT c.id, c.heading, c.text, d.title FROM chunks c JOIN documents d ON d.id = c.document_id`,
        )
        .all() as { id: number; heading: string; text: string; title: string }[];
      const vectors = await embed(rows.map((r) => embeddingInput(r.title, { heading: r.heading, text: r.text })));
      transaction(() => {
        const update = db().prepare("UPDATE chunks SET embedding = ? WHERE id = ?");
        rows.forEach((r, i) => update.run(toBlob(vectors[i]), r.id));
        setMeta("embedding_model", EMBEDDING_MODEL);
      });
      invalidate();
    })().finally(() => {
      g.__ragReindex = undefined;
    });
  }
  await g.__ragReindex;
}

export async function loadIndex(): Promise<RagIndex> {
  await ensureEmbeddingsCurrent();
  if (g.__ragIndex) return g.__ragIndex;
  const rows = db()
    .prepare(
      `SELECT c.id, c.document_id, c.scope, c.patient_id, c.heading, c.text, c.embedding, d.kind, d.title
       FROM chunks c JOIN documents d ON d.id = c.document_id ORDER BY c.id`,
    )
    .all() as {
    id: number;
    document_id: string;
    scope: "patient" | "kb";
    patient_id: string | null;
    heading: string;
    text: string;
    embedding: Uint8Array;
    kind: string;
    title: string;
  }[];
  const chunks: IndexedChunk[] = rows.map((r) => ({
    id: r.id,
    documentId: r.document_id,
    scope: r.scope,
    patientId: r.patient_id,
    kind: r.kind,
    title: r.title,
    heading: r.heading,
    text: r.text,
    vec: fromBlob(r.embedding),
  }));
 
  const bm25 = buildBm25(chunks.map((c) => embeddingInput(c.title, c)));
  g.__ragIndex = { chunks, bm25 };
  return g.__ragIndex;
}