import 'server-only';
import fs from 'node:fs';
import path from 'node:path';
import type { DatabaseSync } from 'node:sqlite';

const { DatabaseSync: Database } = process.getBuiltinModule("node:sqlite");
const globalForDb = globalThis as unknown as { __notetakerDb?: DatabaseSync };

const SCHEMA = `
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    dob TEXT NOT NULL,
    sex TEXT NOT NULL,
    allergies TEXT NOT NULL DEFAULT '[]',
    medications TEXT NOT NULL DEFAULT '[]',
    conditions TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL CHECK (scope IN ('patient', 'kb')),
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_patient ON documents(patient_id);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    scope TEXT NOT NULL,
    patient_id TEXT,
    ord INTEGER NOT NULL,
    heading TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

CREATE TABLE IF NOT EXISTS encounters (
  id TEXT PRIMARY KEY,
  patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'draft',
  transcript TEXT NOT NULL DEFAULT '[]',
  note TEXT,
  sources TEXT NOT NULL DEFAULT '[]',
  queries TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  signed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_encounters_patient ON encounters(patient_id);
`;

export function db(): DatabaseSync {
    if (!globalForDb.__notetakerDb) {
        const file = process.env.DATABASE_PATH
            ? path.resolve(process.env.DATABASE_PATH)    
            : path.join(process.cwd(), "storage", "notetaker.db");
        fs.mkdirSync(path.dirname(file), { recursive: true });
        const conn = new Database(file);
        conn.exec("PRAGMA journal_mode = WAL; PRAGMA foreign_keys = ON;");
        conn.exec(SCHEMA);
        globalForDb.__notetakerDb = conn;
    }
    return globalForDb.__notetakerDb;
}

export function getMeta(key: string): string | null {
    const row = db().prepare("SELECT value FROM meta WHERE key = ?").get(key) as
        | { value: string }
        | undefined;
    return row?.value ?? null;
}

export function setMeta(key: string, value: string): void {
    db()
        .prepare("INSERT meta (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value = excluded.value")
        .run(key, value);
}

export function transaction<T>(fn: () => T): T {
    const conn = db();
    conn.exec("BEGIN");
    try {
        const result = fn();
        conn.exec("COMMIT");
        return result;
    } catch (err) {
        conn.exec("ROLLBACK");
        throw err;
    }
}