import "server-only";
import type { RetrievedSource, SearchHit, Turn } from "../types";
import { bm25Score, tokenize } from "./bm25";
import { dot, embed } from "./embed";
import { loadIndex, type IndexedChunk } from "./store";

const RRF_K = 60;
const POOL = 20;
const DENSE_FLOOR = 0.3;
const MAX_CHUNKS_PER_DOC = 4;
const GUARANTEED_PER_QUERY = 2;

export interface ChartProfile {
  conditions: string[];
  medications: string[];
  allergies: string[];
}

type Scope = "patient" | "kb";

interface Scored {
  chunk: IndexedChunk;
  dense: number;
  bm25: number;
  fused: number;
}

interface SearchOptions {
  scope: Scope;
  patientId?: string | null;
  k: number;
  excludeKinds?: string[];
}

function inScope(chunk: IndexedChunk, opts: SearchOptions): boolean {
  if (opts.excludeKinds?.includes(chunk.kind)) return false;
  if (opts.scope === "kb") return chunk.scope === "kb";
  return chunk.scope === "patient" && !!opts.patientId && chunk.patientId === opts.patientId;
}

async function searchOne(
  query: string,
  qvec: Float32Array,
  opts: SearchOptions,
  index: Awaited<ReturnType<typeof loadIndex>>,
): Promise<Scored[]> {
  const qTokens = tokenize(query);
  const candidates: { i: number; dense: number; bm25: number }[] = [];
  index.chunks.forEach((chunk, i) => {
    if (!inScope(chunk, opts)) return;
    candidates.push({ i, dense: dot(qvec, chunk.vec), bm25: bm25Score(index.bm25, qTokens, i) });
  });

  const denseRank = new Map<number, number>();
  [...candidates]
    .sort((a, b) => b.dense - a.dense)
    .slice(0, POOL)
    .forEach((c, r) => denseRank.set(c.i, r + 1));
  const bm25Rank = new Map<number, number>();
  [...candidates]
    .filter((c) => c.bm25 > 0)
    .sort((a, b) => b.bm25 - a.bm25)
    .slice(0, POOL)
    .forEach((c, r) => bm25Rank.set(c.i, r + 1));

  const scored: Scored[] = [];
  for (const c of candidates) {
    const dr = denseRank.get(c.i);
    const br = bm25Rank.get(c.i);
    if (!dr && !br) continue;
    const relevant = c.dense >= DENSE_FLOOR || (br !== undefined && br <= 3);
    if (!relevant) continue;
    const fused = (dr ? 1 / (RRF_K + dr) : 0) + (br ? 1 / (RRF_K + br) : 0);
    scored.push({ chunk: index.chunks[c.i], dense: c.dense, bm25: c.bm25, fused });
  }
  return scored.sort((a, b) => b.fused - a.fused).slice(0, opts.k);
}

export async function search(
  query: string,
  opts: { patientId?: string | null; scopes: Scope[]; k: number },
): Promise<SearchHit[]> {
  const [index, [qvec]] = await Promise.all([loadIndex(), embed([query])]);
  const all: Scored[] = [];
  for (const scope of opts.scopes) {
    all.push(...(await searchOne(query, qvec, { scope, patientId: opts.patientId, k: opts.k, excludeKinds: ["profile"] }, index)));
  }
  return all
    .sort((a, b) => b.fused - a.fused)
    .slice(0, opts.k)
    .map((s, i) => ({ ...toSource(s), rank: i + 1 }));
}

function toSource(s: Scored): Omit<RetrievedSource, "label"> {
  return {
    chunkId: s.chunk.id,
    docId: s.chunk.documentId,
    title: s.chunk.title,
    heading: s.chunk.heading,
    kind: s.chunk.kind,
    scope: s.chunk.scope,
    text: s.chunk.text,
    dense: round(s.dense),
    bm25: round(s.bm25),
    fused: round(s.fused, 5),
  };
}

const round = (n: number, d = 3) => Math.round(n * 10 ** d) / 10 ** d;

export function buildQueries(turns: Turn[], profile?: ChartProfile, maxQueries = 24): string[] {
  const queries: string[] = [];
  const patientText = turns.filter((t) => t.speaker === "patient").map((t) => t.text).join(" ");
  if (patientText.trim()) queries.push(patientText.slice(0, 600));

  if (profile && patientText.trim()) {
    const complaint = patientText.slice(0, 120);
    const items = [
      ...profile.medications.map((m) => `taking ${m.replace(/\s*\(.*\)\s*$/, "")}`),
      ...profile.conditions.map((c) => `has ${c}`),
      ...profile.allergies.filter((a) => !/^no known/i.test(a)).map((a) => `allergic to ${a}`),
    ];
    for (const item of items) queries.push(`${complaint} Patient ${item}.`);
  }

  const windows: string[] = [];
  let i = 0;
  while (i < turns.length) {
    let text = "";
    let j = i;
    while (j < turns.length && text.length < 320) {
      text += (text ? " " : "") + turns[j].text;
      j++;
    }
    windows.push(text.slice(0, 600));
    const advance = Math.max(1, Math.floor((j - i) / 2));
    if (j >= turns.length) break;
    i += advance;
  }
  
  const budget = Math.max(4, maxQueries - queries.length);
  if (windows.length <= budget) return [...queries, ...windows];
  const picked: string[] = [];
  for (let n = 0; n < budget; n++) picked.push(windows[Math.round((n * (windows.length - 1)) / (budget - 1))]);
  return [...queries, ...picked];
}

interface Aggregate {
  chunk: IndexedChunk;
  dense: number;
  bm25: number;
  fused: number;
}

const FREQUENCY_BONUS = 0.15;

async function multiQuery(
  queries: string[],
  vectors: Float32Array[],
  opts: SearchOptions,
  index: Awaited<ReturnType<typeof loadIndex>>,
  guaranteeFirst: number,
): Promise<Aggregate[]> {
  const merged = new Map<number, Aggregate & { best: number; total: number }>();
  const perQuery: Scored[][] = [];
  for (let q = 0; q < queries.length; q++) {
    const hits = await searchOne(queries[q], vectors[q], { ...opts, k: 6 }, index);
    perQuery.push(hits);
    for (const s of hits) {
      const prev = merged.get(s.chunk.id);
      if (prev) {
        prev.best = Math.max(prev.best, s.fused);
        prev.total += s.fused;
        prev.dense = Math.max(prev.dense, s.dense);
        prev.bm25 = Math.max(prev.bm25, s.bm25);
      } else {
        merged.set(s.chunk.id, { ...s, best: s.fused, total: s.fused });
      }
    }
  }
  for (const a of merged.values()) a.fused = a.best + FREQUENCY_BONUS * (a.total - a.best);
  const perDoc = new Map<string, number>();
  const chosen = new Set<number>();
  const selected: Aggregate[] = [];
  const take = (a: Aggregate) => {
    const n = perDoc.get(a.chunk.documentId) ?? 0;
    if (chosen.has(a.chunk.id) || n >= MAX_CHUNKS_PER_DOC) return;
    perDoc.set(a.chunk.documentId, n + 1);
    chosen.add(a.chunk.id);
    selected.push(a);
  };
  for (let q = 0; q < guaranteeFirst && q < perQuery.length; q++) {
    for (const hit of perQuery[q].slice(0, GUARANTEED_PER_QUERY)) take(merged.get(hit.chunk.id)!);
  }
  for (const a of [...merged.values()].sort((x, y) => y.fused - x.fused)) {
    if (selected.length >= opts.k) break;
    take(a);
  }
  return selected.sort((x, y) => y.fused - x.fused);
}

export interface RetrievalResult {
  sources: RetrievedSource[];
  queries: string[];
}

interface RetrieveOptions {
  patientId: string;
  chartK?: number;
  kbK?: number;
  guaranteeFirst?: number;
}

export async function retrieveContext(queries: string[], opts: RetrieveOptions): Promise<RetrievalResult> {
  const cleaned = queries.map((q) => q.trim()).filter(Boolean);
  const index = await loadIndex();
  const sources: RetrievedSource[] = [];

  const profile = index.chunks.filter(
    (c) => c.scope === "patient" && c.patientId === opts.patientId && c.kind === "profile",
  );
  if (profile.length) {
    sources.push({
      label: "C1",
      chunkId: profile[0].id,
      docId: profile[0].documentId,
      title: profile[0].title,
      heading: "",
      kind: "profile",
      scope: "patient",
      text: profile.map((c) => (c.heading ? `${c.heading}:\n${c.text}` : c.text)).join("\n\n"),
      dense: 1,
      bm25: 0,
      fused: 0,
      pinned: true,
    });
  }

  if (cleaned.length) {
    const vectors = await embed(cleaned);
    const g = opts.guaranteeFirst ?? 0;
    const chart = await multiQuery(cleaned, vectors, { scope: "patient", patientId: opts.patientId, k: opts.chartK ?? 8, excludeKinds: ["profile"] }, index, g);
    const kb = await multiQuery(cleaned, vectors, { scope: "kb", k: opts.kbK ?? 12 }, index, g);
    let c = sources.length;
    for (const a of chart) sources.push({ label: `C${++c}`, ...toSource(a) });
    let k = 0;
    for (const a of kb) sources.push({ label: `K${++k}`, ...toSource(a) });
  }
  return { sources, queries: cleaned };
}

export function retrieveForTranscript(
  turns: Turn[],
  patient: ChartProfile & { id: string },
  plannedQueries: string[] = [],
): Promise<RetrievalResult> {
  return retrieveContext([...plannedQueries, ...buildQueries(turns, patient)], {
    patientId: patient.id,
    guaranteeFirst: plannedQueries.length,
  });
}

export function retrieveForQuestion(question: string, patientId: string): Promise<RetrievalResult> {
  return retrieveContext([question], { patientId, chartK: 6, kbK: 4 });
}