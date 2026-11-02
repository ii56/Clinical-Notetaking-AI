const STOPWORDS = new Set(
  `a an and are as at be been but by can did do does for from had has have he her his how i if in into is
  it its me my no not of on or our she so than that the their them then there these they this to too was we
  were what when where which who why will with would you your about also just very`
    .split(/\s+/)
    .filter(Boolean),
);


function stem(token: string): string {
  if (token.length <= 4) return token;
  if (token.endsWith("ies")) return token.slice(0, -3) + "y";
  if (token.endsWith("sses")) return token.slice(0, -2);
  if (token.endsWith("es") && token.length > 5) return token.slice(0, -2);
  if (token.endsWith("s") && !token.endsWith("ss")) return token.slice(0, -1);
  if (token.endsWith("ing") && token.length > 6) return token.slice(0, -3);
  if (token.endsWith("ed") && token.length > 5) return token.slice(0, -2);
  return token;
}

export function tokenize(text: string): string[] {
  const out: string[] = [];
  const matches = text.toLowerCase().match(/[a-z0-9]+(?:[.\-/][a-z0-9]+)*/g) ?? [];
  for (const m of matches) {
    const parts = m.split(/[.\-/]/).filter(Boolean);
    if (parts.length > 1) out.push(m);
    for (const p of parts) {
      if (!STOPWORDS.has(p) && p.length > 1) out.push(stem(p));
    }
  }
  return out;
}

export interface Bm25Index {
  docs: Map<string, number>[];
  lengths: number[];
  avgLength: number;
  df: Map<string, number>;
}

export function buildBm25(texts: string[]): Bm25Index {
  const docs: Map<string, number>[] = [];
  const lengths: number[] = [];
  const df = new Map<string, number>();
  for (const text of texts) {
    const tf = new Map<string, number>();
    const tokens = tokenize(text);
    for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
    for (const t of tf.keys()) df.set(t, (df.get(t) ?? 0) + 1);
    docs.push(tf);
    lengths.push(tokens.length);
  }
  const avgLength = lengths.reduce((a, b) => a + b, 0) / Math.max(1, lengths.length);
  return { docs, lengths, avgLength, df };
}

export function bm25Score(index: Bm25Index, queryTokens: string[], docIdx: number, k1 = 1.2, b = 0.75): number {
  const tf = index.docs[docIdx];
  const n = index.docs.length;
  let score = 0;
  for (const term of new Set(queryTokens)) {
    const f = tf.get(term);
    if (!f) continue;
    const df = index.df.get(term) ?? 0;
    const idf = Math.log(1 + (n - df + 0.5) / (df + 0.5));
    const norm = f + k1 * (1 - b + (b * index.lengths[docIdx]) / index.avgLength);
    score += idf * ((f * (k1 + 1)) / norm);
  }
  return score;
}