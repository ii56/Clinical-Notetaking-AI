import "server-only";
import type { FeatureExtractionPipeline } from "@huggingface/transformers";

export const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL ?? "Xenova/all-MiniLM-L6-v2";

const globalForEmbed = globalThis as unknown as {
  __extractor?: Promise<FeatureExtractionPipeline>;
};

function getExtractor(): Promise<FeatureExtractionPipeline> {
  if (!globalForEmbed.__extractor) {
    globalForEmbed.__extractor = (async () => {
      const { pipeline } = await import("@huggingface/transformers");
      return (await pipeline("feature-extraction", EMBEDDING_MODEL, {
        dtype: "q8",
      })) as FeatureExtractionPipeline;
    })().catch((err) => {
      globalForEmbed.__extractor = undefined;
      throw err;
    });
  }
  return globalForEmbed.__extractor;
}

export async function embed(texts: string[]): Promise<Float32Array[]> {
  if (texts.length === 0) return [];
  const extractor = await getExtractor();
  const out: Float32Array[] = [];
  const batchSize = 16;
  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const tensor = await extractor(batch, { pooling: "mean", normalize: true });
    const dim = tensor.dims[1];
    const data = tensor.data as Float32Array;
    for (let j = 0; j < batch.length; j++) {
      out.push(Float32Array.from(data.subarray(j * dim, (j + 1) * dim)));
    }
  }
  return out;
}

export function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function toBlob(v: Float32Array): Uint8Array {
  return new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
}

export function fromBlob(blob: Uint8Array): Float32Array {
  const copy = Uint8Array.from(blob);
  return new Float32Array(copy.buffer, 0, copy.byteLength / 4);
}