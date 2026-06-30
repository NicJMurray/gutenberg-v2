import { createHash } from "node:crypto";
import { mkdir, readdir, readFile, rm, writeFile, copyFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = path.join(root, "data", "frequencies.csv");
const outputDir = path.join(root, "public", "data");

await mkdir(outputDir, { recursive: true });

for (const file of await readdir(outputDir)) {
  if (/^frequencies-[a-f0-9]{12}\.csv$/.test(file) || file === "manifest.json") {
    await rm(path.join(outputDir, file));
  }
}

const csv = await readFile(source);
const hash = createHash("sha256").update(csv).digest("hex").slice(0, 12);
const fileName = `frequencies-${hash}.csv`;
const target = path.join(outputDir, fileName);
const text = csv.toString("utf8");
const rows = Math.max(0, text.trimEnd().split(/\r?\n/).length - 1);

await copyFile(source, target);
await writeFile(
  path.join(outputDir, "manifest.json"),
  JSON.stringify(
    {
      file: fileName,
      rows,
      bytes: csv.byteLength,
      hash,
      generatedAt: new Date().toISOString(),
    },
    null,
    2,
  ),
);

console.log(`Prepared ${fileName} (${rows.toLocaleString()} rows, ${csv.byteLength.toLocaleString()} bytes).`);
