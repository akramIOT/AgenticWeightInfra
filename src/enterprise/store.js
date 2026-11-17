// Local audit log and program registry. Both stay in the browser storage the
// caller passes in. Nothing is uploaded. Each audit row carries the digest of
// the row before it, so a rewritten record breaks the chain.

import { hashStr } from "../infra/hash.js";

function storageError(e) {
  const full = e && (e.name === "QuotaExceededError" || e.code === 22 || e.code === 1014);
  return new Error(full
    ? "This browser refused to store more local records."
    : "This browser refused to store local records.");
}

const AUDIT_KEY = "awi.audit.v1";
const REGISTRY_KEY = "awi.registry.v1";
const AUDIT_LIMIT = 200;
export const GENESIS = "00000000";

/** Stable JSON with sorted keys, so the audit digest matches the Python runtime. */
export function canonical(value) {
  if (value === undefined || value === null) return "null";
  if (typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map((item) => canonical(item)).join(",")}]`;
  const keys = Object.keys(value).filter((key) => value[key] !== undefined).sort();
  return `{${keys.map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`).join(",")}}`;
}

export function chainDigest(prev, entry) {
  const body = { ...entry };
  delete body.digest;
  return hashStr(`${prev}\n${canonical(body)}`);
}

export function createAuditLog(storage, { key = AUDIT_KEY, limit = AUDIT_LIMIT } = {}) {
  const read = () => {
    try { const v = JSON.parse(storage.getItem(key) || "[]"); return Array.isArray(v) ? v : []; }
    catch { return []; }
  };
  const write = (rows) => {
    const kept = rows.slice(-limit);
    // Dropping the head would leave the next row pointing at a missing digest.
    if (kept.length < rows.length) {
      let prev = GENESIS;
      for (const row of kept) {
        row.prev = prev;
        row.digest = chainDigest(prev, row);
        prev = row.digest;
      }
    }
    try { storage.setItem(key, JSON.stringify(kept)); }
    catch (e) { throw storageError(e); }
  };
  return {
    list() { return read(); },
    record(entry) {
      const rows = read();
      const prev = rows.length ? (rows[rows.length - 1].digest ?? "") : GENESIS;
      const row = { id: `${Date.now().toString(36)}-${rows.length}`, at: new Date().toISOString(), ...entry, prev };
      row.digest = chainDigest(prev, row);
      rows.push(row);
      write(rows);
      return row;
    },
    verify() {
      const rows = read();
      let prev = GENESIS;
      for (let i = 0; i < rows.length; i++) {
        const expect = chainDigest(prev, rows[i]);
        if (rows[i].prev !== prev || rows[i].digest !== expect) return { ok: false, brokenAt: i, head: null };
        prev = rows[i].digest;
      }
      return { ok: true, brokenAt: null, head: prev };
    },
    clear(meta = {}) {
      const rows = read();
      const severed = rows.length ? (rows[rows.length - 1].digest ?? "") : GENESIS;
      write([]);
      return this.record({ action: "clear", decision: "allow", blocked: false, severed, ...meta });
    },
  };
}

export function createRegistry(storage, { key = REGISTRY_KEY } = {}) {
  const read = () => {
    try { const v = JSON.parse(storage.getItem(key) || "[]"); return Array.isArray(v) ? v : []; }
    catch { return []; }
  };
  const write = (rows) => {
    try { storage.setItem(key, JSON.stringify(rows)); }
    catch (e) { throw storageError(e); }
  };
  const find = (name) => read().find((row) => row.name === name) ?? null;
  return {
    list() { return read(); },
    save(entry) {
      const name = String(entry.name ?? "").trim();
      if (!name) throw new Error("Name the program before saving it.");
      const rows = read().filter((row) => row.name !== name);
      const row = {
        name,
        savedAt: new Date().toISOString(),
        modelId: entry.modelId,
        program: entry.program,
        programHash: entry.programHash,
        decision: entry.decision,
        status: "draft",
        author: String(entry.author ?? "").trim(),
      };
      rows.push(row);
      write(rows);
      return row;
    },
    approve(name, approver) {
      const who = String(approver ?? "").trim();
      if (who.length < 2) throw new Error("Record the approver before promoting a program.");
      const rows = read();
      const row = rows.find((item) => item.name === name);
      if (!row) throw new Error("Save the program before approving it.");
      if (row.status === "retired") throw new Error("A retired program must be saved again before it can be approved.");
      const author = String(row.author ?? "").trim();
      if (author.length < 2) throw new Error("Record the operator as the author before a second person approves the program.");
      if (author.toLowerCase() === who.toLowerCase()) throw new Error("The approver must be a different person from the author.");
      row.status = "approved";
      row.approvedBy = who;
      row.approvedAt = new Date().toISOString();
      write(rows);
      return row;
    },
    retire(name) {
      const rows = read();
      const row = rows.find((item) => item.name === name);
      if (!row) throw new Error("Save the program before retiring it.");
      row.status = "retired";
      delete row.approvedBy;
      delete row.approvedAt;
      write(rows);
      return row;
    },
    get(name) { return find(name); },
  };
}
