// Local audit log and program registry. Both stay in the browser storage the
// caller passes in. Nothing is uploaded.

function storageError(e) {
  const full = e && (e.name === "QuotaExceededError" || e.code === 22 || e.code === 1014);
  return new Error(full
    ? "This browser refused to store more local records."
    : "This browser refused to store local records.");
}

const AUDIT_KEY = "awi.audit.v1";
const REGISTRY_KEY = "awi.registry.v1";
const AUDIT_LIMIT = 200;

export function createAuditLog(storage, { key = AUDIT_KEY, limit = AUDIT_LIMIT } = {}) {
  const read = () => {
    try { const v = JSON.parse(storage.getItem(key) || "[]"); return Array.isArray(v) ? v : []; }
    catch { return []; }
  };
  const write = (rows) => {
    try { storage.setItem(key, JSON.stringify(rows.slice(-limit))); }
    catch (e) { throw storageError(e); }
  };
  return {
    list() { return read(); },
    record(entry) {
      const rows = read();
      const row = { id: `${Date.now().toString(36)}-${rows.length}`, at: new Date().toISOString(), ...entry };
      rows.push(row);
      write(rows);
      return row;
    },
    clear(meta = {}) {
      write([]);
      return this.record({ action: "clear", decision: "allow", blocked: false, ...meta });
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
  return {
    list() { return read(); },
    save(entry) {
      const name = String(entry.name ?? "").trim();
      if (!name) throw new Error("Name the program before saving it.");
      const rows = read().filter((r) => r.name !== name);
      const row = { name, savedAt: new Date().toISOString(), modelId: entry.modelId, program: entry.program, programHash: entry.programHash, decision: entry.decision };
      rows.push(row);
      write(rows);
      return row;
    },
    get(name) { return read().find((r) => r.name === name) ?? null; },
  };
}
