// The weight map is one row per layer and one column per tensor kind.
// Column patterns match the MLC names these models actually ship:
// Qwen uses c_attn (with bias); Llama and SmolLM use qkv_proj (no bias).
// The final model.norm is its own row, because a full-model edit touches it.

export const KINDS = [
  ["attn qkv", /self_attn\.(c_attn|qkv_proj)\.weight$/],
  ["qkv bias", /self_attn\.(c_attn|qkv_proj)\.bias$/],
  ["attn out", /self_attn\.o_proj\.weight$/],
  ["mlp in", /mlp\.gate_up_proj\.weight$/],
  ["mlp out", /mlp\.down_proj\.weight$/],
  ["norm 1", /input_layernorm\.weight$/],
  ["norm 2", /post_attention_layernorm\.weight$/],
];

/** Tensor list for a model before its shards are downloaded. */
export function schematicInfo(setup) {
  const spec = setup.layout;
  const tensors = new Map();
  const add = (name) => tensors.set(name, { name, shape: [8], dtype: "float16", format: "f16" });
  add("model.embed_tokens.weight");
  add("model.norm.weight");
  for (let l = 0; l < spec.layers; l++) {
    const p = `model.layers.${l}`;
    add(`${p}.self_attn.${spec.attn}.weight`);
    if (spec.bias) add(`${p}.self_attn.${spec.attn}.bias`);
    add(`${p}.self_attn.o_proj.weight`);
    add(`${p}.mlp.gate_up_proj.weight`);
    add(`${p}.mlp.down_proj.weight`);
    add(`${p}.input_layernorm.weight`);
    add(`${p}.post_attention_layernorm.weight`);
  }
  return {
    schematic: true,
    tensors,
    config: {
      model_config: {
        num_hidden_layers: spec.layers,
        num_attention_heads: spec.heads,
        num_key_value_heads: spec.kv,
        head_dim: spec.headDim,
        hidden_size: spec.hidden,
      },
    },
  };
}

/**
 * Place every tensor into the grid. `uncovered` is the planned tensors that
 * have no cell, so a full-model edit cannot disappear from the picture.
 */
export function mapRows(info, plan = new Map()) {
  const names = [...info.tensors.keys()];
  const used = new Set();
  const take = (name) => {
    if (!name) return null;
    used.add(name);
    const ops = plan.get(name) ?? null;
    return { name, ops };
  };
  const find = (pred) => names.find(pred) ?? null;
  const rows = [];
  rows.push({
    label: "embed",
    cells: pad([take(find((n) => /embed_tokens\.weight$/.test(n)))]),
  });
  const n = info.config.model_config.num_hidden_layers;
  for (let l = 0; l < n; l++) {
    rows.push({
      label: `L${l}`,
      cells: KINDS.map(([, re]) => take(find((n) => n.includes(`layers.${l}.`) && re.test(n)))),
    });
  }
  rows.push({
    label: "norm",
    cells: pad([take(find((n) => /(^|\.)norm\.weight$/.test(n) && !n.includes("layers.")))]),
  });
  const uncovered = [...plan.keys()].filter((n) => !used.has(n));
  return { rows, uncovered };
}

function pad(cells) {
  while (cells.length < KINDS.length) cells.push(null);
  return cells;
}
