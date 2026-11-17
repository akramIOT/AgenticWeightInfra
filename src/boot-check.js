// Runs even when the module graph fails to load. app.js sets data-awi once it starts.
const timer = setTimeout(() => {
  if (document.documentElement.dataset.awi === "1") return;
  const n = document.getElementById("boot");
  if (!n) return;
  n.dataset.kind = "slow";
  n.hidden = false;
  n.textContent = "AgenticWeightInfra did not start. This page needs JavaScript modules and the WebLLM package. Reload, and allow this host to load scripts from esm.run.";
}, 12000);

const watch = new MutationObserver(() => {
  if (document.documentElement.dataset.awi !== "1") return;
  clearTimeout(timer);
  const n = document.getElementById("boot");
  if (n?.dataset.kind === "slow") n.hidden = true;
  watch.disconnect();
});
watch.observe(document.documentElement, { attributes: true, attributeFilter: ["data-awi"] });
