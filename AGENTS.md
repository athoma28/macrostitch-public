# Storage and test artifacts

- Preserve source videos, recipes, delivered output images, validation JSON and
  logs. Reproducible candidates, alignment TIFFs, scan arrays and benchmark images
  are scratch data. Never weaken quality checks to save space; compute and record
  their results before deleting temporary inputs.
- Run tests with the configured pytest retention. Do not leave a persistent
  `--basetemp` tree. If a custom base is required, enclose it in a
  `tempfile.TemporaryDirectory` and clean it on success and failure.
- Use the benchmark runners' default cleanup. Enable `--keep-artifacts` or
  `--keep-frames` only when inspection needs those files, and remove them after
  review. Do not silently disable the normal 2 GiB stage-cache limit.
- New experiments and audit scripts must put bulky generated media/arrays in a
  temporary-directory context with `finally` cleanup. Persist JSON results,
  hashes, logs and necessary small review crops in `diagnostics/`; do not retain
  full TIFF stacks, fresh-cache trees or duplicate video decodes there.
- Use `python -m macrostitch.storage --workspace .` to preview legacy cleanup.
  Preserve lock files, never traverse symlinked roots, and refuse cleanup while
  jobs hold their locks. Record before/after allocated bytes (hardlinks make
  summed path sizes misleading).
