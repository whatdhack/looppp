# Third-party material

## KernelBench-Hard (problem deck)

looppp grades candidates with the KernelBench-Hard problem deck from
https://github.com/Infatoshi/kernelbench.com (`benchmarks/hard`), pinned to commit
`62e4c346f5f351e06169676c605267d0291acaa5`. The deck is **not vendored**: `looppp fetch-deck` makes a
shallow sparse clone into `.looppp/deck/`.

MIT License, Copyright (c) 2026 Elliot Arledge. KernelBench is an independent project and is not
affiliated with looppp.

## KernelBench-Hard agent traces (calibration only)

`looppp calibrate` / `trace-solution` download published traces from
https://huggingface.co/datasets/Infatoshi/kernelbench-hard-traces (MIT) and replay the agent's
Write/Edit calls to rebuild a published `solution.py`. Nothing from the dataset is committed here.
