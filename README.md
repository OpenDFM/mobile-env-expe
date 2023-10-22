<!-- vimc: call SyntaxRange#Include('```bibtex', '```', 'bib', 'NonText'): -->

Experiment codes for paper [*Mobile-Env: An Evaluation Platform and Benchmark
for Interactive Agents in LLM Era*](https://arxiv.org/abs/2305.08144).

`launch.sh` is the launcher for experiments. To launch the program,
[Mobile-Env](https://github.com/X-LANCE/Mobile-Env) environment should be set
up. Additionally, a tokenizer is required for `VhIoWrapper` wrapper, which can
be downloaded from [Hugging Face](https://huggingface.co). The tokenizer of
`bert-base-uncased` is ok.

Remote daemon for
[LLaMA-13B](https://github.com/facebookresearch/llama/tree/llama_v1) is
available at
[zdy023/llama-remote](https://hub.docker.com/r/zdy023/llama-remote).

### Citation

```bibtex
@article{DanyangZhang2023_MobileEnv,
  title     = {{Mobile-Env}: An Evaluation Platform and Benchmark for Interactive Agents in LLM Era},
  author    = {Danyang Zhang and
               Lu Chen and
               Zihan Zhao and
               Ruisheng Cao and
               Kai Yu},
  journal   = {CoRR},
  volume    = {abs/2305.08144},
  year      = {2023},
  url       = {https://arxiv.org/abs/2305.08144},
  eprinttype = {arXiv},
  eprint    = {2305.08144},
}
```
