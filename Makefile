# edgeinfer: common tasks. Paths default to ~/edgeinfer-data; override on the command line.
DATA   ?= $(HOME)/edgeinfer-data
HF     ?= $(DATA)/hf/Qwen3.5-0.8B
MODEL  ?= $(DATA)/models/qwen3.5-0.8b-q4m.eim
PY     ?= .venv/bin/python

.PHONY: build venv test check convert imatrix serve bench-raw bench-serve quality demo

build:
	cmake -B build -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build build -j

venv:
	python3 -m venv .venv && $(PY) -m pip install -q -r requirements.txt

# unit tests (no model needed) + template test (needs the tokenizer dir)
test: build
	$(PY) tests/test_kernels.py
	$(PY) tests/test_chat.py $(HF)

# end-to-end correctness against transformers + speculative/prefix-cache invariants
check: build
	$(PY) tests/ref_check.py $(DATA)/models/qwen3.5-0.8b-bf16.eim $(DATA)/ref/qwen3.5-0.8b.npz
	$(PY) tests/test_spec.py $(MODEL) $(HF)

imatrix: build
	$(PY) -m edgeinfer.convert $(HF) $(DATA)/models/tmp-q8.eim --qtype q8 --head q8
	$(PY) tools/imatrix.py $(DATA)/models/tmp-q8.eim $(HF) $(DATA)/imatrix/$(notdir $(HF)).npz

convert: build
	$(PY) -m edgeinfer.convert $(HF) $(MODEL) --profile q4m --imatrix $(DATA)/imatrix/$(notdir $(HF)).npz

serve: build
	$(PY) -m edgeinfer.server --config $(DATA)/models.json --port 8091

bench-raw: build
	$(PY) bench/compare_raw.py --name $(notdir $(HF)) --eim $(MODEL) --gguf $(GGUF) --hf $(HF)

bench-serve: build
	$(PY) bench/compare_serve.py --name $(notdir $(HF)) --eim $(MODEL) --tokenizer $(HF) --gguf $(GGUF)

quality: build
	$(PY) bench/quality.py --base $(DATA)/models/qwen3.5-0.8b-bf16.eim --hf $(HF) --test $(MODEL)

demo:
	$(PY) -m apps.analyst.demo.make_demo_db $(DATA)/demo/shop.db
	$(PY) -m apps.analyst.build_index apps/analyst/demo/demo.json
