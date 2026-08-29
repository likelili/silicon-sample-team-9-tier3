.PHONY: help check clean manifest zenodo_citation

help:
	@echo "make check                  validate this submission (files, metadata, data)"
	@echo "make clean                  clean the raw Tier-1 export in raw_data_deposit/ into predictions/"
	@echo "make clean INPUT=raw.csv    clean a specific raw export instead"
	@echo "make manifest               fingerprint predictions/ and record them in metadata.json"
	@echo "make zenodo_citation        (re)generate .zenodo.json from metadata.json (Zenodo deposit metadata)"

check:
	Rscript scripts/check.R

clean:
	@if [ -n "$(INPUT)" ]; then Rscript scripts/clean.R "$(INPUT)"; else Rscript scripts/clean.R; fi

manifest:
	Rscript scripts/manifest.R

zenodo_citation:
	Rscript scripts/zenodo_citation.R
