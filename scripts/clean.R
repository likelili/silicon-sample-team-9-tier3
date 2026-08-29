#!/usr/bin/env Rscript
## Entry point: clean a raw Tier-1 survey export into the target schema.
## With no argument it reads the single CSV you placed in raw_data_deposit/.
##   make clean                          |     Rscript scripts/clean.R
##   make clean INPUT=path/to/raw.csv    |     Rscript scripts/clean.R path/to/raw.csv [output.csv]
.a    <- commandArgs(FALSE)
.dir  <- dirname(normalizePath(sub("^--file=", "", .a[grep("^--file=", .a)])))
.root <- dirname(.dir)
suppressPackageStartupMessages(library(jsonlite))
source(file.path(.dir, "lib", "clean_lib.R"))

args <- commandArgs(trailingOnly = TRUE)

## Resolve the input: an explicit path, or auto-discover it in raw_data_deposit/.
input <- if (length(args) >= 1) args[1] else {
  deposit <- file.path(.root, "raw_data_deposit")
  csvs <- list.files(deposit, pattern = "\\.csv$", full.names = TRUE, ignore.case = TRUE)
  if (length(csvs) == 0L)
    stop("No CSV found in raw_data_deposit/. Put your raw Qualtrics export there and re-run, ",
         "or pass a path: Rscript scripts/clean.R <raw_export.csv>", call. = FALSE)
  if (length(csvs) > 1L)
    stop("Multiple CSVs in raw_data_deposit/:\n  ", paste(basename(csvs), collapse = "\n  "),
         "\nLeave only one, or name it explicitly: make clean INPUT=raw_data_deposit/<file>.csv",
         call. = FALSE)
  csvs[1]
}

## Name the output from metadata.json so a `secondary-k` (or any) entry lands at
## the right filename: <team_id>_T1_<entry>_v1.csv. Edit metadata.json *before*
## running clean. An explicit second argument overrides this.
auto_named <- length(args) < 2
out <- if (!auto_named) args[2] else {
  mp <- file.path(.root, "metadata.json")
  m  <- if (file.exists(mp)) tryCatch(fromJSON(mp), error = function(e) NULL) else NULL
  team  <- m$team_id
  entry <- if (is.null(m$entry) || !nzchar(m$entry)) "primary" else m$entry
  file.path(.root, "predictions",
            if (is.null(team)) "cleaned_T1.csv" else sprintf("%s_T1_%s_v1.csv", team, entry))
}
clean_submission(input, out)

## Refresh metadata.json -> prediction_files with the new file's fingerprint, so
## the participant never edits a SHA-256 by hand. Best-effort: a metadata problem
## (e.g. team_id still unset) shouldn't fail the clean itself.
if (auto_named) {
  .manifest_sourced <- TRUE
  source(file.path(.dir, "manifest.R"))
  tryCatch(update_manifest(.root),
           error = function(e) message("manifest: skipped (", conditionMessage(e), ")"))
}
