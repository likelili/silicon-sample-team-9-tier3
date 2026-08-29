#!/usr/bin/env Rscript
## Entry point: fingerprint this submission's prediction files and record them in
## metadata.json, so you never paste a SHA-256 by hand.  `make manifest`
##
## Scans predictions/ for files named <team_id>_*.csv (team_id is read from
## metadata.json), computes each file's SHA-256, and rewrites the
## `prediction_files` block. Run it after producing/updating any prediction file
## and before `make check`. `make clean` calls it for you.
.a    <- commandArgs(FALSE)
.dir  <- dirname(normalizePath(sub("^--file=", "", .a[grep("^--file=", .a)])))
.root <- dirname(.dir)
suppressPackageStartupMessages({ library(jsonlite); library(digest) })

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a

update_manifest <- function(root = ".") {
  mp <- file.path(root, "metadata.json")
  if (!file.exists(mp)) stop("metadata.json not found at ", mp, call. = FALSE)
  ## simplifyVector = FALSE keeps `models` (and other arrays) as JSON arrays and
  ## stops auto_unbox from collapsing length-1 arrays into scalars on write.
  m <- tryCatch(fromJSON(mp, simplifyVector = FALSE), error = function(e) NULL)
  if (is.null(m)) stop("metadata.json is not valid JSON", call. = FALSE)

  team <- m$team_id %||% ""
  if (!nzchar(team)) stop("metadata.json: team_id is missing/empty — set it first", call. = FALSE)
  tier  <- suppressWarnings(as.integer(m$tier %||% NA))
  entry <- m$entry %||% "primary"

  ## Match exactly the files of THIS (tier, entry) — same grammar `make check`
  ## enforces. A repo holds one entry, so this is the entry's file set: 2 files
  ## for Tier 2 (main + moderator cells), 1 otherwise. Scoping by tier also keeps
  ## the shipped Tier-1 example from sweeping in the T2/T3 example files.
  pat <- if (isTRUE(tier == 2L))
    sprintf("^%s_T2_%s_v\\d+_cells_(main|moderator)\\.csv$", team, entry)
  else if (tier %in% c(1L, 3L))
    sprintf("^%s_T%d_%s_v\\d+\\.csv$", team, tier, entry)
  else
    sprintf("^%s_T\\d+_%s_v\\d+", team, entry)   # tier unset: best-effort prefix

  preds <- file.path(root, "predictions")
  files <- sort(list.files(preds, pattern = "\\.csv$"))
  mine  <- files[grepl(pat, files)]
  if (length(mine) == 0L)
    stop("No Tier-", m$tier %||% "?", " '", entry, "' prediction files for team '", team,
         "' in predictions/. Produce them first (expected e.g. ",
         team, "_T", m$tier %||% "1", "_", entry, "_v1.csv).", call. = FALSE)

  m$prediction_files <- lapply(mine, function(f)
    list(file   = paste0("predictions/", f),
         sha256 = digest(file = file.path(preds, f), algo = "sha256")))

  write_json(m, mp, auto_unbox = TRUE, null = "null", pretty = TRUE)
  message("manifest: recorded ", length(mine), " prediction file(s) in metadata.json:")
  for (f in mine) message("  predictions/", f)
  invisible(m$prediction_files)
}

## Run when invoked directly. clean.R sets .manifest_sourced before sourcing this
## file so it can reuse update_manifest() without triggering a second run.
if (!exists(".manifest_sourced")) update_manifest(.root)
