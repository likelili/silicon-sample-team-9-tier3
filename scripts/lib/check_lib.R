## ---------------------------------------------------------------------------
## check_lib.R — self-assess a Silicon Sample Benchmark submission
## Library sourced by check.R (`make check`); not run directly.
##
## Checks a submission against the published spec: metadata.json schema, file
## naming, SHA-256 integrity, per-tier data structure / coverage, and value
## sanity. Every team can run this before depositing.
##
## Usage:
##   source("check_lib.R")                     # also sources submission_spec.R
##   check_submission("metadata.json", dir = ".")
##
## Returns (invisibly) a tibble of checks; prints a report and an overall
## verdict (PASS / PASS WITH WARNINGS / FAIL) and writes <metadata>_check_report.txt.
## ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse); library(jsonlite); library(digest)
})

if (!exists("sst")) {
  ## Locate submission_spec.R robustly: `ofile` works when sourced with chdir,
  ## but is unreliable under `Rscript -e/-f`, so fall back to the --file= arg dir
  ## and a few known relative paths. (Or source submission_spec.R yourself first.)
  .cands <- character()
  .of <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(.of)) .cands <- c(.cands, file.path(dirname(.of), "submission_spec.R"))
  .fa <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  if (length(.fa)) .cands <- c(.cands, file.path(dirname(normalizePath(.fa[1])), "submission_spec.R"))
  .cands <- c(.cands, "scripts/lib/submission_spec.R", "lib/submission_spec.R", "submission_spec.R")
  .spec <- Find(file.exists, .cands)
  if (is.null(.spec)) stop("check_lib.R: cannot locate submission_spec.R — source it first, ",
                           "or run via `make check`.", call. = FALSE)
  source(.spec)
}

## Zenodo legacy deposit-metadata vocabulary (the format the GitHub->Zenodo
## integration consumes), used by the .zenodo.json checks in check_repo().
.ZENODO_UPLOAD_TYPES  <- c("publication","poster","presentation","dataset","image",
                           "video","software","lesson","physicalobject","other")
.ZENODO_ACCESS_RIGHTS <- c("open","embargoed","restricted","closed")
.ZENODO_RELATIONS <- c("isCitedBy","cites","isSupplementTo","isSupplementedBy",
  "isContinuedBy","continues","isDescribedBy","describes","hasMetadata",
  "isMetadataFor","isNewVersionOf","isPreviousVersionOf","isPartOf","hasPart",
  "isReferencedBy","references","isDocumentedBy","documents","isCompiledBy",
  "compiles","isVariantFormOf","isOriginalFormOf","isIdenticalTo",
  "isAlternateIdentifier","isReviewedBy","reviews","isDerivedFrom","isSourceOf",
  "requires","isRequiredBy","isObsoletedBy","obsoletes")

## ISO 7064 MOD-11-2 — the ORCID checksum Zenodo enforces; a failure (e.g. the
## dummy 0000-0000-0000-0000) aborts the deposit as an opaque HTTP 500.
.orcid_ok <- function(x) {
  d <- gsub("-", "", x)
  if (!grepl("^[0-9]{15}[0-9X]$", d)) return(FALSE)
  total <- 0L
  for (ch in strsplit(substr(d, 1, 15), "")[[1]]) total <- (total + as.integer(ch)) * 2L
  result <- (12L - total %% 11L) %% 11L
  (if (result == 10L) "X" else as.character(result)) == substr(d, 16, 16)
}

.check_bundle <- function(mpath, dir = ".") {
  res <- tibble(check = character(), status = character(), detail = character())
  add <- function(check, status, detail = "") {
    res <<- add_row(res, check = check, status = status, detail = detail); invisible()
  }
  ok   <- function(cond, check, bad, good = "") if (isTRUE(cond)) add(check, "PASS", good) else add(check, "FAIL", bad)
  warn <- function(cond, check, bad, good = "") if (isTRUE(cond)) add(check, "PASS", good) else add(check, "WARN", bad)
  rng  <- function(x, lo, hi) sum(!is.na(x) & (x < lo | x > hi))

  if (!file.exists(mpath)) { add("metadata.json present", "FAIL", mpath); return(res) }
  m <- tryCatch(fromJSON(mpath), error = function(e) NULL)
  if (is.null(m)) { add("metadata.json parses", "FAIL", "invalid JSON"); return(res) }
  add("metadata.json parses", "PASS")

  ## ---- metadata schema ----
  ok(is.character(m$team_id) && nzchar(m$team_id), "team_id present", "missing/empty")
  ok(is.character(m$team_name) && nzchar(m$team_name), "team_name present", "missing/empty")
  ok(is.character(m$contact) && nzchar(m$contact), "contact present", "missing/empty")
  tier <- suppressWarnings(as.integer(m$tier))
  ok(!is.na(tier) && tier %in% 1:3, "tier in {1,2,3}", paste("got", m$tier))
  ok(is.character(m$entry) && grepl("^(primary|secondary-\\d+)$", m$entry %||% ""),
     "entry is primary|secondary-k", paste("got", m$entry))
  ok(length(m$models) >= 1, "models listed", "none listed")
  ok(is.character(m$approach_family) && nzchar(m$approach_family), "approach_family present", "missing")
  dc <- m$disclosure_class
  ok(isTRUE(dc %in% c("A", "B", "C")), "disclosure_class in {A,B,C}", paste("got", dc))
  if (isTRUE(dc == "A"))
    warn(is.null(m$escrow_doi) || is.na(m$escrow_doi), "escrow_doi null for Class A", "set but class A")
  if (isTRUE(dc %in% c("B", "C")))
    warn(!is.null(m$escrow_doi) && !is.na(m$escrow_doi), "escrow_doi set for Class B/C", "missing escrow_doi")
  ok(isTRUE(m$blinding_attestation), "blinding_attestation == true", "must be true")
  ok(!is.null(m$coverage$interventions) && !is.null(m$coverage$outcomes),
     "coverage declared", "coverage.interventions/outcomes missing")
  ci <- length(sst$interventions)
  co <- length(sst$outcomes)
  ok(isTRUE(suppressWarnings(as.integer(m$coverage$interventions)) == ci) &&
       isTRUE(suppressWarnings(as.integer(m$coverage$outcomes)) == co),
     paste0("coverage is full (", ci, " interventions, ", co, " outcomes)"),
     paste0("declared ", m$coverage$interventions %||% "?", " intervention(s), ",
            m$coverage$outcomes %||% "?", " outcome(s) — partial coverage is not ",
            "accepted; every intervention and outcome must be predicted"))
  pf <- m$prediction_files
  if (is.null(pf) || nrow(as.data.frame(pf)) < 1) {
    add("prediction_files listed", "FAIL", "none"); return(res)
  }
  pf <- as.data.frame(pf)
  add("prediction_files listed", "PASS", paste(nrow(pf), "file(s)"))

  team <- m$team_id %||% ""
  ## Pre-generation / "staged" state: team_id is set but prediction_files still
  ## point at the shipped example(s) — predictions haven't been generated and the
  ## manifest re-run yet. Report it as one clear WARN, not a cascade of
  ## filename/hash FAILs that read like a broken repo.
  if (nzchar(team) && team != "example" && all(grepl("^example_", basename(pf$file)))) {
    add(paste0("submission staged for team '", team, "'"), "WARN",
        "prediction_files still reference the example — generate your predictions, then run `make manifest`, before depositing")
    return(res)
  }

  ## One repo = one entry: a Tier-2 entry has 2 files (main + moderator cells),
  ## every other tier has 1. Extra entries belong in their own repo/deposit.
  n_expected <- if (isTRUE(tier == 2)) 2L else 1L
  warn(nrow(pf) == n_expected, paste0("file count for Tier ", tier),
       paste0("expected ", n_expected, " for one entry, got ", nrow(pf),
              " — a repo holds one entry; put extra entries in their own repo"))

  ## ---- per-file: name, hash, structure ----
  ## Completeness below always runs against the full grid (`ci`/`co` from the
  ## spec): full coverage is a hard requirement, partial coverage fails above.
  for (i in seq_len(nrow(pf))) {
    f <- pf$file[i]; sha <- pf$sha256[i]; fp <- file.path(dir, f)
    pat <- if (isTRUE(tier == 2))
      sprintf("^%s_T2_(primary|secondary-\\d+)_v\\d+_cells_(main|moderator)\\.csv$", team)
    else
      sprintf("^%s_T%s_(primary|secondary-\\d+)_v\\d+\\.csv$", team, tier)
    ok(grepl(pat, basename(f)), paste0("filename ok: ", basename(f)),
       "does not match <team_id>_T<tier>_<entry>_v<n>[...].csv")

    if (!file.exists(fp)) { add(paste0("file present: ", f), "FAIL", "not found"); next }
    add(paste0("file present: ", f), "PASS")
    actual <- digest(file = fp, algo = "sha256")
    ok(identical(tolower(actual), tolower(sha %||% "")), paste0("sha256 matches: ", f),
       paste0("metadata=", substr(sha, 1, 10), "… actual=", substr(actual, 1, 10), "…"))

    d <- tryCatch(suppressWarnings(read_csv(fp, show_col_types = FALSE)), error = function(e) NULL)
    if (is.null(d)) { add(paste0("readable CSV: ", f), "FAIL", "could not parse"); next }

    if (isTRUE(tier == 1))                  .check_t1(d, f, add, ok, warn, rng)
    else if (isTRUE(tier == 3))             .check_t3(d, f, add, ok, warn, rng, ci, co)
    else if (grepl("_cells_moderator", f))  .check_t2_mod(d, f, add, ok, warn, rng, ci, co)
    else                                    .check_t2_main(d, f, add, ok, warn, rng, ci, co)
  }
  res
}

## ---------------- public entry points ----------------

## Validate a single deposited bundle (metadata.json + its prediction files in `dir`).
check_submission <- function(metadata = "metadata.json", dir = ".") {
  .finish(.check_bundle(file.path(dir, metadata), dir), metadata, dir)
}

## Validate a whole cloned submission repository: presence-of-files first, then
## the metadata + per-file checks above.
check_repo <- function(root = ".") {
  res <- tibble(check = character(), status = character(), detail = character())
  add  <- function(check, status, detail = "")
    res <<- add_row(res, check = check, status = status, detail = detail)
  ok   <- function(cond, check, bad) if (isTRUE(cond)) add(check, "PASS") else add(check, "FAIL", bad)
  warn <- function(cond, check, bad) if (isTRUE(cond)) add(check, "PASS") else add(check, "WARN", bad)

  has <- function(p) file.exists(file.path(root, p))
  ok(has("metadata.json"),   "metadata.json present", "missing at repo root")
  ok(has("registration.md"), "registration.md present", "missing at repo root")
  ok(has("codebook.csv"),    "codebook.csv present", "missing at repo root")
  ok(dir.exists(file.path(root, "survey")), "survey/ present", "missing")
  warn(has(file.path("survey", "survey.qsf")), "survey/survey.qsf present",
       "not present yet (provided on invitation)")

  preds <- list.files(file.path(root, "predictions"), pattern = "\\.csv$")
  ok(length(preds) >= 1, "predictions/ has a CSV", "no prediction file in predictions/")

  if (has("registration.md")) {
    rl <- readLines(file.path(root, "registration.md"), warn = FALSE)
    ## An item is blank only when its label line (a "- **N.N Label** — prompt:")
    ## ends in a colon AND nothing follows it — i.e. the next non-blank line is
    ## another bullet or a section header. Anchoring to the bullet start (not "any
    ## colon-terminated line") means a filled answer that wraps onto a line ending
    ## in a colon is no longer miscounted as blank.
    is_label <- grepl("^- \\*\\*.*\\*\\* [—-] .*:\\s*$", rl)
    nxt <- c(trimws(rl[-1]), "")
    blank <- sum(is_label & (nxt == "" | grepl("^- \\*\\*", nxt) | grepl("^#", nxt)))
    warn(blank == 0, "registration.md filled in", paste(blank, "checklist item(s) still blank"))
  }

  is_example <- TRUE
  if (has("metadata.json")) {
    m <- tryCatch(fromJSON(file.path(root, "metadata.json")), error = function(e) NULL)
    if (!is.null(m)) {
      is_example <- identical(m$team_id, "example")
      warn(!is_example, "team_id set (not the example)",
           "still 'example' — edit metadata.json before submitting")
      cr <- m$code_repository
      warn(!is.null(cr) && nzchar(cr) && !grepl("your-team/your-repo", cr),
           "code_repository set",
           "link your generation code in metadata.json (code_repository / code_doi)")
    }
  }
  # .zenodo.json controls the permanent Zenodo deposit record (run `make zenodo_citation`).
  # Structural validation mirrors Zenodo's legacy deposit schema (after Valentin
  # Kriegmair's validate_zenodo.py): required fields, enums, license, creators, and
  # the ORCID MOD-11-2 checksum. The one thing it can't confirm offline is the exact
  # license id string, which Zenodo resolves at deposit time.
  if (!has(".zenodo.json")) {
    warn(FALSE, ".zenodo.json present",
         "missing — run `make zenodo_citation` to generate Zenodo deposit metadata")
  } else {
    z <- tryCatch(fromJSON(file.path(root, ".zenodo.json")), error = function(e) NULL)
    ok(!is.null(z), ".zenodo.json parses", "invalid JSON in .zenodo.json")
    if (!is.null(z)) {
      nm <- if (is.null(z$creators)) character() else z$creators$name
      warn(nzchar(z$title %||% "") && nzchar(z$description %||% "") &&
             length(nm) >= 1 && all(nzchar(nm)),
           ".zenodo.json has title/description/creator",
           "fill title, description, and at least one creator name")
      warn(isTRUE((z$upload_type %||% "") %in% .ZENODO_UPLOAD_TYPES),
           ".zenodo.json upload_type valid",
           paste0("upload_type '", z$upload_type %||% "", "' not a Zenodo upload type"))
      warn(is.null(z$access_right) || isTRUE(z$access_right %in% .ZENODO_ACCESS_RIGHTS),
           ".zenodo.json access_right valid",
           paste0("access_right '", z$access_right %||% "", "' invalid"))
      warn(is.character(z$license) && nzchar(z$license %||% ""),
           ".zenodo.json license set", "license must be a non-empty id string (e.g. CC-BY-4.0)")
      warn(is.null(z$keywords) ||
             (is.character(z$keywords) && all(nzchar(z$keywords))),
           ".zenodo.json keywords valid", "keywords must be a list of strings")
      warn(!any(grepl("Lastname, Firstname", nm)),
           ".zenodo.json creators filled",
           "creator still 'Lastname, Firstname' — fill `creators` in metadata.json, then re-run make zenodo_citation")
      orc <- if (is.null(z$creators)) NULL else z$creators$orcid
      orc <- orc[!is.na(orc) & nzchar(orc)]
      bad <- orc[!vapply(orc, .orcid_ok, logical(1))]
      warn(length(bad) == 0,
           ".zenodo.json ORCIDs valid (format + checksum)",
           paste0("invalid ORCID (format or MOD-11-2 checksum): ", paste(bad, collapse = ", "),
                  " — Zenodo would reject the deposit (HTTP 500); fix or remove it"))
      ri <- z$related_identifiers
      warn(is.null(ri) || (all(nzchar(ri$identifier %||% "")) &&
                             all(ri$relation %in% .ZENODO_RELATIONS)),
           ".zenodo.json related_identifiers valid",
           "a related_identifier is missing an identifier or uses an unknown relation")
    }
  }

  leftover <- list.files(file.path(root, "predictions"), pattern = "^example_.*\\.csv$")
  if (!is_example && length(leftover))
    warn(FALSE, "example files removed from predictions/",
         paste("delete before depositing:", paste(leftover, collapse = ", ")))
  if (!is_example && has(file.path("raw_data_deposit", "example_raw_export.csv")))
    warn(FALSE, "example raw export removed from raw_data_deposit/",
         "delete raw_data_deposit/example_raw_export.csv before depositing")

  res <- bind_rows(res, .check_bundle(file.path(root, "metadata.json"), root))
  .finish(res, "metadata.json", root)
}

## ---------------- per-tier structural checks ----------------

.range_warn <- function(d, f, add, rng) {
  for (o in intersect(c(sst$scale_0_100, sst$trust_items), names(d)))
    if (rng(d[[o]], 0, 100) > 0) add(paste0(o, " in [0,100]: ", f), "WARN",
                                     paste(rng(d[[o]], 0, 100), "value(s) out of range"))
}

## Missing prediction values FAIL for every prediction grid, the Tier-2
## moderator grid included: an NA that passes the row-count check would
## silently score as missing, and pairwise exclusion would score different
## teams on different, self-chosen cell sets. There is no cell a team cannot
## fill — to predict "no moderation for this group", repeat the condition
## mean from the main file in that group's cells (see FAQ.md).
.no_na_fail <- function(d, f, add, ok, vcols) {
  for (v in intersect(vcols, names(d))) {
    n_na <- sum(is.na(d[[v]]))
    ok(n_na == 0, paste0("no missing ", v, ": ", f),
       paste0(n_na, " NA value(s) — every cell needs a prediction"))
  }
}

.check_t1 <- function(d, f, add, ok, warn, rng) {
  miss <- setdiff(sst$tier1_required, names(d))
  ok(length(miss) == 0, paste0("Tier-1 required columns: ", f),
     paste("missing:", paste(miss, collapse = ", ")))
  if ("condition" %in% names(d)) {
    bad <- setdiff(unique(as.character(d$condition)), sst$conditions)
    ok(length(bad) == 0, paste0("condition labels valid: ", f),
       paste("unknown:", paste(bad, collapse = ", ")))
    pres <- intersect(sst$conditions, unique(as.character(d$condition)))
    warn(length(pres) == length(sst$conditions), paste0("all 17 conditions present: ", f),
         paste(length(pres), "of", length(sst$conditions), "present"))
  }
  ## Moderator label validity is a FAIL, not a WARN: a level string that doesn't
  ## exactly match the spec silently drops those respondents from every subgroup
  ## analysis at scoring time. All-NA moderators FAIL for the same reason; a
  ## partially-NA moderator is worth a WARN so the team sees the effective N.
  for (mod in names(sst$moderators)) if (mod %in% names(d)) {
    bad <- setdiff(na.omit(unique(as.character(d[[mod]]))), sst$moderators[[mod]])
    ok(length(bad) == 0, paste0(mod, " levels valid: ", f),
       paste0("unknown: ", paste(bad, collapse = ", "),
              " — must exactly match the spec strings (see codebook.csv)"))
    n_na <- sum(is.na(d[[mod]]))
    if (n_na == nrow(d)) {
      add(paste0(mod, " has data: ", f), "FAIL",
          "entirely NA — this moderator would be missing from all subgroup analyses")
    } else if (n_na > 0.1 * nrow(d)) {
      add(paste0(mod, " mostly present: ", f), "WARN",
          paste0(n_na, " of ", nrow(d), " rows NA — these drop out of ",
                 mod, " subgroup analyses"))
    }
  }
  if ("condition" %in% names(d)) {
    n_cond <- table(as.character(d$condition))
    add(paste0("per-condition N: ", f), "PASS",
        if (min(n_cond) == max(n_cond)) paste0(min(n_cond), " in every condition")
        else paste0("min ", min(n_cond), " (", names(n_cond)[which.min(n_cond)],
                    "), max ", max(n_cond), " (", names(n_cond)[which.max(n_cond)], ")"))
    ## Precision requirement (benchmark preregistration): a Tier 1 entry needs
    ## at least the human half's per-cell sizes -- 500 per intervention and
    ## 1,000 in control -- so its effects are not noisier than the reference
    ## for reasons unrelated to the method. WARN rather than FAIL so pilot
    ## files still pass a structural check; the deposited entry must meet it.
    n_int <- n_cond[setdiff(names(n_cond), "control")]
    below <- c(names(n_int)[n_int < 500],
               if ("control" %in% names(n_cond) && n_cond[["control"]] < 1000) "control")
    warn(length(below) == 0,
         paste0("precision floor (500/intervention, 1,000 control): ", f),
         paste0(length(below), " condition(s) below the preregistered minimum: ",
                paste(head(below, 5), collapse = ", "),
                if (length(below) > 5) ", ..." else ""))
  }
  if ("profile_id" %in% names(d))
    warn(!any(duplicated(d$profile_id)), paste0("profile_id unique: ", f),
         paste(sum(duplicated(d$profile_id)), "duplicate(s)"))
  ## Scoring reads the composite column as submitted — it is NOT recomputed from
  ## the items. Warn when the submitted primary outcome diverges from its
  ## item-level definition (mean of the four subscale means), so a hand-built
  ## file can't silently score on values its own items contradict.
  if (all(c("trust_multidimensional", sst$trust_items) %in% names(d))) {
    subscales <- sapply(c("competence", "integrity", "benevolence", "openness"),
                        function(s) rowMeans(d[paste0("trust_", s, "_", 1:3)], na.rm = TRUE))
    expect <- rowMeans(subscales, na.rm = TRUE)
    n_bad  <- sum(abs(d$trust_multidimensional - expect) > 0.51, na.rm = TRUE)
    warn(n_bad == 0, paste0("trust_multidimensional consistent with items: ", f),
         paste0(n_bad, " row(s) deviate from the codebook definition by > 0.5 — ",
                "scoring uses the composite as submitted"))
  }
  .range_warn(d, f, add, rng)
  if ("donation_ams" %in% names(d))
    warn(rng(d$donation_ams, sst$donation_range[1], sst$donation_range[2]) == 0,
         paste0("donation_ams in [0,10]: ", f), "value(s) out of range")
  if ("newsletter_signup" %in% names(d)) {
    vals <- unique(na.omit(as.character(d$newsletter_signup)))
    warn(all(vals %in% c("0", "1", "TRUE", "FALSE")), paste0("newsletter_signup binary: ", f),
         paste("unexpected:", paste(setdiff(vals, c("0","1","TRUE","FALSE")), collapse = ", ")))
  }
}

.check_cells <- function(d, f, add, ok, warn, rng, cols) {
  ok(all(cols %in% names(d)), paste0("columns present: ", f),
     paste("missing:", paste(setdiff(cols, names(d)), collapse = ", ")))
  if ("outcome" %in% names(d))
    ok(length(setdiff(unique(d$outcome), sst$outcomes)) == 0, paste0("outcome labels valid: ", f),
       paste("unknown:", paste(setdiff(unique(d$outcome), sst$outcomes), collapse = ", ")))
  if ("condition" %in% names(d))
    ok(length(setdiff(unique(as.character(d$condition)), sst$conditions)) == 0,
       paste0("condition labels valid: ", f),
       paste("unknown:", paste(setdiff(unique(as.character(d$condition)), sst$conditions), collapse = ", ")))
}

## Per-outcome value ranges for Tier-2 cell means (#19): newsletter_signup is a
## 0-1 proportion, donation_ams a 0-10 dollar mean, every other outcome a 0-100
## mean. (Tier-3 `ate` is an unbounded difference, so it is intentionally NOT
## range-checked — only Tier-1 ran value-range checks before.)
.cell_value_warn <- function(d, f, warn, rng, vcol) {
  if (!all(c("outcome", vcol) %in% names(d))) return(invisible())
  hi_of <- function(o) if (o == "newsletter_signup") 1 else if (o == "donation_ams") 10 else 100
  for (o in intersect(sst$outcomes, unique(d$outcome))) {
    hi <- hi_of(o); n <- rng(d[[vcol]][d$outcome == o], 0, hi)
    if (n > 0) warn(FALSE, paste0(o, " ", vcol, " in [0,", hi, "]: ", f),
                    paste(n, "value(s) out of range"))
  }
}

## Cell completeness + de-duplication (#4/#22): every cell of the full grid
## present exactly once. `dims` is a named vector of expected distinct counts per
## key column; expected cells = prod(dims). These FAIL (not WARN) so an incomplete
## or duplicated submission cannot pass clean.
.grid_complete <- function(d, f, keys, dims, add, ok) {
  if (!all(keys %in% names(d))) return(invisible())
  cells <- d[keys]
  dup <- sum(duplicated(cells))
  ok(dup == 0, paste0("no duplicate cells: ", f),
     paste(dup, "duplicate (", paste(keys, collapse = ","), ") cell(s)"))
  got   <- vapply(keys, function(k) n_distinct(d[[k]]), integer(1))
  ncell <- nrow(distinct(cells)); exp_cells <- prod(dims)
  ok(all(got[names(dims)] == dims) && ncell == exp_cells,
     paste0("complete coverage (", exp_cells, " cells): ", f),
     paste0(ncell, " of ", exp_cells, " cells present (",
            paste(sprintf("%s %d/%d", names(dims), got[names(dims)], dims), collapse = ", "),
            ") — every intervention and outcome must be predicted (add missing cells)"))
}

.check_t2_main <- function(d, f, add, ok, warn, rng, ci, co) {
  .check_cells(d, f, add, ok, warn, rng, sst$tier2_main_cols)
  .cell_value_warn(d, f, warn, rng, "mean")
  .no_na_fail(d, f, add, ok, "mean")
  .grid_complete(d, f, c("condition", "outcome"),
                 c(condition = ci + 1L, outcome = co), add, ok)
}

.check_t2_mod <- function(d, f, add, ok, warn, rng, ci, co) {
  .check_cells(d, f, add, ok, warn, rng, sst$tier2_mod_cols)
  .cell_value_warn(d, f, warn, rng, "mean")
  ## NA cells FAIL here too (policy change 2026-08: previously WARN + pairwise
  ## scoring). To predict "no moderation for this group", repeat the condition
  ## mean from the main file in that group's cells — an always-available,
  ## honest prediction that the intervention works the same for that group.
  .no_na_fail(d, f, add, ok, "mean")
  if ("moderator" %in% names(d))
    ok(length(setdiff(unique(d$moderator), names(sst$moderators))) == 0,
       paste0("moderator labels valid: ", f),
       paste("unknown:", paste(setdiff(unique(d$moderator), names(sst$moderators)), collapse = ", ")))
  if (all(c("moderator", "moderator_level") %in% names(d))) {
    bad <- d |> distinct(moderator, moderator_level) |>
      filter(!map2_lgl(moderator, moderator_level,
                       ~ .y %in% sst$moderators[[.x]]))
    warn(nrow(bad) == 0, paste0("moderator_level values valid: ", f),
         paste(nrow(bad), "invalid moderator/level pair(s)"))
    ## Completeness across condition x (moderator,level) x outcome — the moderator
    ## grid that was previously not checked at all.
    d2 <- d |> mutate(.ml = paste(moderator, moderator_level, sep = "::"))
    .grid_complete(d2, f, c("condition", ".ml", "outcome"),
                   c(condition = ci + 1L,
                     .ml = sum(lengths(sst$moderators)), outcome = co), add, ok)
  }
}

.check_t3 <- function(d, f, add, ok, warn, rng, ci, co) {
  ok(all(sst$tier3_cols %in% names(d)), paste0("Tier-3 columns present: ", f),
     paste("missing:", paste(setdiff(sst$tier3_cols, names(d)), collapse = ", ")))
  if ("condition" %in% names(d)) {
    ok(!("control" %in% d$condition), paste0("no control row (ATEs are vs control): ", f),
       "control present")
    ok(length(setdiff(unique(as.character(d$condition)), sst$interventions)) == 0,
       paste0("intervention labels valid: ", f),
       paste("unknown:", paste(setdiff(unique(as.character(d$condition)), sst$interventions), collapse = ", ")))
  }
  if ("outcome" %in% names(d))
    ok(length(setdiff(unique(d$outcome), sst$outcomes)) == 0, paste0("outcome labels valid: ", f),
       paste("unknown:", paste(setdiff(unique(d$outcome), sst$outcomes), collapse = ", ")))
  .no_na_fail(d, f, add, ok, "ate")
  ## Positive completeness: all intervention x outcome ATEs present exactly once
  ## (replaces the one-sided "row count <= 208" warn that let partial files pass).
  .grid_complete(d, f, c("condition", "outcome"),
                 c(condition = ci, outcome = co), add, ok)
}

## ---------------- reporting ----------------

.finish <- function(res, metadata, dir) {
  verdict <- if (any(res$status == "FAIL")) "FAIL"
             else if (any(res$status == "WARN")) "PASS WITH WARNINGS" else "PASS"
  mark <- c(PASS = "[ok]  ", WARN = "[warn]", FAIL = "[FAIL]")
  lines <- c(
    "Silicon Sample Benchmark — submission self-check",
    paste0(rep("-", 52), collapse = ""),
    sprintf("%s %s%s", mark[res$status], res$check,
            if_else(nzchar(res$detail), paste0("  — ", res$detail), "")),
    paste0(rep("-", 52), collapse = ""),
    sprintf("OVERALL: %s   (%d pass, %d warn, %d fail)", verdict,
            sum(res$status == "PASS"), sum(res$status == "WARN"), sum(res$status == "FAIL"))
  )
  cat(lines, sep = "\n"); cat("\n")
  out <- file.path(dir, paste0(tools::file_path_sans_ext(metadata), "_check_report.txt"))
  writeLines(lines, out)
  invisible(res)
}

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a
