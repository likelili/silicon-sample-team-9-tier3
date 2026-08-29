#!/usr/bin/env Rscript
## Entry point: generate `.zenodo.json` from metadata.json so your Zenodo deposit
## gets a well-formed permanent record instead of Zenodo's auto-scraped default
## (which has an empty description and no affiliation/ORCID/license/keywords).
## `make zenodo_citation`
##
## Reads metadata.json and (re)writes a standalone .zenodo.json at the repo root:
## title, description, creators, license, keywords, and links back to the
## benchmark. .zenodo.json is a DERIVED artifact — every field comes from
## metadata.json — so edit metadata.json (creators, abstract, license, …) and
## re-run; this always overwrites .zenodo.json to match. Don't hand-edit
## .zenodo.json; your changes there would be lost on the next run.
##
## .zenodo.json is what Zenodo reads when you cut the GitHub release. A DOI is
## permanent, so make sure metadata.json is filled before releasing.
.a    <- commandArgs(FALSE)
.dir  <- dirname(normalizePath(sub("^--file=", "", .a[grep("^--file=", .a)])))
.root <- dirname(.dir)
suppressPackageStartupMessages({ library(jsonlite) })

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a

# Tier -> human-readable method descriptor used in the title/description.
.tier_descriptor <- function(tier) {
  switch(as.character(tier),
         "1" = "individual simulation",
         "2" = "group-level reasoning",
         "3" = "direct effect forecast",
         "submission")
}

# A malformed ORCID — wrong shape OR a failed ISO 7064 MOD-11-2 checksum (e.g. the
# dummy 0000-0000-0000-0000) — makes Zenodo abort the deposit with an opaque HTTP
# 500, so such an ORCID must never be written. orcid_checksum_ok() is the exact
# check Zenodo applies.
.orcid_checksum_ok <- function(x) {
  d <- gsub("-", "", x)
  if (!grepl("^[0-9]{15}[0-9X]$", d)) return(FALSE)
  total <- 0L
  for (ch in strsplit(substr(d, 1, 15), "")[[1]]) total <- (total + as.integer(ch)) * 2L
  result   <- (12L - total %% 11L) %% 11L
  expected <- if (result == 10L) "X" else as.character(result)
  expected == substr(d, 16, 16)
}
.valid_orcid <- function(x) {
  is.character(x) && length(x) == 1 &&
    grepl("^[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9X]$", x) && .orcid_checksum_ok(x)
}

build_zenodo <- function(root = ".") {
  mp <- file.path(root, "metadata.json")
  if (!file.exists(mp)) stop("metadata.json not found at ", mp, call. = FALSE)
  m <- tryCatch(fromJSON(mp, simplifyVector = FALSE), error = function(e) NULL)
  if (is.null(m)) stop("metadata.json is not valid JSON", call. = FALSE)

  team  <- m$team_id %||% ""
  if (!nzchar(team)) stop("metadata.json: team_id is missing/empty — set it first", call. = FALSE)
  tier  <- suppressWarnings(as.integer(m$tier %||% NA))
  desc  <- .tier_descriptor(tier)
  tname <- m$team_name %||% team
  family <- m$approach_family %||% "(approach family — see registration.md)"
  models <- paste(unlist(m$models %||% list()), collapse = ", ")
  ints   <- m$coverage$interventions %||% 16
  outs   <- m$coverage$outcomes %||% 13

  creators <- lapply(m$creators %||% list(), function(c) {
    out <- list(name = c$name %||% "Lastname, Firstname")
    if (nzchar(c$affiliation %||% "")) out$affiliation <- c$affiliation
    orc <- c$orcid %||% ""
    if (nzchar(orc)) {
      if (.valid_orcid(orc)) out$orcid <- orc
      else message("  note: ORCID '", orc, "' for ", out$name,
                   " is invalid (format or checksum) and was OMITTED — ",
                   "fix it in metadata.json or Zenodo would reject the deposit.")
    }
    out
  })
  if (length(creators) == 0)
    creators <- list(list(name = "Lastname, Firstname", affiliation = "Your institution"))

  # Description: a team-supplied `abstract` in metadata.json (plain text, blank
  # lines = paragraphs) wins; otherwise a factual scaffold built from metadata.
  abstract <- m$abstract %||% ""
  description <- if (nzchar(trimws(abstract))) {
    paras <- strsplit(trimws(abstract), "\n[ \t]*\n")[[1]]
    paste(sprintf("<p>%s</p>", trimws(paras)), collapse = "")
  } else {
    paste0(
      "<p>Tier&nbsp;", tier, " (", desc, ") submission to the ",
      "<a href=\"https://janpfander.github.io/llm_predictions_megastudy/\">",
      "Silicon Sample Benchmark</a> (team <code>", team, "</code>).</p>",
      "<p>Approach family: ", family, ". Model(s): ", models,
      ". Coverage: ", ints, " interventions &times; ", outs, " preregistered outcomes.</p>",
      "<p>This record archives the prediction file(s), generation code, and method ",
      "registration form.</p>")
  }

  z <- list(
    upload_type  = "software",
    title        = sprintf("Silicon Sample Benchmark — Tier %s (%s) submission (team %s)",
                           tier, desc, team),
    version      = sprintf("1.0-t%s", tier),
    language     = "eng",
    access_right = "open",
    license      = m$license %||% "CC-BY-4.0",
    creators     = creators,
    description  = description,
    keywords     = list(
      "Silicon Sample Benchmark", "silicon sampling", "large language models",
      "computational social science", "survey methodology", "public opinion",
      "climate communication", "treatment effect prediction"),
    related_identifiers = list(
      list(identifier = "https://janpfander.github.io/llm_predictions_megastudy/",
           relation = "isPartOf", scheme = "url", resource_type = "publication-other"),
      list(identifier = "https://janpfander.github.io/llm_predictions_megastudy/amendment_preregistration.html",
           relation = "references", scheme = "url", resource_type = "publication-preprint")),
    notes = sprintf("Team %s. Disclosure class %s.", team, m$disclosure_class %||% "A"))

  # Link the generation code if a public repo is recorded.
  repo <- m$code_repository %||% ""
  if (nzchar(repo) && !grepl("your-team/your-repo", repo))
    z$related_identifiers <- c(z$related_identifiers, list(
      list(identifier = repo, relation = "isCompiledBy", scheme = "url",
           resource_type = "software")))
  z
}

write_zenodo <- function(root = ".") {
  zp <- file.path(root, ".zenodo.json")
  z  <- build_zenodo(root)
  write_json(z, zp, auto_unbox = TRUE, pretty = TRUE)
  message("zenodo_citation: wrote .zenodo.json (version ", z$version, ", ", length(z$creators),
          " creator(s)) from metadata.json. Review the ORCIDs before depositing.")
  invisible(z)
}

write_zenodo(.root)
