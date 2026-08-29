## ---------------------------------------------------------------------------
## clean_lib.R — turn a raw Tier-1 survey export into the target schema
## Library sourced by clean.R (`make clean`); not run directly.
##
## The Silicon Sample Benchmark survey is the same Qualtrics instrument the human
## study uses. A raw export therefore carries Qualtrics variable names
## (e.g. `trust_competent_1`, `policy_1_1`, `funding_5`, `donation`, `newsletter`)
## and lacks the constructed scale variables the locked analysis needs
## (`trust_multidimensional`, the `*_mean` composites, the reverse-coded
## `funding_perceptions`, `age_band`). This script reproduces exactly the recodes
## and composites in the study's own pipeline (data/cleaning.qmd :: clean_common)
## and returns a tibble in the Tier-1 submission schema.
##
## Usage:
##   source("clean_lib.R")                     # also sources submission_spec.R
##   clean <- clean_submission("my_raw_export.csv", "vienna_T1_primary_v1.csv")
##
## Expected raw columns (Qualtrics labels): see the codebook on the website.
## `condition` must already hold a canonical condition title (see submission_spec).
## `profile_id` is your own unique respondent identifier (unique within the submission).
## Demographic columns may be numeric Qualtrics codes OR already-labelled values
## (e.g. injected from your own profiles); both are handled.
## ---------------------------------------------------------------------------

suppressPackageStartupMessages(library(tidyverse))

if (!exists("sst")) {
  .here <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) NULL)
  source(file.path(if (is.null(.here)) "." else .here, "submission_spec.R"))
}

## raw Qualtrics label -> target name (only items needed for the Tier-1 schema)
.rename_map <- c(
  trust_competence_1 = "trust_competent_1",
  trust_competence_2 = "trust_intelligent_1",
  trust_competence_3 = "trust_qualified_1",
  trust_integrity_1  = "trust_honest_1",
  trust_integrity_2  = "trust_ethical_1",
  trust_integrity_3  = "trust_sincere_1",
  trust_benevolence_1 = "trust_concerned_1",
  trust_benevolence_2 = "trust_improve_1",
  trust_benevolence_3 = "trust_considerate_1",
  trust_openness_1   = "trust_feedback_1",
  trust_openness_2   = "trust_transparent_1",
  trust_openness_3   = "trust_attention_1",
  trust_post          = "trust_post_1",
  distrust_post       = "distrust_1",
  donation_ams        = "donation",
  newsletter_signup   = "newsletter",
  funding_perceptions = "funding_5",
  policy_role_1 = "policy_1_1", policy_role_2 = "policy_2_1",
  policy_role_3 = "policy_3_1", policy_role_4 = "policy_4_1",
  inst_trust_epa = "inst_trust_epa_1", inst_trust_nasa = "inst_trust_nasa_1",
  inst_trust_noaa = "inst_trust_noaa_1",
  inst_trust_universities = "inst_trust_uni_1",
  inst_trust_federal_gov  = "inst_trust_gov_1",
  belief_post = "belief_post_1",
  concern_1 = "concern_1_1", concern_2 = "concern_2_1", concern_3 = "concern_3_1",
  policy_general = "policy_general_1",
  policy_specific_1 = "policy_specific_1_1", policy_specific_2 = "policy_specific_2_1",
  policy_specific_3 = "policy_specific_3_1", policy_specific_4 = "policy_specific_4_1",
  policy_specific_5 = "policy_specific_5_1", policy_specific_6 = "policy_specific_6_1",
  policy_specific_7 = "policy_specific_7_1",
  behavior_meat = "individual_meat_1", behavior_transport = "individual_transport_1",
  behavior_solar = "individual_solar_1", behavior_fly = "individual_fly_1",
  behavior_talk = "individual_talk_1", behavior_donate = "individual_donate_1"
)

## numeric Qualtrics code -> label, tolerant of values that are already labels.
## Unrecognized non-missing values are a hard error, not a silent NA: a label
## that doesn't exactly match the spec (see submission_spec.R) would otherwise
## drop the respondent from every subgroup analysis without anyone noticing.
.recode_demo <- function(x, map, what) {
  x   <- trimws(as.character(x))
  lab <- unname(map[x])
  out <- ifelse(!is.na(lab), lab, ifelse(x %in% unname(map), x, NA_character_))
  bad <- unique(x[is.na(out) & !is.na(x) & nzchar(x)])
  if (length(bad))
    stop("clean_submission(): unrecognized ", what, " value(s): ",
         paste0('"', bad, '"', collapse = ", "),
         "\nAllowed labels (exact strings): ",
         paste0('"', unique(unname(map)), '"', collapse = ", "),
         call. = FALSE)
  out
}

## Numeric codes are the values a Qualtrics export carries (party uses Qualtrics
## RecodeValues, so exported codes are 1=Republican, 2=Democrat, 3=Independent,
## 4=Other even though the on-screen choice order is Rep/Ind/Dem/Other). The
## extra label entries are the survey's on-screen wordings where they differ
## from the canonical submission strings (e.g. hyphenated race labels), so a
## team that injects the survey's own labels also cleans correctly.
.gender_map <- c("1" = "Male", "2" = "Female", "3" = "Other")
.race_map   <- c("1" = "White / Caucasian", "2" = "Black / African American",
                 "3" = "Hispanic / Latino", "4" = "Asian / Asian American",
                 "5" = "Other",
                 "Black / African-American" = "Black / African American",
                 "Latino / Hispanic"        = "Hispanic / Latino",
                 "Asian / Asian-American"   = "Asian / Asian American")
.edu_map    <- c("1" = "Less than high school",
                 "2" = "High school diploma / GED",
                 "3" = "Some college or Associate's degree",
                 "4" = "Bachelor's degree",
                 "5" = "Master's degree / Professional degree",
                 "6" = "Doctorate degree / Ph.D.")
.income_map <- c("1" = "Less than $30,000", "2" = "$30,000 to $55,999",
                 "3" = "$56,000 to $99,999", "4" = "$100,000 to $167,999",
                 "5" = "$168,000 or more")
.party_map  <- c("1" = "Republican", "2" = "Democrat",
                 "3" = "Independent", "4" = "Other",
                 "Other (please specify)" = "Other")

## Birth-year parser, ported verbatim from data/cleaning.qmd
.extract_birth_year <- function(x) {
  year_pat   <- "19\\d{2}|200[0-9]"
  candidates <- str_extract_all(x, paste0("\\b(", year_pat, ")\\b"))
  map_dbl(seq_along(x), \(i) {
    cands <- candidates[[i]]
    if (length(cands) == 1L) return(as.numeric(cands))
    if (length(cands) > 1L) {
      born_in <- str_extract(x[i], paste0("born in (", year_pat, ")"), group = 1)
      if (!is.na(born_in)) return(as.numeric(born_in))
      return(as.numeric(tail(cands, 1L)))
    }
    age <- as.numeric(str_extract(x[i], "\\d+"))
    if (!is.na(age) && age >= 18 && age <= 100) return(2026 - age)
    NA_real_
  })
}

.to_binary <- function(x) {
  s  <- tolower(trimws(as.character(x)))
  nx <- suppressWarnings(as.numeric(s))
  case_when(
    nx == 1 | s %in% c("yes", "true")  ~ 1L,
    nx %in% c(0, 2) | s %in% c("no", "false") ~ 0L,
    TRUE ~ NA_integer_
  )
}

## A real Qualtrics export carries two extra header rows under the column names
## (the question text, then an {"ImportId":...} row) plus ~17 system columns.
## Drop those metadata rows so a real export reads the same as a plain one; the
## system columns fall away later via select(all_of(sst$tier1_required)).
strip_qualtrics_header <- function(df) {
  n_check <- min(2L, nrow(df))
  if (n_check == 0L) return(df)
  is_hdr <- vapply(seq_len(n_check), function(i)
    any(grepl("ImportId", as.character(unlist(df[i, ])), fixed = TRUE)), logical(1))
  if (any(is_hdr)) df[-seq_len(max(which(is_hdr))), , drop = FALSE] else df
}

clean_submission <- function(input, output = NULL) {
  raw <- if (is.data.frame(input)) input else
    readr::read_csv(input, col_types = cols(.default = col_character()),
                    show_col_types = FALSE)

  raw <- strip_qualtrics_header(raw)
  d   <- raw |> rename(any_of(.rename_map))

  needed <- c("profile_id", "condition", "gender", "year_birth", "race",
              "education", "income", "party",
              sst$trust_items, "trust_post", "distrust_post",
              "funding_perceptions", paste0("policy_role_", 1:4),
              "inst_trust_epa", "inst_trust_nasa", "inst_trust_noaa",
              "inst_trust_universities", "inst_trust_federal_gov",
              "belief_post", paste0("concern_", 1:3), "policy_general",
              paste0("policy_specific_", 1:7),
              "behavior_meat", "behavior_transport", "behavior_solar",
              "behavior_fly", "behavior_talk", "behavior_donate",
              "donation_ams", "newsletter_signup")
  missing <- setdiff(needed, names(d))
  if (length(missing))
    stop("clean_submission(): missing required raw columns after renaming:\n  ",
         paste(missing, collapse = ", "),
         "\nCheck the codebook for the expected Qualtrics column names.",
         call. = FALSE)

  num_vars <- c(sst$trust_items, "trust_post", "distrust_post",
                paste0("policy_role_", 1:4),
                "inst_trust_epa", "inst_trust_nasa", "inst_trust_noaa",
                "inst_trust_universities", "inst_trust_federal_gov",
                "belief_post", paste0("concern_", 1:3), "policy_general",
                paste0("policy_specific_", 1:7),
                "behavior_meat", "behavior_transport", "behavior_solar",
                "behavior_fly", "behavior_talk", "behavior_donate")

  ## Map raw survey code names (incl. the four semicolon-joined multi-pair
  ## names) to canonical titles, so a genuine Qualtrics re-run of the shipped
  ## survey cleans out of the box. Values that already hold a canonical title
  ## pass through; anything else is a hard error, not a silent NA.
  d <- d |>
    mutate(condition = {
      x   <- str_squish(as.character(condition))
      out <- ifelse(x %in% names(sst$codenames), unname(sst$codenames[x]), x)
      bad <- setdiff(unique(out[!is.na(out) & nzchar(out)]), sst$conditions)
      if (length(bad))
        stop("clean_submission(): unrecognized condition value(s): ",
             paste0('"', bad, '"', collapse = ", "),
             "\nConditions must be canonical titles or raw survey code names ",
             "(see survey/condition_codenames.csv — join on the FULL code name; ",
             "semicolons are part of the name, do not split it).",
             call. = FALSE)
      out
    })

  d |>
    mutate(
      age      = 2026 - .extract_birth_year(year_birth),
      gender   = .recode_demo(gender, .gender_map, "gender"),
      race     = .recode_demo(race, .race_map, "race"),
      education = .recode_demo(education, .edu_map, "education"),
      income   = .recode_demo(income, .income_map, "income"),
      party    = .recode_demo(party, .party_map, "party"),
      across(all_of(num_vars), ~ suppressWarnings(as.numeric(.x))),
      donation_ams        = suppressWarnings(as.numeric(donation_ams)),
      funding_perceptions = 100 - suppressWarnings(as.numeric(funding_perceptions)),
      newsletter_signup   = .to_binary(newsletter_signup)
    ) |>
    mutate(
      trust_competence  = rowMeans(pick(trust_competence_1, trust_competence_2, trust_competence_3),   na.rm = TRUE),
      trust_integrity   = rowMeans(pick(trust_integrity_1, trust_integrity_2, trust_integrity_3),       na.rm = TRUE),
      trust_benevolence = rowMeans(pick(trust_benevolence_1, trust_benevolence_2, trust_benevolence_3), na.rm = TRUE),
      trust_openness    = rowMeans(pick(trust_openness_1, trust_openness_2, trust_openness_3),          na.rm = TRUE),
      trust_multidimensional = rowMeans(
        pick(trust_competence, trust_integrity, trust_benevolence, trust_openness), na.rm = TRUE),
      policy_role_mean = rowMeans(pick(policy_role_1, policy_role_2, policy_role_3, policy_role_4), na.rm = TRUE),
      inst_trust_mean  = rowMeans(pick(inst_trust_epa, inst_trust_nasa, inst_trust_noaa,
                                       inst_trust_universities, inst_trust_federal_gov), na.rm = TRUE),
      concern_mean         = rowMeans(pick(concern_1, concern_2, concern_3), na.rm = TRUE),
      policy_specific_mean = rowMeans(pick(all_of(paste0("policy_specific_", 1:7))), na.rm = TRUE),
      behavior_mean        = rowMeans(pick(behavior_meat, behavior_transport, behavior_solar,
                                           behavior_fly, behavior_talk, behavior_donate), na.rm = TRUE)
    ) |>
    mutate(
      condition = factor(condition, levels = sst$conditions),
      gender    = factor(gender, levels = sst$moderators$gender),
      race      = factor(race, levels = sst$moderators$race),
      education = factor(education, levels = sst$moderators$education),
      income    = factor(income, levels = sst$moderators$income),
      party     = factor(party, levels = sst$moderators$party),
      age_band  = cut(age, breaks = c(17, 29, 44, 59, Inf),
                      labels = sst$moderators$age_band, right = TRUE)
    ) |>
    select(all_of(sst$tier1_required)) ->
  out

  ## Surface effective sample sizes so silent data loss is visible: a condition
  ## or moderator that lost rows to NA here would be quietly excluded from the
  ## corresponding analyses at scoring time.
  n_cond <- table(out$condition)
  message("clean_submission(): per-condition N — ",
          if (min(n_cond) == max(n_cond)) paste0(min(n_cond), " in every condition")
          else paste0("min ", min(n_cond), " (", names(n_cond)[which.min(n_cond)],
                      "), max ", max(n_cond), " (", names(n_cond)[which.max(n_cond)], ")"))
  for (mod in c(names(sst$moderators))) {
    n_na <- sum(is.na(out[[mod]]))
    if (n_na > 0)
      message("clean_submission(): NOTE — ", mod, " is NA for ", n_na, " of ",
              nrow(out), " rows; these rows drop out of every ", mod,
              " subgroup analysis")
  }

  if (!is.null(output)) {
    readr::write_csv(out, output)
    message("clean_submission(): wrote ", nrow(out), " rows to ", output)
  }
  out
}
