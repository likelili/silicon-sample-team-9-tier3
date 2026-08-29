## ---------------------------------------------------------------------------
## submission_spec.R — canonical schema for Silicon Sample Benchmark submissions
##
## Single source of truth shared by clean_lib.R, check_lib.R, and
## the example-data generator. Sourcing this file defines the `sst` list.
##
## Condition labels are the 16 text-intervention titles + "control", i.e. the
## titles in data/interventions.csv minus the 4 interactive arms (3 LLM-chatbot
## conditions + the "Value similarity" quiz). Edit here and nowhere else.
## ---------------------------------------------------------------------------

sst <- local({

  interventions <- c(
    "Corporate reliance",
    "Social justice",
    "Interview Prof. Maraun",
    "Funding",
    "Oil industry misinformation",
    "Measurement & modeling (1)",
    "Former skeptics",
    "High public trust",
    "Measurement & modeling (2)",
    "Peer-review",
    "Scientist community helpers",
    "Consensus",
    "Portrait Prof. Cherry",
    "Model accuracy",
    "Interview Prof. Sebille",
    "Extreme weather predictions"
  )

  conditions <- c("control", interventions)

  ## Raw survey code name -> canonical condition title. The raw survey files
  ## (survey.qsf / survey.json) key conditions by these internal code names;
  ## four are semicolon-joined multi-pair names — the semicolons are part of
  ## the name, never split them. survey/condition_codenames.csv mirrors this
  ## map for human readers; this vector is the canonical version.
  codenames <- c(
    "control neckties" = "control",
    "control baseball" = "control",
    "control dances"   = "control",
    "practical planarian"   = "Extreme weather predictions",
    "complicated cockroach" = "Portrait Prof. Cherry",
    "flimsy fish"           = "Interview Prof. Maraun",
    "honored haddock"       = "Peer-review",
    "jealous jaguar"        = "Consensus",
    "phony parrotfish"      = "Funding",
    "crushing chicken; gross grasshopper; homely halibut" = "High public trust",
    "worse wildfowl"        = "Oil industry misinformation",
    "periwinkle partridge"  = "Scientist community helpers",
    "difficult dog"         = "Social justice",
    "giant gibbon; brick bobcat"          = "Corporate reliance",
    "limping llama; friendly frog"        = "Former skeptics",
    "perfect prawn"                       = "Measurement & modeling (1)",
    "orchid orangutan; defiant dragonfly" = "Measurement & modeling (2)",
    "apple aardvark"        = "Model accuracy",
    "heartfelt hummingbird" = "Interview Prof. Sebille"
  )

  trust_items <- c(
    paste0("trust_competence_",  1:3),
    paste0("trust_integrity_",   1:3),
    paste0("trust_benevolence_", 1:3),
    paste0("trust_openness_",    1:3)
  )

  ## 13 preregistered outcomes (the 12 trust items are sub-components of the
  ## primary, shipped in Tier 1 but not counted among the 13).
  outcomes <- c(
    "trust_multidimensional",
    "trust_post", "distrust_post", "funding_perceptions",
    "policy_role_mean", "inst_trust_mean",
    "belief_post", "concern_mean", "policy_general",
    "policy_specific_mean", "behavior_mean",
    "donation_ams", "newsletter_signup"
  )

  ## scale type per outcome (drives value-sanity checks)
  scale_0_100 <- c(
    "trust_multidimensional", "trust_post", "distrust_post",
    "funding_perceptions", "policy_role_mean", "inst_trust_mean",
    "belief_post", "concern_mean", "policy_general",
    "policy_specific_mean", "behavior_mean"
  )

  moderators <- list(
    gender    = c("Male", "Female", "Other"),
    age_band  = c("18-29", "30-44", "45-59", "60+"),
    race      = c("White / Caucasian", "Black / African American",
                  "Hispanic / Latino", "Asian / Asian American", "Other"),
    education = c("Less than high school",
                  "High school diploma / GED",
                  "Some college or Associate's degree",
                  "Bachelor's degree",
                  "Master's degree / Professional degree",
                  "Doctorate degree / Ph.D."),
    income    = c("Less than $30,000", "$30,000 to $55,999",
                  "$56,000 to $99,999", "$100,000 to $167,999",
                  "$168,000 or more"),
    party     = c("Republican", "Democrat", "Independent", "Other")
  )

  ## Tier-1 required columns (one row per synthetic respondent)
  tier1_required <- c(
    "profile_id", "condition", names(moderators),
    "trust_multidimensional", trust_items,
    "trust_post", "distrust_post", "funding_perceptions",
    "policy_role_mean", "inst_trust_mean",
    "belief_post", "concern_mean", "policy_general",
    "policy_specific_mean", "behavior_mean",
    "donation_ams", "newsletter_signup"
  )

  tier2_main_cols <- c("condition", "outcome", "mean")
  tier2_mod_cols  <- c("condition", "moderator", "moderator_level",
                       "outcome", "mean")
  tier3_cols      <- c("condition", "outcome", "ate")

  list(
    interventions   = interventions,
    conditions      = conditions,
    codenames       = codenames,
    trust_items     = trust_items,
    outcomes        = outcomes,
    scale_0_100     = scale_0_100,
    donation_range  = c(0, 10),
    moderators      = moderators,
    tier1_required  = tier1_required,
    tier2_main_cols = tier2_main_cols,
    tier2_mod_cols  = tier2_mod_cols,
    tier3_cols      = tier3_cols
  )
})
