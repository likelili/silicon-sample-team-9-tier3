#!/usr/bin/env Rscript
## Entry point: validate the whole submission repo.  `make check` / `Rscript scripts/check.R`
.a    <- commandArgs(FALSE)
.dir  <- dirname(normalizePath(sub("^--file=", "", .a[grep("^--file=", .a)])))
.root <- dirname(.dir)
source(file.path(.dir, "lib", "check_lib.R"))
res <- check_repo(.root)
if (any(res$status == "FAIL")) quit(status = 1)
