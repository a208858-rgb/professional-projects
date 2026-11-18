library(phyloseq)
library(tidyverse)

# Prepare genus-level cohort CSVs from the cleaned phyloseq object.
ps <- readRDS("physeq_clean_n=598.rds")

tax <- as.data.frame(tax_table(ps), stringsAsFactors = FALSE)
otu <- as(otu_table(ps), "matrix")
if (!taxa_are_rows(ps)) otu <- t(otu)

gen <- as.character(tax[rownames(otu), "genus"])
ok <- !is.na(gen) & gen != ""
gen_f <- factor(gen[ok], levels = unique(gen[ok]))
otu_g <- rowsum(otu[ok, , drop = FALSE], group = gen_f, reorder = FALSE)
cat("Genus used as features:", nrow(otu_g), "\n")

meta <- as.data.frame(sample_data(ps))[colnames(otu_g), , drop = FALSE]
dat <- data.frame(
  Label = ifelse(meta$Weight_Group == "Normal", 0, 1),
  meta,
  t(otu_g),
  check.names = FALSE
)
dat <- dat[meta$Weight_Group %in% c("Normal", "Obese_Overweight"), , drop = FALSE]

cohorts <- list(
  overall = dat,
  malay = dat[dat$Ethnicity == "Malay", , drop = FALSE],
  chinese = dat[dat$Ethnicity == "Chinese", , drop = FALSE]
)
print(sapply(cohorts, nrow))

for (nm in names(cohorts)) {
  write.csv(cohorts[[nm]], paste0("cohort_", nm, ".csv"), row.names = TRUE)
}
