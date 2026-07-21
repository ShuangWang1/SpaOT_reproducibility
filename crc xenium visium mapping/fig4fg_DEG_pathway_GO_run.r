library(clusterProfiler)
library(enrichplot)
library(org.Hs.eg.db)
library(ReactomePA)
library(dplyr)
library(ggplot2)

outdir <- "CRC_GO&Pathway_SPP1"
dir.create(outdir, showWarnings = FALSE)

deg <- read.csv("DEGs_SPP1.csv")

deg_sig <- deg %>%
  filter(!is.na(names), !is.na(logfoldchanges), !is.na(pvals_adj)) %>%
  filter(pvals_adj < 0.05)

up_genes   <- deg_sig %>% filter(logfoldchanges > 0) %>% pull(names) %>% unique()
down_genes <- deg_sig %>% filter(logfoldchanges < 0) %>% pull(names) %>% unique()

all_genes   <- deg_sig %>% pull(names) %>% unique()

up_entrez   <- bitr(up_genes,   fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db) %>% pull(ENTREZID) %>% unique()
down_entrez <- bitr(down_genes, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db) %>% pull(ENTREZID) %>% unique()
all_entrez  <- bitr(all_genes,  fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db) %>% pull(ENTREZID) %>% unique()

# GO BP enrichment (no universe = full genome background)
ego_up <- enrichGO(gene = up_entrez, OrgDb = org.Hs.eg.db,
                   keyType = "ENTREZID", ont = "BP", pAdjustMethod = "BH",
                   pvalueCutoff = 0.1, qvalueCutoff = 0.3, readable = TRUE)

ego_down <- enrichGO(gene = down_entrez, OrgDb = org.Hs.eg.db,
                     keyType = "ENTREZID", ont = "BP", pAdjustMethod = "BH",
                     pvalueCutoff = 0.1, qvalueCutoff = 0.3, readable = TRUE)

# KEGG enrichment (no universe)
ekk_up <- enrichKEGG(gene = up_entrez, organism = "hsa",
                     pAdjustMethod = "BH", pvalueCutoff = 0.1)

ekk_down <- enrichKEGG(gene = down_entrez, organism = "hsa",
                       pAdjustMethod = "BH", pvalueCutoff = 0.1)

# Reactome enrichment (no universe)
er_up <- enrichPathway(gene = up_entrez, organism = "human",
                       pAdjustMethod = "BH", pvalueCutoff = 0.1, readable = TRUE)

er_down <- enrichPathway(gene = down_entrez, organism = "human",
                         pAdjustMethod = "BH", pvalueCutoff = 0.1, readable = TRUE)

# Combined (all significant DEGs)
ego_all <- enrichGO(gene = all_entrez, OrgDb = org.Hs.eg.db,
                    keyType = "ENTREZID", ont = "BP", pAdjustMethod = "BH",
                    pvalueCutoff = 0.1, qvalueCutoff = 0.3, readable = TRUE)

ekk_all <- enrichKEGG(gene = all_entrez, organism = "hsa",
                      pAdjustMethod = "BH", pvalueCutoff = 0.1)

er_all <- enrichPathway(gene = all_entrez, organism = "human",
                        pAdjustMethod = "BH", pvalueCutoff = 0.1, readable = TRUE)

# Save CSV results
write.csv(as.data.frame(ego_up),   file.path(outdir, "GO_BP_up.csv"),   row.names = FALSE)
write.csv(as.data.frame(ego_down), file.path(outdir, "GO_BP_down.csv"), row.names = FALSE)
write.csv(as.data.frame(ekk_up),   file.path(outdir, "KEGG_up.csv"),   row.names = FALSE)
write.csv(as.data.frame(ekk_down), file.path(outdir, "KEGG_down.csv"), row.names = FALSE)
write.csv(as.data.frame(er_up),    file.path(outdir, "Reactome_up.csv"),   row.names = FALSE)
write.csv(as.data.frame(er_down),  file.path(outdir, "Reactome_down.csv"), row.names = FALSE)
write.csv(as.data.frame(ego_all),  file.path(outdir, "GO_BP_combined.csv"),      row.names = FALSE)
write.csv(as.data.frame(ekk_all),  file.path(outdir, "KEGG_combined.csv"),       row.names = FALSE)
write.csv(as.data.frame(er_all),   file.path(outdir, "Reactome_combined.csv"),   row.names = FALSE)

save_plot <- function(p, name, w = 10, h = 8) {
  ggsave(file.path(outdir, name), plot = p, width = w, height = h)
}

# GO BP plots
if (!is.null(ego_up) && nrow(as.data.frame(ego_up)) > 0) {
  save_plot(dotplot(ego_up,   showCategory = 15) + ggtitle("GO BP - Upregulated"),   "GO_BP_up_dotplot.pdf")
  save_plot(barplot(ego_up,   showCategory = 15) + ggtitle("GO BP - Upregulated"),   "GO_BP_up_barplot.pdf")
}
if (!is.null(ego_down) && nrow(as.data.frame(ego_down)) > 0) {
  save_plot(dotplot(ego_down, showCategory = 15) + ggtitle("GO BP - Downregulated"), "GO_BP_down_dotplot.pdf")
  save_plot(barplot(ego_down, showCategory = 15) + ggtitle("GO BP - Downregulated"), "GO_BP_down_barplot.pdf")
}

# KEGG plots
if (!is.null(ekk_up) && nrow(as.data.frame(ekk_up)) > 0) {
  save_plot(dotplot(ekk_up,   showCategory = 15) + ggtitle("KEGG - Upregulated"),   "KEGG_up_dotplot.pdf")
  save_plot(barplot(ekk_up,   showCategory = 15) + ggtitle("KEGG - Upregulated"),   "KEGG_up_barplot.pdf")
}
if (!is.null(ekk_down) && nrow(as.data.frame(ekk_down)) > 0) {
  save_plot(dotplot(ekk_down, showCategory = 15) + ggtitle("KEGG - Downregulated"), "KEGG_down_dotplot.pdf")
  save_plot(barplot(ekk_down, showCategory = 15) + ggtitle("KEGG - Downregulated"), "KEGG_down_barplot.pdf")
}

# Reactome plots
if (!is.null(er_up) && nrow(as.data.frame(er_up)) > 0) {
  save_plot(dotplot(er_up,   showCategory = 15) + ggtitle("Reactome - Upregulated"),   "Reactome_up_dotplot.pdf")
  save_plot(barplot(er_up,   showCategory = 15) + ggtitle("Reactome - Upregulated"),   "Reactome_up_barplot.pdf")
}
if (!is.null(er_down) && nrow(as.data.frame(er_down)) > 0) {
  save_plot(dotplot(er_down, showCategory = 15) + ggtitle("Reactome - Downregulated"), "Reactome_down_dotplot.pdf")
  save_plot(barplot(er_down, showCategory = 15) + ggtitle("Reactome - Downregulated"), "Reactome_down_barplot.pdf")
}

# Combined plots
if (!is.null(ego_all) && nrow(as.data.frame(ego_all)) > 0) {
  save_plot(dotplot(ego_all, showCategory = 15) + ggtitle("GO BP - Combined"), "GO_BP_combined_dotplot.pdf")
  save_plot(barplot(ego_all, showCategory = 15) + ggtitle("GO BP - Combined"), "GO_BP_combined_barplot.pdf")
}
if (!is.null(ekk_all) && nrow(as.data.frame(ekk_all)) > 0) {
  save_plot(dotplot(ekk_all, showCategory = 15) + ggtitle("KEGG - Combined"), "KEGG_combined_dotplot.pdf")
  save_plot(barplot(ekk_all, showCategory = 15) + ggtitle("KEGG - Combined"), "KEGG_combined_barplot.pdf")
}
if (!is.null(er_all) && nrow(as.data.frame(er_all)) > 0) {
  save_plot(dotplot(er_all, showCategory = 15) + ggtitle("Reactome - Combined"), "Reactome_combined_dotplot.pdf")
  save_plot(barplot(er_all, showCategory = 15) + ggtitle("Reactome - Combined"), "Reactome_combined_barplot.pdf")
}

message("Done. Results saved to: ", outdir)
