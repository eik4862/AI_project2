### LOAD LIBRARIES
library(ggplot2)
library(ggsci)
library(cowplot)
library(plyr)
library(dplyr)
library(latex2exp)

### GRAPHICS SETTING
colors <- c('#3170BC', '#E8C241', '#BE5B51', '#8E7800', '#043967', '#83A5D8', '#868686', '#3B3B3B', '#A5302F', 
            '#4B6890')

### PLOTING FUNCTIONS
plot.signal <- function(amp, freq, t, fs, color = '#3170BC') {
  sampling_t <- seq(0, t, length.out = floor(fs * t))
  signal <- amp * sin(2 * pi * freq * sampling_t)
  plot <- data.frame(t = 1000 * sampling_t, signal = signal) %>%
    ggplot(aes(x = t, y = signal)) +
    geom_line(size = 1, color = color) +
    labs(x = 'Time (ms)', y = TeX('Signal ($\\mu$V)')) +
    theme_classic()
  
  return(plot)
}

plot.noise <- function(amp, t, fs, color = '#3170BC') {
  sampling_t <- seq(0, t, length.out = floor(fs * t))
  noise <- amp * rnorm(length(sampling_t))
  plot <- data.frame(t = 1000 * sampling_t, signal = noise) %>%
    ggplot(aes(x = t, y = signal)) +
    geom_line(size = 1, color = color) +
    labs(x = 'Time (ms)', y = TeX('Signal ($\\mu$V)')) +
    theme_classic()
}

plot.feature <- function(data, target_col, ylab, n_sample = 1) {
  colnames(data)[target_col] = 'feature'
  plot <- data %>%
    ggplot(aes(x = t, y = feature, color = label)) +
    geom_line(size = 1) +
    labs(x = 'Time', y = ylab, color = '') +
    theme_classic() +
    scale_color_manual(values = colors)
  
  return(plot)
}

plot.HMM.SVM.tune <- function(data) {
  max.mask <- data$F1.mean == max(data$F1.mean)
  plot <- data %>%
    ggplot(aes(x = log10(gamma), y = log10(C), fill = F1.mean)) +
    geom_tile() +
    geom_point(data = data[max.mask,], shape = 4, color = 'red', size = 2) +
    scale_x_continuous(breaks = -9:3) +
    scale_y_continuous(breaks = -3:9) +
    labs(x = TeX('$\\gamma$'), y = 'C', fill = '') +
    theme_classic() +
    scale_fill_viridis_c()
  
  return(plot)
}

plot.roc <- function(data) {
  plot <- data %>%
    ggplot(aes(x = fpr, y = tpr, color = classifier, group = classifier)) +
    geom_line(size = 1) +
    labs(x = 'FPR', y = 'TPR', color = '') +
    theme_classic() +
    theme(legend.position = 'bottom') +
    scale_color_manual(values = colors)
  
  return(plot)
}

plot.stat <- function(data, target_col, ylab) {
  colnames(data)[target_col] <- 'stat'
  plot <- data %>%
    ggplot(aes(x = classifier, y = stat, fill = classifier)) +
    geom_bar(stat = 'identity') +
    geom_text(aes(label = round(stat, 2), vjust = 2), size = 5) +
    labs(x = '', y = ylab, fill = '') +
    theme_classic() +
    theme(legend.position = 'bottom') +
    scale_fill_manual(values = colors)
  
  return(plot)
}

strip.labs <- function(plot, x = T, y = T, legend = T) {
  if (x) {plot <- plot + labs(x = '')}
  if (y) {plot <- plot + labs(y = '')}
  if (legend) {plot <- plot + theme(legend.position = 'None')}
  
  return(plot)
}

### PLOTS
# Simualation
signal.1 <- plot.signal(1, 1000, .01, 100000, color = colors[2]) + coord_cartesian(ylim = c(-2, 2))
signal.2 <- plot.signal(.5, 2000, .01, 100000, color = colors[3]) + coord_cartesian(ylim = c(-2, 2))
signal.3 <- plot.signal(2, 500, .01, 100000, color = colors[4]) + coord_cartesian(ylim = c(-2, 2))
noise <- plot.noise(.1, .01, 100000, color = colors[5]) + coord_cartesian(ylim = c(-2, 2))

# 599 * 526
plot_grid(signal.1, strip.labs(signal.2), strip.labs(signal.3), strip.labs(noise), nrow = 4, ncol = 1)

# 599 * 263
sampling_t <- seq(0, .01, length.out = 1000)
signal.1 <- sin(2000 * pi * sampling_t) + .5 * sin(4000 * pi * sampling_t) + 2 * sin(1000 * pi * sampling_t)
signal.2 <- sin(2000 * pi * sampling_t) + 2 * sin(1000 * pi * sampling_t)

data.frame(t = sampling_t, signal = signal.1 + .1 * rnorm(1000)) %>%
  ggplot(aes(x = 1000 * t, y = signal)) +
  geom_line(size = 1, color = colors[1]) +
  labs(x = 'Time (ms)', y = TeX('Signal ($\\mu$V)')) +
  theme_classic()
data.frame(t = sampling_t, signal = signal.2 + .1 * rnorm(1000)) %>%
  ggplot(aes(x = 1000 * t, y = signal)) +
  geom_line(size = 1, color = colors[1]) +
  labs(x = 'Time (ms)', y = TeX('Signal ($\\mu$V)')) +
  theme_classic()

data_psd <- read.csv('./Exports/psd.csv', colClasses = rep('numeric', 2))
data_csd <- read.csv('./Exports/csd.csv', colClasses = rep('numeric', 2))

# 599 * 263
data_psd[1:250,] %>% ggplot(aes(x = f / 1000, y = log10(psd))) +
  geom_line(size = 1, color = colors[1]) +
  labs(x = 'Frequency (kHz)', y = TeX('log PSD (log $\\mu V^2/Hz$)')) +
  theme_classic()

data_csd[1:250,] %>% ggplot(aes(x = f / 1000, y = log10(csd))) +
  geom_line(size = 1, color = colors[1]) +
  labs(x = 'Frequency (kHz)', y = TeX('log CSD (log $\\mu V^2/Hz$)')) +
  theme_classic()

data_signal <- read.csv('./Exports/signal.csv', colClasses = rep('numeric', 2))
data_psd_band <- read.csv('./Exports/psd_band.csv', colClasses = rep('numeric', 2))

# 599 * 263
signal.comlex <- data_signal %>% ggplot(aes(x = t * 1000, y = signal)) +
  geom_line(size = 1, color = colors[1]) +
  labs(x = 'Time (ms)', y = TeX('Signal ($\\mu$V)')) +
  theme_classic()
signal.simple <- plot.signal(1, 1000, .01, 100000, color = colors[2])

plot_grid(signal.comlex, strip.labs(signal.simple), nrow = 2, ncol = 1)

data_psd_band %>% ggplot(aes(x = f, y = psd)) +
  geom_ribbon(data = data_psd_band[1:5,], aes(ymin = 0, ymax = psd), alpha = .4, fill = colors[2]) +
  geom_ribbon(data = data_psd_band[5:13,], aes(ymin = 0, ymax = psd), alpha = .4, fill = colors[3]) +
  geom_ribbon(data = data_psd_band[13:31,], aes(ymin = 0, ymax = psd), alpha = .4, fill = colors[4]) +
  geom_ribbon(data = data_psd_band[31:51,], aes(ymin = 0, ymax = psd), alpha = .4, fill = colors[5]) +
  geom_ribbon(data = data_psd_band[51:101,], aes(ymin = 0, ymax = psd), alpha = .4, fill = colors[6]) +
  geom_line(size = 1, color = colors[1]) +
  labs(x = 'Frequency (Hz)', y = TeX('PSD ($\\mu V^2/Hz$)')) +
  theme_classic()

data_psd_simple <- read.csv('./Exports/psd_simple.csv', colClasses = rep('numeric', 2))
data_psd_complex <- read.csv('./Exports/psd_complex.csv', colClasses = rep('numeric', 2))

psd.complex <- data_psd_complex %>% ggplot(aes(x = f / 1000, y = psd)) +
  geom_line(size = 1, color = colors[1]) +
  labs(x = 'Frequency (kHz)', y = TeX('PSD ($\\mu V^2/Hz$)')) +
  theme_classic()

psd.simple <- data_psd_simple %>% ggplot(aes(x = f / 1000, y = psd)) +
  geom_line(size = 1, color = colors[2]) +
  labs(x = 'Frequency (kHz)', y = TeX('PSD ($\\mu V^2/Hz$)')) +
  theme_classic()

plot_grid(psd.complex, strip.labs(psd.simple), nrow = 2, ncol = 1)

# 599 * 263
plot.line(data_power$f, data_power$psd, xlab = 'Frequency (Hz)', ylab = TeX('PSD ($\\mu V^2/Hz$)'))

data_fm <- read.csv('./Exports/fm.csv', colClasses = rep('numeric', 2))
data_psd_fm <- read.csv('./Exports/psd_fm.csv', colClasses = c('factor', rep('numeric', 2)))
data_psd_fm_split <- split(data_psd_fm, data_psd_fm$window)

# 300 * 132
plot.line(data_fm$t, data_fm$signal, xlab = 'Time (s)', ylab = TeX('Signal ($\\mu V$)'))

# 300 * 132
plot.line(data_psd_fm_split[[1]]$f[1:50], data_psd_fm_split[[1]]$psd[1:50], colors[1], 'Frequency (Hz)', 
          TeX('PSD ($\\mu V^2/Hz$)'))

# 300 * 132
plot.line(data_psd_fm_split[[2]]$f[1:50], data_psd_fm_split[[2]]$psd[1:50], colors[2], 'Frequency (Hz)', 
          TeX('PSD ($\\mu V^2/Hz$)'))

# 300 * 132
plot.line(data_psd_fm_split[[3]]$f[1:50], data_psd_fm_split[[3]]$psd[1:50], colors[3], 'Frequency (Hz)', 
          TeX('PSD ($\\mu V^2/Hz$)'))

# Feature
data_eeg <- read.csv('./Exports/data_eeg.csv', colClasses = c(rep('factor', 2), rep('numeric', 65)))

# 6 * 9
eegs <- list()
for (i in 1:10) {
  eegs[[i]] <- plot.feature(data_eeg, i + 3, paste0('Chan.', i))
}
panel.A <- plot_grid(strip.labs(eegs[[1]], y = F), strip.labs(eegs[[2]], y = F), strip.labs(eegs[[3]], y = F), 
                     strip.labs(eegs[[4]], y = F), strip.labs(eegs[[5]], y = F), nrow = 5, ncol = 1)
panel.B <- plot_grid(strip.labs(eegs[[6]], y = F), strip.labs(eegs[[7]], y = F), strip.labs(eegs[[8]], y = F), 
                     strip.labs(eegs[[9]], y = F), strip.labs(eegs[[10]], y = F), nrow = 5, ncol = 1)
panel.C <- plot_grid(panel.A, panel.B, nrow = 1, ncol = 2)
plot_grid(panel.C, get_legend(eegs[[1]] + theme(legend.position = 'bottom')), nrow = 2, ncol = 1,
          rel_heights = c(5, .3))

# AUC
data_roc_hmm_svm <- read.csv('./Exports/roc_HMM_SVM.csv', colClasses = rep('numeric', 2))
data_roc_knn <- read.csv('./Exports/roc_KNN.csv', colClasses = rep('numeric', 2))
data_roc_naive_bayes <- read.csv('./Exports/roc_NB.csv', colClasses = rep('numeric', 2))
data_roc_hmm_svm$classifier <- rep('HMM-SVM', nrow(data_roc_hmm_svm))
data_roc_knn$classifier <- rep('KNN', nrow(data_roc_knn))
data_roc_naive_bayes$classifier <- rep('NB', nrow(data_roc_naive_bayes))
data_roc <- rbind(data_roc_hmm_svm, data_roc_knn, data_roc_naive_bayes)
data_roc$classifier <- factor(data_roc$classifier, levels = c('HMM-SVM', 'KNN', 'NB'))
roc <- plot.roc(data_roc)

data_roc_hmm_eeg <- read.csv('./Exports/roc_HMM_eeg.csv', colClasses = rep('numeric', 2))
data_roc_hmm_stats <- read.csv('./Exports/roc_HMM_stats.csv', colClasses = rep('numeric', 2))
data_roc_hmm_cross_bp <- read.csv('./Exports/roc_HMM_cross_bandpower.csv', colClasses = rep('numeric', 2))
data_roc_hmm_bp <- read.csv('./Exports/roc_HMM_bandpower.csv', colClasses = rep('numeric', 2))
data_roc_hmm_entropy <- read.csv('./Exports/roc_HMM_entropy.csv', colClasses = rep('numeric', 2))
data_roc_hmm_eeg$classifier <- rep('HMM-EEG', nrow(data_roc_hmm_eeg))
data_roc_hmm_stats$classifier <- rep('HMM-stats', nrow(data_roc_hmm_stats))
data_roc_hmm_cross_bp$classifier <- rep('HMM-CSDBP', nrow(data_roc_hmm_cross_bp))
data_roc_hmm_bp$classifier <- rep('HMM-PSDBP', nrow(data_roc_hmm_bp))
data_roc_hmm_entropy$classifier <- rep('HMM-H', nrow(data_roc_hmm_entropy))
data_roc_hmm <- rbind(data_roc_hmm_svm, data_roc_hmm_eeg, data_roc_hmm_stats, data_roc_hmm_cross_bp, data_roc_hmm_bp, data_roc_hmm_entropy)
data_roc_hmm$classifier <- factor(data_roc_hmm$classifier,
                                  levels = c('HMM-EEG', 'HMM-stats', 'HMM-CSDBP', 'HMM-PSDBP', 'HMM-H', 'HMM-SVM'))
roc_hmm <- plot.roc(data_roc_hmm)

# 1578 * 526
panel.A <- plot_grid(strip.labs(roc, x = F, y = F), strip.labs(acc, x = F, y = F), strip.labs(auc, x = F, y = F), 
                     nrow = 1, ncol = 3)
plot_grid(panel.A, get_legend(acc + theme(legend.position = 'bottom')), nrow = 2, ncol = 1, rel_heights = c(1, .1))

# 1578 * 526
panel.A <- plot_grid(strip.labs(roc_hmm, x = F, y = F), strip.labs(acc_hmm, x = F, y = F), 
                     strip.labs(auc_hmm, x = F, y = F), nrow = 1, ncol = 3)
plot_grid(panel.A, get_legend(acc_hmm + theme(legend.position = 'bottom')), nrow = 2, ncol = 1, rel_heights = c(1, .1))

HMM_SVM_tune <- read.csv('./Exports/tune_HMM_SVM_S1.csv', colClasses = rep('numeric', 4))
plot.HMM.SVM.tune(HMM_SVM_tune)

round(data_summary[1,-1], 2)


