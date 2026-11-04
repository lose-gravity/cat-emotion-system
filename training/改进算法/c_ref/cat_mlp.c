/* cat_mlp.c - MLP推理, 权重从CSV运行时加载 */
#include "cat_mlp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- CSV读取辅助 ---- */
static double *read_csv_rows(const char *path, int *rows, int *cols) {
    FILE *f = fopen(path, "r");
    char line[65536];
    int nrows = 0, ncols = 0;
    double *data = NULL;
    if (!f) { fprintf(stderr, "无法打开: %s\n", path); return NULL; }
    if (!fgets(line, sizeof(line), f)) { fclose(f); return NULL; }
    /* 数第一行列数 */
    {
        char *p = line, *tok;
        int n = 0;
        while ((tok = strtok(p, ",\n\r")) != NULL) { n++; p = NULL; }
        ncols = n;
    }
    {
        /* 预读行数 */
        long pos = ftell(f);
        nrows = 1;
        while (fgets(line, sizeof(line), f)) nrows++;
        fseek(f, 0, SEEK_SET);
    }
    data = (double *)malloc((size_t)nrows * ncols * sizeof(double));
    {
        int r = 0, c;
        while (r < nrows && fgets(line, sizeof(line), f)) {
            char *p = line, *tok;
            for (c = 0; c < ncols; c++) {
                tok = strtok(p, ",\n\r");
                data[r * ncols + c] = tok ? atof(tok) : 0.0;
                p = NULL;
            }
            r++;
        }
    }
    fclose(f);
    *rows = nrows;
    *cols = ncols;
    return data;
}

/* ---- 按文件名前缀找CSV并确定维度 ---- */
#include <dirent.h>

static int load_by_prefix(const char *dir, const char *prefix,
                          double **out, int *dim1, int *dim2) {
    /* 在dir下找 prefix_*.csv, 从文件名解析 out x in */
    DIR *d = opendir(dir);
    struct dirent *e;
    char target[512] = "";
    if (!d) { fprintf(stderr, "目录不存在: %s\n", dir); return -1; }
    while ((e = readdir(d)) != NULL) {
        int plen = (int)strlen(prefix);
        if (strncmp(e->d_name, prefix, plen) == 0 &&
            strstr(e->d_name, ".csv")) {
            snprintf(target, sizeof(target), "%s/%s", dir, e->d_name);
            break;
        }
    }
    closedir(d);
    if (!target[0]) { fprintf(stderr, "未找到 %s_*.csv in %s\n", prefix, dir); return -1; }
    {
        const char *base = strrchr(target, '/');
        int a, b;
        if (sscanf(base ? base + 1 : target, "%*[^_]_%dx%d.csv", &a, &b) == 2) {
            *dim1 = a; *dim2 = b;
        } else {
            *dim1 = 0; *dim2 = 0;   /* 一维文件(偏置/标量) */
        }
    }
    {
        int rows, cols;
        double *d = read_csv_rows(target, &rows, &cols);
        if (!d) return -1;
        if (*dim1 == 0) {
            *dim1 = rows * cols;
            *dim2 = 1;
        }
        *out = d;
        return 0;
    }
}

/* ---- MLP ---- */
int mlp_load(MLP *m, const char *dir) {
    int h1 = 0, in = 0, h2 = 0, o3 = 0;
    if (load_by_prefix(dir, "W1", &m->W1, &h1, &in)) return -1;
    m->h1 = h1; m->in = in;
    if (load_by_prefix(dir, "b1", &m->b1, &h1, &in)) return -1;
    if (load_by_prefix(dir, "W2", &m->W2, &h2, &in)) return -1;
    m->h2 = h2;
    if (load_by_prefix(dir, "b2", &m->b2, &h2, &in)) return -1;
    if (load_by_prefix(dir, "W3", &m->W3, &o3, &in)) return -1;
    if (load_by_prefix(dir, "b3", &m->b3, &o3, &in)) return -1;
    if (load_by_prefix(dir, "scaler_mean", &m->mean, &in, &h2)) return -1;
    if (load_by_prefix(dir, "scaler_scale", &m->scale, &in, &h2)) return -1;
    printf("  加载 %s: in=%d, h1=%d, h2=%d\n", dir, m->in, m->h1, m->h2);
    return 0;
}

void mlp_forward(const MLP *m, const double *x, double logits[3], double prob[3]) {
    double *h1 = (double *)malloc((size_t)m->h1 * sizeof(double));
    double *h2 = (double *)malloc((size_t)m->h2 * sizeof(double));
    int i, j;
    /* x' = (x-mean)/scale -> W1 -> relu */
    for (j = 0; j < m->h1; j++) {
        double acc = m->b1[j];
        for (i = 0; i < m->in; i++)
            acc += ((x[i] - m->mean[i]) / m->scale[i]) * m->W1[j * m->in + i];
        h1[j] = acc > 0 ? acc : 0;
    }
    for (j = 0; j < m->h2; j++) {
        double acc = m->b2[j];
        for (i = 0; i < m->h1; i++)
            acc += h1[i] * m->W2[j * m->h1 + i];
        h2[j] = acc > 0 ? acc : 0;
    }
    for (j = 0; j < 3; j++) {
        double acc = m->b3[j];
        for (i = 0; i < m->h2; i++)
            acc += h2[i] * m->W3[j * m->h2 + i];
        logits[j] = acc;
    }
    {
        double mx = logits[0];
        double sum = 0;
        if (logits[1] > mx) mx = logits[1];
        if (logits[2] > mx) mx = logits[2];
        for (j = 0; j < 3; j++) {
            prob[j] = exp(logits[j] - mx);
            sum += prob[j];
        }
        for (j = 0; j < 3; j++) prob[j] /= sum;
    }
    free(h1);
    free(h2);
}

void mlp_free(MLP *m) {
    free(m->W1); free(m->b1); free(m->W2); free(m->b2);
    free(m->W3); free(m->b3); free(m->mean); free(m->scale);
}
